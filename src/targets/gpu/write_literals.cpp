/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/gpu/write_literals.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/env.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/liveness.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_COPY_LITERALS)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_WRITE_LITERALS)

// Share one VRAM copy of a weight literal across programs with identical weights
// (e.g. an LLM's prefill and decode programs). Default off.
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_SHARE_LITERALS)

namespace {

// Sampling geometry for literal_fingerprint: 64 chunks of 4 KiB, so the key
// costs a bounded ~256 KiB of reads regardless of tensor size.
constexpr std::size_t fingerprint_chunk   = 4096;
constexpr std::size_t fingerprint_samples = 64;

// 64-bit mixer (murmur3 finalizer): full avalanche at 5 ops per 8 bytes.
inline std::uint64_t mix64(std::uint64_t x)
{
    x ^= x >> 33u;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33u;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33u;
    return x;
}

// Cheap content fingerprint, used only to pick a pool bucket. It is never a
// correctness gate: a candidate is always confirmed byte-for-byte with
// same_bytes() before anything is shared, so a fingerprint collision costs one
// failed compare and never an incorrect share.
//
// Hashing every byte is not affordable. Measured on DeepSeek-1L (2023 MiB of
// literals, gfx1201), a whole-tensor to_value(data).hash() key cost 3.1 s per
// program load. This samples head, tail and strided interior chunks instead, so
// the cost is ~O(1) in tensor size. The byte length is folded in first, so two
// tensors that agree on sampled bytes but differ in size cannot collide.
std::size_t literal_fingerprint(const argument& data)
{
    const auto* p       = reinterpret_cast<const unsigned char*>(data.data());
    const std::size_t n = data.get_shape().bytes();

    // Four independent accumulators: mix64's multiplies have several cycles of
    // latency, so one chained accumulator would serialise on them. Seeded
    // distinctly so they cannot collapse to a common state on uniform input.
    std::uint64_t a0 = 0x9e3779b97f4a7c15ULL ^ n;
    std::uint64_t a1 = 0xbf58476d1ce4e5b9ULL;
    std::uint64_t a2 = 0x94d049bb133111ebULL;
    std::uint64_t a3 = 0x2545f4914f6cdd1dULL;

    auto mix = [&](std::size_t off, std::size_t len) {
        const auto* q = p + off;
        std::size_t i = 0;
        for(; i + 32 <= len; i += 32)
        {
            // memcpy is the portable unaligned load; compilers emit a plain mov.
            std::uint64_t w0 = 0;
            std::uint64_t w1 = 0;
            std::uint64_t w2 = 0;
            std::uint64_t w3 = 0;
            std::memcpy(&w0, q + i, 8);
            std::memcpy(&w1, q + i + 8, 8);
            std::memcpy(&w2, q + i + 16, 8);
            std::memcpy(&w3, q + i + 24, 8);
            a0 = mix64(a0 ^ w0);
            a1 = mix64(a1 ^ w1);
            a2 = mix64(a2 ^ w2);
            a3 = mix64(a3 ^ w3);
        }
        std::uint64_t tail = len;
        for(std::size_t k = 0; i + k < len; ++k)
            tail = (tail << 8u) ^ q[i + k];
        a0 = mix64(a0 ^ tail);
    };

    if(n <= fingerprint_chunk * fingerprint_samples)
    {
        mix(0, n); // small enough that sampling would save nothing
    }
    else
    {
        mix(0, fingerprint_chunk);
        mix(n - fingerprint_chunk, fingerprint_chunk);
        for(std::size_t s = 1; s + 1 < fingerprint_samples; ++s)
            mix((n / fingerprint_samples) * s, fingerprint_chunk);
    }

    std::uint64_t h = n;
    for(std::uint64_t v : {a0, a1, a2, a3})
        h = mix64(h ^ v);
    return static_cast<std::size_t>(h);
}

// Exact byte equality. This is the share/no-share gate.
//
// argument::operator== is deliberately NOT used here. For floating-point types
// it compares element-wise with float_equal, which is a 1-ULP TOLERANCE check
// built on two std::nextafter calls per element. That is wrong for dedup -- two
// weights one ULP apart would compare equal and one would be silently
// substituted for the other -- and it is also ruinously slow: measured at 21.1 s
// of a 24.3 s per-load cost on DeepSeek-1L, roughly 400x a memcmp over the same
// bytes.
bool same_bytes(const argument& a, const argument& b)
{
    const auto& sa = a.get_shape();
    const auto& sb = b.get_shape();
    if(sa != sb)
        return false;
    return std::memcmp(a.data(), b.data(), sa.bytes()) == 0;
}

// Process- and device-scoped pool of uploaded weights. Each entry keeps the host
// literal (to confirm a candidate with same_bytes) and the device buffer, shared
// via argument::share() so N identical weights use 1 VRAM copy. Guarded by a
// mutex since finalize may run on multiple compile threads.
struct pooled_literal
{
    argument host;
    argument gpu;
};

struct shared_literal_pool
{
    std::mutex mtx;
    // A bucket per fingerprint. On a collision the new literal is APPENDED
    // rather than dropped: with a single entry per key a colliding weight could
    // never register, so every later copy of it would re-upload and sharing
    // would silently stop working for that weight.
    std::unordered_map<std::string, std::vector<pooled_literal>> buckets;

    static shared_literal_pool& instance()
    {
        static shared_literal_pool pool;
        return pool;
    }
};

} // namespace

struct gpu_literal
{
    argument data{};
    bool host = false;

    argument gpu_data{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.data, "data"), f(self.host, "host"));
    }

    std::string name() const { return "gpu::literal"; }

    shape compute_shape(const std::vector<shape>&) const { return data.get_shape(); }

    argument compute(const shape&, const std::vector<argument>&) const { return gpu_data; }

    void finalize(context& ctx, const shape&, const std::vector<shape>&)
    {
        // Dedup identical weights to one VRAM buffer. host-pinned literals are not
        // device VRAM so they are never pooled, and non-packed literals are skipped
        // because a raw-buffer compare is only meaningful when elements are
        // contiguous.
        if(enabled(MIGRAPHX_SHARE_LITERALS{}) and not host and data.get_shape().packed())
        {
            // Keyed by DEVICE, not just architecture. Two GPUs of the same model in
            // one process share a gfx name, so keying on that alone would hand a
            // buffer allocated on one device to a program running on another.
            const std::string key = ctx.get_current_device().get_gfx_name() + ":" +
                                    std::to_string(get_device_id()) + ":" +
                                    std::to_string(literal_fingerprint(data));
            auto& pool = shared_literal_pool::instance();
            std::lock_guard<std::mutex> lock(pool.mtx);
            auto& bucket = pool.buckets[key];
            for(const auto& entry : bucket)
            {
                if(same_bytes(entry.host, data))
                {
                    gpu_data = entry.gpu.share();
                    return;
                }
            }
            gpu_data = to_gpu(data);
            bucket.push_back(pooled_literal{data.share(), gpu_data.share()});
            return;
        }
        if(host)
            gpu_data = register_on_gpu(data);
        else
            gpu_data = to_gpu(data);
    }

    friend std::ostream& operator<<(std::ostream& os, const gpu_literal& x)
    {
        os << x.name();
        if(x.host)
            os << ":host";
        else
            os << ":gpu";
        return os;
    }
};
MIGRAPHX_REGISTER_OP(gpu_literal);

static bool is_allocate(instruction_ref ins)
{
    return contains({"hip::allocate", "allocate"}, ins->name());
}

// Scratch bytes an instruction occupies: only allocations count, since literals and parameters
// live outside the scratch buffer
static std::size_t allocation_bytes(instruction_ref ins, std::size_t alignment)
{
    if(not is_allocate(ins))
        return 0;
    auto b = (ins->get_shape().bytes() + alignment - 1) / alignment;
    return b * alignment;
}

// Pad the scratch estimate by overhead_percent since memory coloring is NP-hard and liveness is
// incomplete without the scheduler, so more space might be needed
static std::size_t pad_scratch(std::size_t bytes, std::size_t overhead_percent)
{
    return bytes + bytes * overhead_percent / 100;
}

// Code objects with the same module_id share one loaded module; unassigned ones load their own
static std::size_t estimate_code_object_size(const module& m)
{
    std::vector<std::int64_t> ids;
    transform_if(
        m.begin(),
        m.end(),
        std::back_inserter(ids),
        [](const instruction& ins) { return ins.name() == "gpu::code_object"; },
        [](const instruction& ins) {
            return any_cast<code_object_op>(ins.get_operator()).module_id;
        });
    ids.erase(distinct(ids.begin(),
                       ids.end(),
                       [](std::int64_t a, std::int64_t b) { return a >= 0 and a == b; }),
              ids.end());
    return ids.size() * code_module_bytes;
}

static std::size_t get_total_literals(const module& m)
{
    return transform_accumulate(m.begin(),
                                m.end(),
                                std::size_t{0},
                                std::plus<>{},
                                [&](const instruction& ins) -> std::size_t {
                                    if(not contains({"@literal", "@param"}, ins.name()))
                                        return 0;
                                    return ins.get_shape().bytes();
                                });
}

static std::size_t get_available_memory()
{
    std::size_t free_memory = 0;
    auto status             = hipMemGetInfo(&free_memory, nullptr);
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed to get GPU memory info: " + std::string(hipGetErrorString(status)));
    return free_memory;
}

static std::size_t extra_needed(std::size_t available, std::size_t used)
{
    if(available > used)
        return 0;
    return used - available;
}

static std::size_t literal_bytes(instruction_ref ins) { return ins->get_shape().bytes(); }

static std::size_t total_bytes(const std::unordered_set<instruction_ref>& literals)
{
    return transform_accumulate(
        literals.begin(), literals.end(), std::size_t{0}, std::plus<>{}, &literal_bytes);
}

// True when every use of lit in user is an input the kernel reads at index-selected positions
static bool is_indexed_use(instruction_ref lit, instruction_ref user)
{
    auto indexed =
        user->get_operator().attributes().get("indexed_inputs", std::vector<std::size_t>{});
    const auto& inputs = user->inputs();
    return all_of(range(inputs.size()),
                  [&](std::size_t i) { return inputs[i] != lit or contains(indexed, i); });
}

// A literal only read a few elements at a time can stay in host memory without a staging copy
static bool is_zero_copy_literal(instruction_ref ins)
{
    const auto& outputs = ins->outputs();
    return not outputs.empty() and
           all_of(outputs, [&](instruction_ref user) { return is_indexed_use(ins, user); });
}

struct literal_plan
{
    // Host literals the kernels read directly
    std::unordered_set<instruction_ref> zero_copy{};
    // Host literals copied into scratch each run
    std::unordered_set<instruction_ref> staged{};
    // Scratch the staged copies add to the estimate
    std::size_t scratch_growth = 0;
};

// Literals ins reads, through views and from its submodules; a view of a literal is not a read,
// its consumers are
static std::vector<instruction_ref> read_literals(instruction_ref ins,
                                                  const ins_dep_map& implicit_deps)
{
    auto inputs = ins->inputs();
    if(contains(implicit_deps, ins))
    {
        const auto& deps = implicit_deps.at(ins);
        inputs.insert(inputs.end(), deps.begin(), deps.end());
    }
    auto aliases = instruction::get_output_alias(ins);
    std::vector<instruction_ref> result;
    for(auto input : inputs)
    {
        copy_if(instruction::get_output_alias(input),
                std::back_inserter(result),
                [&](instruction_ref root) {
                    return root->name() == "@literal" and not contains(aliases, root);
                });
    }
    return result;
}

// Positions over which a staged copy of each literal is live: the scheduler sorts a copy next to
// its consumer, so that is from the literal's first to its last read
struct interval
{
    std::size_t first;
    std::size_t last;
};
static std::unordered_map<instruction_ref, interval> read_intervals(const module& m)
{
    auto implicit_deps = m.calc_implicit_deps();
    std::unordered_map<instruction_ref, interval> result;
    std::size_t pos = 0;
    for(auto ins : iterator_for(m))
    {
        // Positions ascend, so the first read sets first and each later read moves last
        for(auto lit : read_literals(ins, implicit_deps))
            result.try_emplace(lit, interval{pos, pos}).first->second.last = pos;
        pos++;
    }
    return result;
}

// Allocation bytes live at each instruction position
static std::vector<std::size_t> allocation_profile(const module& m, std::size_t alignment)
{
    std::unordered_map<instruction_ref, std::size_t> positions;
    for(auto ins : iterator_for(m))
        positions.emplace(ins, positions.size());
    // Liveness only visits instructions something reads; the others hold at most what the
    // preceding instruction does, so that is carried forward
    constexpr auto unknown = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> bytes(positions.size(), unknown);
    liveness(m, [&](instruction_ref ins, const auto& live_set) {
        // ins was just removed from the live set, so its own buffer is counted separately
        bytes[positions.at(ins)] =
            transform_accumulate(live_set.begin(),
                                 live_set.end(),
                                 allocation_bytes(ins, alignment),
                                 std::plus<>{},
                                 [&](instruction_ref i) { return allocation_bytes(i, alignment); });
    });
    std::partial_sum(bytes.begin(),
                     bytes.end(),
                     bytes.begin(),
                     [](std::size_t prev, std::size_t cur) { return cur == unknown ? prev : cur; });
    std::replace(bytes.begin(), bytes.end(), unknown, std::size_t{0});
    return bytes;
}

// Scratch at each instruction position: the allocations live there plus the staged copies, where
// copies read by the same instruction stack. A copy costs only what it raises the peak by, so it
// is free where the allocations leave room under the peak.
struct scratch_profile
{
    std::unordered_map<instruction_ref, interval> ranges;
    std::vector<std::size_t> bytes;
    // Peak before any copy is staged
    std::size_t base;
    // Bytes of the staged copies
    std::size_t staged = 0;

    explicit scratch_profile(const module& m, std::size_t alignment = 32)
        : ranges(read_intervals(m)), bytes(allocation_profile(m, alignment)), base(peak())
    {
    }

    template <class Op>
    void update(instruction_ref ins, Op op)
    {
        auto n  = literal_bytes(ins);
        staged  = op(staged, n);
        auto it = ranges.find(ins);
        // A literal never read occupies no scratch
        if(it == ranges.end())
            return;
        auto first = bytes.begin() + it->second.first;
        auto last  = bytes.begin() + it->second.last + 1;
        std::transform(first, last, first, [&](std::size_t b) { return op(b, n); });
    }
    void add(instruction_ref ins) { update(ins, std::plus<>{}); }
    void remove(instruction_ref ins) { update(ins, std::minus<>{}); }

    std::size_t peak() const
    {
        auto it = std::max_element(bytes.begin(), bytes.end());
        return it == bytes.end() ? 0 : *it;
    }
    // Scratch the staged copies add beyond the peak the allocations alone reach
    std::size_t growth() const { return peak() - base; }
    // Bytes the staged copies free net of the scratch they add
    std::size_t freed() const { return staged - growth(); }
};

// A staged copy occupies scratch while live, so staging is credited with the staged bytes less the
// scratch they add; zero-copy literals free their full size and are taken first. Within
// the staging peak, literals are taken largest-first (fewest copies), beyond it smallest-first
// (least peak growth); taken literals are then dropped, smallest first, while the rest still cover
// the need.
static literal_plan plan_literals(const module& m,
                                  scratch_profile profile,
                                  std::size_t needed,
                                  std::size_t staging_peak)
{
    literal_plan plan;
    if(needed == 0)
        return plan;

    std::vector<instruction_ref> literals;
    copy_if(iterator_for(m), std::back_inserter(literals), [](instruction_ref ins) {
        return ins->name() == "@literal";
    });
    auto staged_first =
        std::stable_partition(literals.begin(), literals.end(), &is_zero_copy_literal);
    std::stable_sort(literals.begin(), staged_first, by(std::greater<>{}, &literal_bytes));
    auto zero_copy_last = std::find_if(literals.begin(), staged_first, [&](instruction_ref ins) {
        if(needed == 0)
            return true;
        needed -= std::min(needed, literal_bytes(ins));
        return false;
    });
    plan.zero_copy.insert(literals.begin(), zero_copy_last);
    if(needed == 0)
        return plan;

    auto beyond = std::stable_partition(staged_first, literals.end(), [&](instruction_ref ins) {
        return literal_bytes(ins) <= staging_peak;
    });
    std::stable_sort(staged_first, beyond, by(std::greater<>{}, &literal_bytes));
    std::stable_sort(beyond, literals.end(), by(std::less<>{}, &literal_bytes));
    auto last = std::find_if(staged_first, literals.end(), [&](instruction_ref ins) {
        profile.add(ins);
        return profile.freed() >= needed;
    });
    if(last == literals.end())
    {
        // Even staging everything falls short, so stage it all to free the most possible
        plan.staged.insert(staged_first, literals.end());
        plan.scratch_growth = profile.growth();
        return plan;
    }
    ++last;
    // Dropping a copy that stacks on the peak costs little, so every taken literal is tried
    std::stable_sort(staged_first, last, by(std::less<>{}, &literal_bytes));
    std::copy_if(staged_first,
                 last,
                 std::inserter(plan.staged, plan.staged.end()),
                 [&](instruction_ref ins) {
                     profile.remove(ins);
                     if(profile.freed() >= needed)
                         return false;
                     profile.add(ins);
                     return true;
                 });
    plan.scratch_growth = profile.growth();
    return plan;
}

void write_literals::apply(module& m) const
{
    // Sort module to get better liveness analysis
    m.sort();
    scratch_profile profile{m};
    std::size_t available = max_memory == 0 ? get_available_memory() : max_memory;
    std::size_t scratch   = pad_scratch(profile.base, scratch_overhead_percent);
    std::size_t needed =
        extra_needed(available, get_total_literals(m) + estimate_code_object_size(m) + scratch);
    auto plan = plan_literals(m, profile, needed, needed * staging_peak_percent / 100);
    if(enabled(MIGRAPHX_TRACE_WRITE_LITERALS{}))
    {
        std::cout << "write_literals: available " << available << " bytes, literals "
                  << get_total_literals(m) << ", code objects " << estimate_code_object_size(m)
                  << ", scratch estimate " << scratch << ", shortfall " << needed << ", zero-copy "
                  << plan.zero_copy.size() << " (" << total_bytes(plan.zero_copy)
                  << " bytes), staged " << plan.staged.size() << " (" << total_bytes(plan.staged)
                  << " bytes, +" << plan.scratch_growth << " scratch)" << std::endl;
    }

    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "@literal")
            continue;
        auto data = ins->get_literal().get_argument();
        if(enabled(MIGRAPHX_COPY_LITERALS{}) or contains(plan.staged, ins))
        {
            auto lit = m.insert_instruction(ins, gpu_literal{.data = data, .host = true});
            auto a   = m.insert_instruction(ins, hip_allocate{data.get_shape()});
            m.replace_instruction(ins, hip_copy{}, lit, a);
        }
        else
        {
            m.replace_instruction(ins,
                                  gpu_literal{.data = data, .host = contains(plan.zero_copy, ins)});
        }
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
