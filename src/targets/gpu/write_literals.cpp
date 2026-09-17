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
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/env.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/liveness.hpp>
#include <migraphx/algorithm.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_COPY_LITERALS)
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
            const std::string key = ctx.get_current_device().get_gfx_name() + ":" +
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

static std::size_t
estimate_scratch_size(const module& m, std::size_t overhead_percent, std::size_t alignment = 32)
{
    std::size_t scratch_size = 0;
    liveness(m, [&](instruction_ref ins, const auto& live_set) {
        std::size_t n =
            transform_accumulate(live_set.begin(),
                                 live_set.end(),
                                 ins->get_shape().bytes(),
                                 std::plus<>{},
                                 [&](instruction_ref i) -> std::size_t {
                                     if(not is_allocate(i))
                                         return 0;
                                     auto b = (i->get_shape().bytes() + alignment - 1) / alignment;
                                     return b * alignment;
                                 });
        scratch_size = std::max(scratch_size, n);
    });
    // Pad the estimate by overhead_percent since memory coloring is NP-hard and liveness is
    // incomplete without the scheduler, so we might need more space
    return scratch_size + scratch_size * overhead_percent / 100;
}

static std::size_t get_total_literals(const module& m)
{
    return transform_accumulate(m.begin(),
                                m.end(),
                                std::size_t{0},
                                std::plus<>{},
                                [&](const instruction& ins) -> std::size_t {
                                    // each code obj takes 2mb of gpu memory
                                    if(ins.name() == "gpu::code_object")
                                        return 1024 * 1024 * 2;
                                    if(not contains({"@literal", "@param"}, ins.name()))
                                        return 0;
                                    return ins.get_shape().bytes();
                                });
}

static std::size_t get_max_literals(const module& m)
{
    return transform_accumulate(m.begin(),
                                m.end(),
                                std::size_t{0},
                                MIGRAPHX_LIFT(std::max),
                                [&](const instruction& ins) -> std::size_t {
                                    if(not contains({"@literal", "@param"}, ins.name()))
                                        return 0;
                                    return ins.get_shape().bytes();
                                });
}

static std::size_t get_total_memory(const module& m, std::size_t scratch_overhead_percent)
{
    return get_total_literals(m) + get_max_literals(m) * 2 +
           estimate_scratch_size(m, scratch_overhead_percent);
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

static std::unordered_set<instruction_ref> find_copy_literals(const module& m, std::ptrdiff_t n)
{
    std::unordered_set<instruction_ref> result;
    for(auto ins : reverse_iterator_for(m))
    {
        if(n <= 0)
            break;
        if(ins->name() != "@literal")
            continue;
        result.insert(ins);
        n -= std::ptrdiff_t(ins->get_shape().bytes());
    }
    return result;
}

void write_literals::apply(module& m) const
{
    // Sort module to get better liveness analysis
    m.sort();
    std::size_t available  = max_memory == 0 ? get_available_memory() : max_memory;
    std::size_t total_used = get_total_memory(m, scratch_overhead_percent);
    std::unordered_set<instruction_ref> copy_literals =
        find_copy_literals(m, extra_needed(available, total_used));

    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "@literal")
            continue;
        bool copy_literal = enabled(MIGRAPHX_COPY_LITERALS{}) or contains(copy_literals, ins);
        if(copy_literal)
        {
            auto lit = m.insert_instruction(
                ins, gpu_literal{.data = ins->get_literal().get_argument(), .host = true});
            auto a = m.insert_instruction(ins, hip_allocate{ins->get_literal().get_shape()});
            m.replace_instruction(ins, hip_copy{}, lit, a);
        }
        else
        {
            m.replace_instruction(ins, gpu_literal{ins->get_literal().get_argument()});
        }
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
