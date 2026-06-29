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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_COPY_LITERALS)
// Shares a single VRAM copy of a weight literal across multiple compiled
// programs on the same device when their weight bytes are identical (for
// example, an LLM's prefill and decode programs). Default off: each program
// uploads its own copy as before.
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_SHARE_LITERALS)

namespace {
// FNV-1a 64-bit over the literal's raw host bytes, used as the content key. The
// hash only selects a candidate; the bytes are always compared before sharing
// (see same_bytes), so a hash collision never causes an incorrect share.
std::uint64_t literal_content_hash(const argument& data)
{
    const auto* p       = reinterpret_cast<const unsigned char*>(data.data());
    const std::size_t n = data.get_shape().bytes();
    std::uint64_t h     = 1469598103934665603ULL; // FNV offset basis
    for(std::size_t i = 0; i < n; ++i)
    {
        h ^= p[i];
        h *= 1099511628211ULL; // FNV prime
    }
    return h;
}

// Exact equality of two host literals (same byte length and identical bytes).
// Guards against FNV-1a hash collisions: two distinct weights that hash equal
// must not be aliased to the same device buffer.
bool same_bytes(const argument& a, const argument& b)
{
    const std::size_t n = a.get_shape().bytes();
    if(n != b.get_shape().bytes())
        return false;
    return std::memcmp(a.data(), b.data(), n) == 0;
}

// Process-lifetime, per-(device,content-hash) pool of already-uploaded weights.
// Each entry retains the host literal (for the collision byte-compare) and the
// device buffer; gpu_literals that opt in share the SAME device `argument`
// (shared_ptr-refcounted), so N identical weights => 1 VRAM copy. Refs are held
// for the process lifetime, which suits the co-residency use case (the weights
// stay live as long as the resident programs); a process that repeatedly loads
// and unloads distinct models would retain their weights until exit. Guarded by
// a mutex since finalize may run on multiple compile threads.
struct pooled_literal
{
    argument host; // host bytes, for the collision check
    argument gpu;  // device buffer, shared on a match
};
struct shared_literal_pool
{
    std::mutex mtx;
    std::unordered_map<std::string, pooled_literal> buffers; // key: "<gfx_name>:<hash>"

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
        // When sharing is enabled, dedup identical weight bytes to a single VRAM
        // buffer shared across programs. host-pinned literals are not device VRAM,
        // so they are never pooled (only the to_gpu/device path is).
        if(enabled(MIGRAPHX_SHARE_LITERALS{}) and not host)
        {
            // Namespace the pool per-device via the public gfx-name (device_id is
            // private on hip_device). Content-hash keys identical bytes together.
            const std::string key = ctx.get_current_device().get_gfx_name() + ":" +
                                    std::to_string(literal_content_hash(data));
            auto& pool = shared_literal_pool::instance();
            std::lock_guard<std::mutex> lock(pool.mtx);
            auto it = pool.buffers.find(key);
            if(it != pool.buffers.end() and same_bytes(it->second.host, data))
            {
                gpu_data = it->second.gpu.share(); // alias existing VRAM buffer (refcount++)
                return;
            }
            // Miss, or a hash collision with different bytes: upload a private copy.
            // On a genuine miss, register it so later identical weights can share.
            gpu_data = to_gpu(data);
            if(it == pool.buffers.end())
                pool.buffers.emplace(key, pooled_literal{data.share(), gpu_data.share()});
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
