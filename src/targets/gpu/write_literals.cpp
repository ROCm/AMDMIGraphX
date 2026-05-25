/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <cstddef>
#include <migraphx/gpu/write_literals.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/env.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/liveness.hpp>
#include <migraphx/algorithm.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_COPY_LITERALS)

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

    void finalize(context&, const shape&, const std::vector<shape>&)
    {
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

static std::size_t estimate_scratch_size(const module& m, std::size_t alignment = 32)
{
    std::size_t scratch_size = 0;
    liveness(m, [&](instruction_ref ins, auto live_set) {
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
    // Add 50% since memory coloring is NP-hard and liveness is incomplete without the scheduler, so
    // we might need more space
    return scratch_size + scratch_size / 2;
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

static std::size_t get_total_memory(const module& m)
{
    return get_total_literals(m) + get_max_literals(m) * 2 + estimate_scratch_size(m);
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
    auto rp = reverse(m);
    for(auto rins : iterator_for(rp))
    {
        if(n <= 0)
            break;
        // The base iterator is one ahead, so we need to use the previous iterator
        auto ins = std::prev(rins.base());
        if(ins->name() != "@literal")
            continue;
        result.insert(ins);
        n -= ins->get_shape().bytes();
    }
    return result;
}

void write_literals::apply(module& m) const
{
    // Sort module to get better liveness analysis
    m.sort();
    std::size_t available  = max_memory == 0 ? get_available_memory() : max_memory;
    std::size_t total_used = get_total_memory(m);
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
