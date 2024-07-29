/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/iterator_for.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/env.hpp>
#include <migraphx/liveness.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_COPY_LITERALS)

struct literal_as_argument
{
    std::string name() const { return "gpu::literal_as_argument"; }
    argument data;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.data, "data"));
    }

    shape compute_shape(const std::vector<shape>&) const { return data.get_shape(); }
    argument compute(context&, const shape&, const std::vector<argument>&) const { return data; }
    friend std::ostream& operator<<(std::ostream& os, const literal_as_argument& x)
    {
        os << x.name() << "[data=";
        if (x.compute_shape(std::vector<shape>{}).elements() < 10)
        {
            os << x.data;
        }
        else
        {
            os << "{ ... }";
        }
        os << "]";
        return os;
    }
};
MIGRAPHX_REGISTER_OP(literal_as_argument);

void write_literals::apply(module& m) const
{
    assert(ctx != nullptr);
    std::size_t n = 0;

    if(weight_streaming)
    {
        std::size_t bytes_on_gpu = 0;
        size_t scratch_size      = 0;
        liveness(m, [&](auto ins, auto live_set) {
            if(ins->name() != "hip::allocate" or ins->get_shape().bytes() == 0)
            {
                return;
            }
            size_t temp_size = 0;
            for(auto i : live_set)
            {
                if(i->name() != "hip::allocate" or i->get_shape().bytes() == 0)
                {
                    continue;
                }
                temp_size += i->get_shape().bytes();
            }

            if(temp_size > scratch_size)
            {
                scratch_size = temp_size;
            }
        });

        std::vector<instruction_ref> ins_list;
        size_t size_of_literals = 0;
        for(auto ins : iterator_for(m))
        {
            if(ins->name() == "@literal")
            {
                ins_list.push_back(ins);
                size_of_literals += ins->get_shape().bytes();
            }
        }

        long budget = streaming_budget;
        if(budget == LONG_MAX)
        {
            budget = static_cast<long>(size_of_literals / 4);
        }

        size_t free_memory = 0;
        auto status        = hipMemGetInfo(&free_memory, nullptr);

        std::cout << "Using weight streaming..."
                  << "\n"
                  << "Streaming budget: " << budget << "\n"
                  << "Scratch size: " << scratch_size << "\n"
                  << "Total size of literals: " << size_of_literals << "\n"
                  << "[Before] Free memory: " << free_memory << " Status: " << status << std::endl;

        // std::sort(ins_list.begin(),
        //           ins_list.end(),
        //           [](const instruction_ref& a, const instruction_ref& b) {
        //               return a->get_shape().bytes() > b->get_shape().bytes();
        //           });

        for(auto ins : ins_list)
        {
            if(bytes_on_gpu + ins->get_shape().bytes() > budget)
            {
                literal l  = ins->get_literal();
                // auto pre   = m.add_literal(l);
                auto pre   = m.insert_instruction(ins, literal_as_argument{l.get_argument()});
                auto alloc = m.insert_instruction(std::next(pre), hip_allocate{l.get_shape()});
                m.replace_instruction(ins, hip_copy_to_gpu{}, pre, alloc);
            }

            else
            {
                bytes_on_gpu += ins->get_shape().bytes();
                std::string id = m.name() + ":@literal:" + std::to_string(n);
                m.replace_instruction(ins, hip_copy_literal{ins->get_literal(), id});
                n++;
            }
        }
    }

    else
    {
        for(auto ins : iterator_for(m))
        {
            if(ins->name() == "@literal")
            {
                if(enabled(MIGRAPHX_COPY_LITERALS{}))
                {
                    literal l  = ins->get_literal();
                    auto pre   = m.add_literal(l);
                    auto alloc = m.insert_instruction(std::next(pre), hip_allocate{l.get_shape()});
                    m.replace_instruction(ins, hip_copy_to_gpu{}, pre, alloc);
                }
                else
                {
                    std::string id = m.name() + ":@literal:" + std::to_string(n);
                    m.replace_instruction(ins, hip_copy_literal{ins->get_literal(), id});
                    n++;
                }
            }
        }
    }

    size_t free_mem = 0;
    auto status     = hipMemGetInfo(&free_mem, nullptr);
    std::cout << "[After]  Free memory: " << free_mem << " status: " << status << std::endl;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
