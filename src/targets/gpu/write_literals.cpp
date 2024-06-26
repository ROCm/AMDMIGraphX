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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_COPY_LITERALS)

void write_literals::apply(module& m) const
{
    assert(ctx != nullptr);
    std::size_t n = 0;
    std::size_t free;
    std::size_t total;
    std::size_t bytes_on_gpu = 0;
    std::size_t bytes_total  = 0;

    size_t scratch_size = 0;
    liveness(m, [&](auto ins, auto live_set) {
        if (ins->name() != "hip::allocate" || ins->get_shape().bytes() == 0) {
            return;
        } 
        size_t temp_size = 0;
        for (auto i : live_set) {
            if (i->name() != "hip::allocate" || i->get_shape().bytes() == 0) {
                continue;
            } 
            temp_size += i->get_shape().bytes();
        }

        if (temp_size > scratch_size) {
            scratch_size = temp_size;
        }

        // m.debug_print(ins);
        // std::cout << "Scratch size: " << temp_size << "\n\n";
    });

    // std::cout << "Max Scratch Size: " << scratch_size << "\n";

    // if(weight_streaming)
    // {
    //     std::cout << "Using weight streaming..." << std::endl;
    //     std::cout << "Streaming budget: " << streaming_budget << std::endl;
    // }

    // hipMemGetInfo(&free, &total);
    // std::cout << "Free: " << free << " Total: " << total << std::endl;

    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "@literal")
        {
            // hipMemGetInfo(&free, &total);
            // std::cout << "Free: " << free << " Total: " << total << std::endl;
            // std::cout << n << ": " << ins->get_shape().bytes() << " bytes" << std::endl;
            bytes_total += ins->get_shape().bytes();
            if(enabled(MIGRAPHX_COPY_LITERALS{}) ||
               (weight_streaming &&
                static_cast<long>(bytes_on_gpu + ins->get_shape().bytes()) >= scratch_size * 2))
            {
                literal l  = ins->get_literal();
                auto pre   = m.add_literal(l);
                // auto alloc = m.insert_instruction(std::next(pre), hip_allocate{l.get_shape()});
                // m.replace_instruction(ins, hip_copy_to_gpu{}, pre, alloc);
                m.replace_instruction(ins, hip_copy_to_gpu{}, pre);
            }
            else
            {
                bytes_on_gpu += ins->get_shape().bytes();
                std::string id = m.name() + ":@literal:" + std::to_string(n);
                m.replace_instruction(ins, hip_copy_literal{ins->get_literal(), id});
                n++;
            }

            // std::string id = m.name() + ":@literal:" + std::to_string(n);
            // m.replace_instruction(ins, hip_weight_streaming_literal{ins, ins->get_literal(), id, &m, false});
            // n++;
        }
    }

    // std::cout << "Free: " << free << " Total: " << total << std::endl;
    // std::cout << "Literal size on gpu (bytes): " << bytes_on_gpu << std::endl;
    // std::cout << "Total size of literals (bytes): " << bytes_total << std::endl;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
