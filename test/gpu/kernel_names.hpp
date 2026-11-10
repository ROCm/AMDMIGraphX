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
#ifndef MIGRAPHX_GUARD_TEST_GPU_KERNEL_NAMES_HPP
#define MIGRAPHX_GUARD_TEST_GPU_KERNEL_NAMES_HPP

#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <string>
#include <vector>

// The kernels launched by the program, without the output copies
inline std::vector<std::string> get_kernel_names(migraphx::program& p)
{
    std::vector<std::string> result;
    auto im = migraphx::iterator_for(*p.get_main_module());
    migraphx::transform_if(
        im.begin(),
        im.end(),
        std::back_inserter(result),
        [](auto ins) {
            if(ins->name() != "gpu::code_object")
                return false;
            const auto& op = migraphx::any_cast<migraphx::gpu::code_object_op>(ins->get_operator());
            return op.symbol_name != "hip_copy_kernel";
        },
        [](auto ins) {
            return migraphx::any_cast<migraphx::gpu::code_object_op>(ins->get_operator())
                .symbol_name;
        });
    return result;
}

#endif // MIGRAPHX_GUARD_TEST_GPU_KERNEL_NAMES_HPP
