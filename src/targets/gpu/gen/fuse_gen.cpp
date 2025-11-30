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
#include <migraphx/gpu/gen/fuse_gen.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace gen {

void fuse_gen::apply(module_pass_manager& mpm) const
{
    auto& m = mpm.get_module();

    for(auto ins : iterator_for(m))
    {
        // Find pointwise operations that can be compiled with gen IR
        if(ins->name() != "pointwise")
            continue;

        // Skip if no module inputs (malformed pointwise)
        if(ins->module_inputs().empty())
            continue;

        // Skip multi-output pointwise (tuple outputs) - not supported yet
        if(ins->get_shape().type() == shape::tuple_type)
            continue;

        // Skip if any input or output has broadcasted dimensions (stride 0) - vectorization issues
        auto has_broadcast = [](const shape& s) {
            const auto& strides = s.strides();
            return std::any_of(strides.begin(), strides.end(), [](auto st) { return st == 0; });
        };
        if(has_broadcast(ins->get_shape()))
            continue;
        bool skip = false;
        auto out_type = ins->get_shape().type();
        for(auto input : ins->inputs())
        {
            if(has_broadcast(input->get_shape()))
            {
                skip = true;
                break;
            }
            // Skip if input and output types differ (e.g., bit_cast)
            if(input->get_shape().type() != out_type)
            {
                skip = true;
                break;
            }
        }
        if(skip)
            continue;

        // Get inputs and create allocation for output
        auto inputs = ins->inputs();
        auto alloc  = m.insert_instruction(
            ins, make_op("allocate", {{"shape", to_value(ins->get_shape())}}));
        inputs.push_back(alloc);

        // Create gen::pointwise operation wrapped in precompile_op
        m.replace_instruction(
            ins,
            make_op("gpu::precompile_op", {{"op", to_value(make_op("gpu::gen::pointwise"))}}),
            inputs,
            ins->module_inputs());
    }
}

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
