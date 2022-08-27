/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_TEST_INCLUDE_POINTWISE_HPP
#define MIGRAPHX_GUARD_TEST_INCLUDE_POINTWISE_HPP

#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>

template <class F>
migraphx::instruction_ref add_pointwise(migraphx::program& p,
                                        const std::string& name,
                                        std::vector<migraphx::instruction_ref> inputs,
                                        F f)
{
    auto* pm = p.create_module(name);
    auto* mm = p.get_main_module();
    pm->set_bypass();
    std::vector<migraphx::instruction_ref> params;
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(params), [&](auto input) {
        return pm->add_parameter("x" + std::to_string(params.size()),
                                 migraphx::shape{input->get_shape().type()});
    });
    auto r = f(pm, params);
    pm->add_return({r});
    return mm->add_instruction(migraphx::make_op("pointwise"), inputs, {pm});
}

inline auto single_pointwise(const std::string& name)
{
    return [=](auto* pm, const auto& inputs) {
        return pm->add_instruction(migraphx::make_op(name), inputs);
    };
}

#endif // MIGRAPHX_GUARD_TEST_INCLUDE_POINTWISE_HPP
