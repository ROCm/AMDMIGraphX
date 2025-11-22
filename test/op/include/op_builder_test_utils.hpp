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

#ifndef MIGRAPHX_GUARD_TEST_OPBUILDER_TEST_UTILS_HPP
#define MIGRAPHX_GUARD_TEST_OPBUILDER_TEST_UTILS_HPP

#include <migraphx/common.hpp>
#include <migraphx/instruction.hpp>
#include <test.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <migraphx/ranges.hpp>

inline migraphx::module make_op_module(const std::string& op_builder_name,
                                       const migraphx::value& options,
                                       const std::vector<migraphx::instruction_ref>& params)
{
    migraphx::module mm_op_built;

    for(auto param : migraphx::range(params.rbegin(), params.rend()))
    {
        auto param_name =
            migraphx::any_cast<migraphx::builtin::param>(param->get_operator()).parameter;
        mm_op_built.add_parameter(param_name, param->get_shape());
    }

    const auto& params2 = mm_op_built.get_parameters();
    const std::vector<migraphx::instruction_ref>& args2{params2.rbegin(), params2.rend()};
    migraphx::op::builder::add(op_builder_name, mm_op_built, args2, options);

    return mm_op_built;
}

inline migraphx::module make_op_module(const std::string& op_builder_name,
                                       const std::vector<migraphx::instruction_ref>& params)
{
    return make_op_module(op_builder_name, migraphx::value("", {}, false), params);
}

#endif
