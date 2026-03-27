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
#ifndef MIGRAPHX_GUARD_TEST_GPU_MAKE_GROUP_OP_HPP
#define MIGRAPHX_GUARD_TEST_GPU_MAKE_GROUP_OP_HPP

#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/ranges.hpp>

template <class F>
migraphx::instruction_ref add_group(migraphx::program& p,
                                    const std::string& name,
                                    const std::string& group_tag,
                                    std::vector<migraphx::instruction_ref> inputs,
                                    std::vector<std::string> arg_names,
                                    const F& f)
{
    assert(inputs.size() == arg_names.size() and "One interior parameter name given per input.");
    auto* mm = p.get_main_module();
    auto* pm = p.create_module(name);
    pm->set_bypass();
    std::vector<migraphx::instruction_ref> params;
    for(size_t i = 0, e = inputs.size(); i < e; ++i)
    {
        params.push_back(pm->add_parameter(arg_names[i], inputs[i]->get_shape().as_standard()));
    }
    auto r = f(pm, params);

    pm->add_return(r);
    return mm->add_instruction(migraphx::make_op("group", {{"tag", group_tag}}), inputs, {pm});
}

template <class F>
migraphx::instruction_ref add_group(migraphx::program& p,
                                    const std::string& name,
                                    const std::string& group_tag,
                                    std::vector<migraphx::instruction_ref> inputs,
                                    const F& f)
{
    std::vector<std::string> arg_names;
    migraphx::transform(migraphx::range(inputs.size()), std::back_inserter(arg_names), [&](auto i) {
        return migraphx::param_name(i);
    });
    return add_group(p, name, group_tag, std::move(inputs), std::move(arg_names), std::move(f));
}

#endif // MIGRAPHX_GUARD_TEST_GPU_MAKE_GROUP_OP_HPP
