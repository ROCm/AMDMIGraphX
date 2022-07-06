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

#include "migraphx/fpga/vitis_ai_adapter.hpp"

#include "migraphx/module.hpp"

#include "migraphx/stringutils.hpp"
namespace vitis_ai {

migraphx::shape XModel::get_shape() const { return shape_; };

void XModel::set_shape(migraphx::shape shape) { shape_ = shape; }

XModel create_xmodel(migraphx::module_ref mod)
{
    std::cout << "Calling an external function: create_xmodel!\n";
    XModel xmodel;
    xmodel.set_shape(std::prev(mod->end())->get_shape());
    return xmodel;
}

migraphx::argument
execute(XModel xmodel, const migraphx::shape& output_shape, std::vector<migraphx::argument>& args)
{
    (void)xmodel;

    std::cout << "Calling an external function: execute!\n";

    std::cout << "Output Shape: " << output_shape << std::endl;
    std::cout << "Args: " << args.size() << std::endl;
    for(const auto& arg : args)
    {
        std::cout << "  " << arg.get_shape() << std::endl;
    }
    std::cout << std::endl;

    migraphx::argument result{output_shape};

    return result;
}

} // namespace vitis_ai
