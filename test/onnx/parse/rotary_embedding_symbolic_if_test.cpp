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

#include <onnx_test.hpp>
#include <algorithm>

TEST_CASE(rotary_embedding_symbolic_parse_test)
{
    auto sequence = migraphx::sym::var("sequence", {1, 2});

    migraphx::onnx_options options;
    options.use_symbolic_shapes = true;
    options.map_dyn_input_dims["input"] =
        sym_dims({migraphx::sym::lit(1), sequence, migraphx::sym::lit(18)});
    options.map_dyn_input_dims["pos_ids"] = sym_dims({migraphx::sym::lit(1), sequence});

    auto p = read_onnx("rotary_embedding_test.onnx", options);
    auto expected =
        migraphx::shape{migraphx::shape::half_type,
                        sym_dims({migraphx::sym::lit(1), sequence, migraphx::sym::lit(18)})};
    EXPECT(p.get_output_shapes().back() == expected);
    EXPECT(std::count_if(p.get_main_module()->begin(), p.get_main_module()->end(), [](auto ins) {
               return ins.name() == "gathernd";
           }) == 2);
    EXPECT(std::none_of(p.get_main_module()->begin(), p.get_main_module()->end(), [](auto ins) {
        return ins.name() == "reshape" and ins.get_shape().symbolic() and ins.inputs().size() == 1;
    }));
}

TEST_CASE(rotary_embedding_symbolic_if_test)
{
    auto sequence = migraphx::sym::var("sequence_length", {1, 2}, {2});

    migraphx::onnx_options options;
    options.use_symbolic_shapes = true;
    options.map_dyn_input_dims["input"] =
        sym_dims({migraphx::sym::lit(1), sequence, migraphx::sym::lit(18)});
    options.map_dyn_input_dims["pos_ids"] = sym_dims({migraphx::sym::lit(1), sequence});

    auto p = read_onnx("rotary_embedding_symbolic_if_test.onnx", options);
    auto expected_output =
        migraphx::shape{migraphx::shape::half_type,
                        sym_dims({migraphx::sym::lit(1), sequence, migraphx::sym::lit(18)})};
    EXPECT(p.get_output_shapes() == std::vector<migraphx::shape>{expected_output});

    bool found_symbolic_reshape = false;
    for(const auto* mod : p.get_modules())
    {
        for(const auto& ins : *mod)
        {
            if(ins.name() != "reshape" or not ins.get_shape().symbolic())
                continue;
            found_symbolic_reshape = true;
            EXPECT(ins.inputs().size() == 2);
            EXPECT(ins.get_operator().to_value().at("dims").empty());
        }
    }
    EXPECT(found_symbolic_reshape);
}
