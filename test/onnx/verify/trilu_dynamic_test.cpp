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

#include <migraphx/iterator_for.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>

static std::vector<float> eval_trilu(const migraphx::program& p,
                                     const std::vector<std::size_t>& lens)
{
    return gen_trilu_test({migraphx::shape::float_type, lens}, p);
}

static void expect_symbolic_output(const migraphx::program& p)
{
    const auto input = p.get_parameter_shapes().at("x");
    EXPECT(input.symbolic());
    EXPECT(p.get_output_shapes().back() == input);

    std::size_t ranges = 0;
    for(auto ins : iterator_for(*p.get_main_module()))
    {
        if(ins->name() == "dynamic_range")
        {
            ++ranges;
            EXPECT(ins->get_shape().symbolic());
        }
    }
    EXPECT(ranges == 2);
}

TEST_CASE(triu_symbolic_test)
{
    migraphx::onnx_options options;
    options.use_symbolic_shapes     = true;
    options.map_dyn_input_dims["x"] = {{1, 3}, {1, 4}};
    auto p                          = read_onnx("triu_test.onnx", options);
    expect_symbolic_output(p);
    p.compile(migraphx::make_target("ref"));

    EXPECT(eval_trilu(p, {2, 3}) == std::vector<float>{1, 2, 3, 0, 5, 6});
    EXPECT(eval_trilu(p, {3, 4}) == std::vector<float>{1, 2, 3, 4, 0, 6, 7, 8, 0, 0, 11, 12});
}

TEST_CASE(tril_symbolic_negative_k_test)
{
    migraphx::onnx_options options;
    options.use_symbolic_shapes     = true;
    options.map_dyn_input_dims["x"] = {{1, 3}, {1, 4}};
    auto p                          = read_onnx("tril_neg_k_test.onnx", options);
    expect_symbolic_output(p);
    p.compile(migraphx::make_target("ref"));

    EXPECT(eval_trilu(p, {2, 3}) == std::vector<float>{0, 0, 0, 4, 0, 0});
    EXPECT(eval_trilu(p, {3, 4}) == std::vector<float>{0, 0, 0, 0, 5, 0, 0, 0, 9, 10, 0, 0});
}

TEST_CASE(triu_symbolic_batch_nonzero_k_test)
{
    migraphx::onnx_options options;
    options.use_symbolic_shapes     = true;
    options.map_dyn_input_dims["x"] = {{1, 2}, {1, 3}, {1, 4}};
    auto p                          = read_onnx("triu_batch_diff_k_test.onnx", options);
    expect_symbolic_output(p);
    p.compile(migraphx::make_target("ref"));

    EXPECT(eval_trilu(p, {2, 2, 3}) == std::vector<float>{0, 0, 3, 0, 0, 0, 0, 0, 9, 0, 0, 0});
    EXPECT(eval_trilu(p, {1, 3, 4}) == std::vector<float>{0, 0, 3, 4, 0, 0, 0, 8, 0, 0, 0, 0});
}
