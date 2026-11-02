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

TEST_CASE(roialign_dynamic_test)
{
    EXPECT(check_parse(
        "roialign_dynamic_test.onnx",
        {{"x", {migraphx::shape::float_type, {2, 5, 4, 7}}},
         {"rois", {migraphx::shape::float_type, {{0, 8}, {4, 4}}}},
         {"batch_ind",
          {migraphx::shape::int64_type, std::vector<migraphx::shape::dynamic_dimension>{{0, 8}}}}},
        [](migraphx::module& m, const auto& args) {
            auto r = m.add_instruction(
                migraphx::make_op("roialign",
                                  {{"output_height", int64_t{3}}, {"output_width", int64_t{2}}}),
                args);
            m.add_return({r});
        }));
}

TEST_CASE(roialign_symbolic_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;
    auto num_rois = var("num_rois", {0, 8});

    EXPECT(check_parse("roialign_dynamic_test.onnx",
                       {{"x", {migraphx::shape::float_type, {2, 5, 4, 7}}},
                        {"rois", {migraphx::shape::float_type, sym_dims({num_rois, lit(4)})}},
                        {"batch_ind", {migraphx::shape::int64_type, sym_dims({num_rois})}}},
                       [](migraphx::module& m, const auto& args) {
                           auto r =
                               m.add_instruction(migraphx::make_op("roialign",
                                                                   {{"output_height", int64_t{3}},
                                                                    {"output_width", int64_t{2}}}),
                                                 args);
                           m.add_return({r});
                       }));
}
