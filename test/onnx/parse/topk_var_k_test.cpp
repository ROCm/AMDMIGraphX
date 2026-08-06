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

// `k` is a runtime input (graph input, not an initializer), so the parser emits dyn_topk with
// the runtime `k` named by a symbol bounded by the axis length.
static void add_dyn_topk(migraphx::module& m, const std::vector<migraphx::instruction_ref>& args)
{
    auto k_var = migraphx::sym::var("TopK_2", {1, 4});
    auto out   = m.add_instruction(
        migraphx::make_op("dyn_topk",
                            {{"k", migraphx::to_value(k_var)}, {"axis", 1}, {"largest", 1}}),
        args[0],
        args[1]);
    auto val = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), out);
    auto ind = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), out);
    m.add_return({val, ind});
}

TEST_CASE(topk_var_k_test)
{
    EXPECT(check_parse("topk_var_k_test.onnx",
                       {{"data", {migraphx::shape::float_type, {2, 4}}},
                        {"k", {migraphx::shape::int64_type, {1}}}},
                       add_dyn_topk));
}

// Same model, but `data` is overridden to a symbolic shape. The `k` symbol is still bounded by
// the axis length, which comes from the symbol's upper bound rather than a fixed length.
TEST_CASE(topk_var_k_symbolic_test)
{
    using migraphx::sym::var;
    EXPECT(check_parse(
        "topk_var_k_test.onnx",
        {{"data", {migraphx::shape::float_type, sym_dims({var("n", {1, 4}), var("m", {2, 4})})}},
         {"k", {migraphx::shape::int64_type, {1}}}},
        add_dyn_topk));
}
