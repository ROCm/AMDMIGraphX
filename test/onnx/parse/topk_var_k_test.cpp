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

#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <onnx_test.hpp>

#include <algorithm>

// `k` is a runtime input (graph input, not an initializer), so the parser takes the var_k
// path: topk runs over the whole axis, then dyn_slice trims both outputs down to the runtime
// `k`, which the output shape carries as a symbol.
TEST_CASE(topk_var_k_test)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto data = mm->add_parameter("data", {migraphx::shape::float_type, {2, 4}});
    auto k    = mm->add_parameter("k", {migraphx::shape::int64_type, {1}});
    auto zero = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto out  = mm->add_instruction(
        migraphx::make_op("topk", {{"k", 4}, {"axis", 1}, {"largest", 1}}), data);
    auto val   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), out);
    auto ind   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), out);
    auto k_var = migraphx::sym::var("main_TopK_2", {0, 4});
    auto ds    = migraphx::make_op("dyn_slice",
                                   {{"axes", {1}},
                                    {"starts", {0}},
                                    {"ends", migraphx::value::array{migraphx::to_value(k_var)}},
                                    {"always_leq", true}});
    val        = mm->add_instruction(ds, val, zero, k);
    ind        = mm->add_instruction(ds, ind, zero, k);
    mm->add_return({val, ind});

    auto prog = read_onnx("topk_var_k_test.onnx");

    EXPECT(p == prog);
}

// Same model with `data` overridden to a symbolic shape. `k` stays a runtime input, so the
// var_k path still fires and sets the topk `k` to the axis dimension's max length.
TEST_CASE(topk_var_k_symbolic_test)
{
    using migraphx::sym::var;
    auto dims = [] { return sym_dims({var("n", {1, 4}), var("m", {2, 4})}); };

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto data = mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, dims()});
    auto k    = mm->add_parameter("k", {migraphx::shape::int64_type, {1}});
    auto zero = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto out  = mm->add_instruction(
        migraphx::make_op("topk", {{"k", 4}, {"axis", 1}, {"largest", 1}}), data);
    auto val   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), out);
    auto ind   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), out);
    auto k_var = migraphx::sym::var("main_TopK_2", {0, 4});
    auto ds    = migraphx::make_op("dyn_slice",
                                   {{"axes", {1}},
                                    {"starts", {0}},
                                    {"ends", migraphx::value::array{migraphx::to_value(k_var)}},
                                    {"always_leq", true}});
    val        = mm->add_instruction(ds, val, zero, k);
    ind        = mm->add_instruction(ds, ind, zero, k);
    mm->add_return({val, ind});

    migraphx::onnx_options options;
    options.use_symbolic_shapes        = true;
    options.map_dyn_input_dims["data"] = dims();
    auto prog                          = read_onnx("topk_var_k_test.onnx", options);

    EXPECT(p == prog);
}

TEST_CASE(topk_bounded_var_k_symbolic_test)
{
    using migraphx::sym::var;
    auto dims = [] { return sym_dims({var("n", {2, 2}), var("m", {1, 1000})}); };

    migraphx::onnx_options options;
    options.use_symbolic_shapes        = true;
    options.map_dyn_input_dims["data"] = dims();
    auto prog                          = read_onnx("topk_bounded_var_k_test.onnx", options);
    auto& mm                           = *prog.get_main_module();

    auto topks = migraphx::find_all(migraphx::iterator_for(mm),
                                    [](const auto& ins) { return ins->name() == "topk"; });
    EXPECT(topks.size() == 1);
    if(topks.size() == 1)
        EXPECT(topks.front()->get_operator().to_value().at("k").to<int64_t>() == 200);

    auto slices = migraphx::find_all(migraphx::iterator_for(mm),
                                     [](const auto& ins) { return ins->name() == "dyn_slice"; });
    EXPECT(slices.size() == 2);
    EXPECT(std::all_of(slices.begin(), slices.end(), [](auto slice) {
        const auto& output = slice->get_shape();
        if(not output.symbolic() or output.dyn_dims().at(1).get_interval().max != 200 or
           slice->inputs().size() != 3)
            return false;
        auto runtime_k = slice->inputs().at(2)->sym_eval();
        return not runtime_k.empty() and runtime_k.get().size() == 1 and
               runtime_k.get()[0].eval_interval_default().max ==
                   migraphx::sym::scalar{int64_t{200}};
    }));
}

// A range-based dynamic shape has no symbol to slice against, so the runtime `k` is rejected.
TEST_CASE(topk_var_k_range_dynamic_error_test)
{
    migraphx::onnx_options options;
    options.map_dyn_input_dims["data"] = {{1, 4}, {2, 4}};

    EXPECT(test::throws([&] { read_onnx("topk_var_k_test.onnx", options); }));
}
