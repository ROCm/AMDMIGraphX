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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

// `k` is a runtime input: feed both `data` and `k` at eval time to prove `k` is not baked in.
// Top-2 largest along axis 1.
TEST_CASE(topk_var_k_test)
{
    migraphx::program p = read_onnx("topk_var_k_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape ds{migraphx::shape::float_type, {2, 4}};
    std::vector<float> dd = {1, 3, 2, 4, 8, 5, 7, 6};
    migraphx::shape ks{migraphx::shape::int64_type, {1}};
    std::vector<int64_t> kd = {2};

    migraphx::parameter_map pp;
    pp["data"] = migraphx::argument(ds, dd.data());
    pp["k"]    = migraphx::argument(ks, kd.data());

    auto results = p.eval(pp);
    std::vector<float> val_v;
    std::vector<int64_t> ind_v;
    results[0].visit([&](auto o) { val_v.assign(o.begin(), o.end()); });
    results[1].visit([&](auto o) { ind_v.assign(o.begin(), o.end()); });

    std::vector<float> gold_val   = {4, 3, 8, 7};
    std::vector<int64_t> gold_ind = {3, 1, 0, 2};
    EXPECT(migraphx::verify::allclose(val_v, gold_val, migraphx::verify::tolerance{}));
    EXPECT(ind_v == gold_ind);
}

// Same model and runtime `k`, but `data` is parsed as a dynamic shape and a concrete shape within
// range is supplied at eval time.
TEST_CASE(topk_var_k_dynamic_test)
{
    migraphx::onnx_options options;
    options.map_dyn_input_dims["data"] = {{1, 4}, {2, 4}};
    migraphx::program p                = read_onnx("topk_var_k_test.onnx", options);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape ds{migraphx::shape::float_type, {2, 4}};
    std::vector<float> dd = {1, 3, 2, 4, 8, 5, 7, 6};
    migraphx::shape ks{migraphx::shape::int64_type, {1}};
    std::vector<int64_t> kd = {3};

    migraphx::parameter_map pp;
    pp["data"] = migraphx::argument(ds, dd.data());
    pp["k"]    = migraphx::argument(ks, kd.data());

    auto results = p.eval(pp);
    std::vector<float> val_v;
    std::vector<int64_t> ind_v;
    results[0].visit([&](auto o) { val_v.assign(o.begin(), o.end()); });
    results[1].visit([&](auto o) { ind_v.assign(o.begin(), o.end()); });

    EXPECT(migraphx::verify::allclose(
        val_v, std::vector<float>{4, 3, 2, 8, 7, 6}, migraphx::verify::tolerance{}));
    EXPECT(ind_v == std::vector<int64_t>{3, 1, 2, 0, 2, 3});
}
