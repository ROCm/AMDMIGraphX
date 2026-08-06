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

#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

using migraphx::sym::var;

// dyn_topk has no GPU kernel because its output length is data-dependent, so lowering routes it
// to the host ref op. This checks that round trip: inputs copied off the device, the op run on
// the host, and both tuple elements copied back.
TEST_CASE(dyn_topk_gpu_host_fallback)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape data_s{migraphx::shape::float_type, {2, 4}};
    std::vector<float> data_values = {1, 3, 2, 4, 8, 5, 7, 6};
    auto data                      = mm->add_literal(migraphx::literal{data_s, data_values});
    migraphx::shape k_s{migraphx::shape::int64_type, {1}};
    auto k   = mm->add_parameter("k", k_s);
    auto out = mm->add_instruction(
        migraphx::make_op(
            "dyn_topk", {{"k", migraphx::to_value(var("k", {1, 4}))}, {"axis", 1}, {"largest", 1}}),
        data,
        k);
    auto val = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), out);
    auto ind = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), out);
    mm->add_return({val, ind});

    migraphx::target t = migraphx::make_target("gpu");
    p.compile(t);

    std::vector<int64_t> k_data = {2};
    migraphx::parameter_map params;
    for(auto&& x : p.get_parameter_shapes())
    {
        if(x.first == "k")
            params[x.first] = t.copy_to(migraphx::argument(k_s, k_data.data()));
        else
            params[x.first] = t.allocate(x.second);
    }

    auto results = p.eval(params);
    std::vector<float> val_v;
    std::vector<int64_t> ind_v;
    t.copy_from(results.at(0)).visit([&](auto o) { val_v.assign(o.begin(), o.end()); });
    t.copy_from(results.at(1)).visit([&](auto o) { ind_v.assign(o.begin(), o.end()); });

    EXPECT(migraphx::verify::verify_rms_range(val_v, std::vector<float>{4, 3, 8, 7}));
    EXPECT(ind_v == std::vector<int64_t>{3, 1, 0, 2});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
