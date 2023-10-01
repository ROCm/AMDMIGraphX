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
#include <migraphx/rewrite_quantization.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <test.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/env.hpp>

#include <migraphx/serialize.hpp>
#include <migraphx/pass_manager.hpp>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_CK);

bool is_quantizelinear(migraphx::instruction& ins) { return ins.name() == "quantizelinear"; }
bool is_dequantizelinear(migraphx::instruction& ins) { return ins.name() == "dequantizelinear"; }
bool is_clip_scalar(migraphx::instruction& ins)
{
    if(ins.name() == "clip")
    {
        assert(ins.inputs().size() > 1);
        return (std::all_of(ins.inputs().begin() + 1, ins.inputs().end(), [](auto input) {
            return input->get_shape().scalar();
        }));
    }
    return false;
}

void run_pass(migraphx::module& m) { migraphx::run_passes(m, {migraphx::rewrite_quantization{}}); }

migraphx::argument eval(const migraphx::program& p)
{
    auto r = p.eval({});
    EXPECT(r.size() == 1);
    return r.front();
}

TEST_CASE(quantizelinear)
{

    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> xv = {-300, 200, 129, 1, 2, 3, 500, 1000, 50};
    migraphx::shape ss{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> sv = {2, 2, 2, 2, 2, 2, 2, 2, 2};
    auto create_program   = [&]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_literal(xs, xv);
        auto s   = mm->add_literal(ss, sv);
        mm->add_instruction(migraphx::make_op("quantizelinear"), x, s);
        return p;
    };

    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();

    run_pass(*p2.get_main_module());
    EXPECT(eval(p1) == eval(p2));
    EXPECT(any_of(*p1.get_main_module(), &is_quantizelinear));
    EXPECT(none_of(*p2.get_main_module(), &is_quantizelinear));
    // ensure clip literals created in quantized program are scalar
    EXPECT(migraphx::enabled(MIGRAPHX_ENABLE_CK{}) xor any_of(*p2.get_main_module(), &is_clip_scalar));
}

TEST_CASE(dequantizelinear)
{

    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> xv = {0, 1, 2, 5, 10, 50, 100, 150, 250};
    migraphx::shape ss{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> sv = {2, 2, 2, 2, 2, 2, 2, 2, 2};
    migraphx::shape zs{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> zv = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    auto create_program   = [&]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_literal(xs, xv);
        auto s   = mm->add_literal(ss, sv);
        auto z   = mm->add_literal(zs, zv);
        mm->add_instruction(migraphx::make_op("dequantizelinear"), x, s, z);
        return p;
    };

    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();

    run_pass(*p2.get_main_module());
    EXPECT(eval(p1) == eval(p2));
    EXPECT(any_of(*p1.get_main_module(), &is_dequantizelinear));
    EXPECT(none_of(*p2.get_main_module(), &is_dequantizelinear));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
