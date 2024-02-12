/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/fuse_reduce.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>
#include <pointwise.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::fuse_reduce{}, migraphx::dead_code_elimination{}});
}

bool all_instructions_are_local(const migraphx::module& m)
{
    return std::all_of(m.begin(), m.end(), [&](const auto& ins) {
        return std::all_of(ins.inputs().begin(), ins.inputs().end(), [&](auto input) {
            return m.has_instruction(input);
        });
    });
}

template <class F>
migraphx::instruction_ref add_reduce(migraphx::program& p,
                                     const std::string& name,
                                     std::vector<migraphx::instruction_ref> inputs,
                                     const std::vector<int64_t>& axes,
                                     F f)
{
    auto* rm = p.create_module(name);
    auto* mm = p.get_main_module();
    rm->set_bypass();
    std::vector<migraphx::instruction_ref> params;
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(params), [&](auto input) {
        return rm->add_parameter(
            "x" + std::to_string(params.size()),
            migraphx::shape{input->get_shape().type(), input->get_shape().lens()});
    });
    auto r = f(rm, params, axes);
    rm->add_return({r});
    EXPECT(all_instructions_are_local(*rm));
    return mm->add_instruction(migraphx::make_op("fused_reduce", {{"axes", axes}}), inputs, {rm});
}

inline auto single_reduce(const std::string& name)
{
    return [=](auto* rm, const auto& inputs, const auto& axes) {
        return rm->add_instruction(migraphx::make_op(name, {{"axes", axes}}), inputs);
    };
}

TEST_CASE(single)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto rsum1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), y);
        mm->add_return({rsum1, rsum2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto rsum1 = add_reduce(p2, "main:reduce_sum0", {x}, {1}, single_reduce("reduce_sum"));
        auto rsum2 = add_reduce(p2, "main:reduce_sum1", {y}, {1}, single_reduce("reduce_sum"));
        mm->add_return({rsum1, rsum2});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(pointwise_reduce)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add  = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), add);
        mm->add_return({rsum});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto rsum = add_reduce(
            p2,
            "main:pointwise0:main:reduce_sum0",
            {x, y},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto add =
                    add_pointwise(p2, rm, "main:pointwise0", inputs, single_pointwise("add"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add);
            });
        mm->add_return({rsum});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_pointwise)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto add = add_pointwise(p1, "main:pointwise0", {rsumb, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto add = add_reduce(
            p2,
            "main:reduce_sum0:main:pointwise0",
            {x, y},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum  = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
                return add_pointwise(
                    p2, rm, "main:pointwise0", {rsumb, inputs[1]}, single_pointwise("add"));
            });
        mm->add_return({add});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_reduce)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto rsumdiff = add_pointwise(p1, "main:pointwise0", {rsumb, x}, single_pointwise("sub"));
        auto rsum2 =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), rsumdiff);
        auto sqrt = add_pointwise(p1, "main:pointwise1", {rsum2}, single_pointwise("sqrt"));
        mm->add_return({sqrt});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto sqrt = add_reduce(
            p2,
            "main:reduce_sum1:main:reduce_sum0:main:pointwise0:main:pointwise1",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum  = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
                auto rsumdiff = add_pointwise(
                    p2, rm, "main:pointwise0", {rsumb, inputs[0]}, single_pointwise("sub"));
                auto rsum2 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 rsumdiff);
                return add_pointwise(p2, rm, "main:pointwise1", {rsum2}, single_pointwise("sqrt"));
            });
        mm->add_return({sqrt});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_reduce_mismatch_axis)
{
    migraphx::shape s{migraphx::shape::float_type, {4, 2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), rsum1);
        mm->add_return({rsum2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum1 = add_reduce(p2, "main:reduce_sum0", {x}, {1}, single_reduce("reduce_sum"));
        auto rsum2 = add_reduce(p2, "main:reduce_sum1", {rsum1}, {2}, single_reduce("reduce_sum"));
        mm->add_return({rsum2});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(pointwise_reduce_broadcast)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto sqrt  = add_pointwise(p1, "main:pointwise0", {rsum1}, single_pointwise("sqrt"));
        auto sqrtb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), sqrt);
        auto add1  = add_pointwise(p1, "main:pointwise1", {sqrtb, x}, single_pointwise("add"));
        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), add1);
        auto add2  = add_pointwise(p1, "main:pointwise2", {rsum2, rsum1}, single_pointwise("add"));
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto add2 = add_reduce(
            p2,
            "main:pointwise0:main:pointwise1:main:reduce_sum1:main:pointwise2:main:reduce_sum0",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto sqrt =
                    add_pointwise(p2, rm, "main:pointwise0", {rsum1}, single_pointwise("sqrt"));
                auto sqrtb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), sqrt);
                auto add1 = add_pointwise(
                    p2, rm, "main:pointwise1", {sqrtb, inputs[0]}, single_pointwise("add"));
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add1);
                return add_pointwise(
                    p2, rm, "main:pointwise2", {rsum2, rsum1}, single_pointwise("add"));
            });
        mm->add_return({add2});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_reduce_broadcast)
{
    migraphx::shape s{migraphx::shape::float_type, {4, 2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum1 = add_reduce(p1, "test:reduce_sum0", {x}, {1}, single_reduce("reduce_sum"));
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
        auto add = add_reduce(
            p1,
            "test:reduce_sum1",
            {rsumb, x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto add2 =
                    add_pointwise(p1, rm, "test:pointwise0", inputs, single_pointwise("add"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add2);
            });
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = add_reduce(
            p2,
            "test:reduce_sum1:test:reduce_sum0",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
                auto add = add_pointwise(
                    p2, rm, "test:pointwise0", {rsumb, inputs[0]}, single_pointwise("add"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add);
            });
        mm->add_return({rsum});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_reshape_pointwise1)
{
    migraphx::shape s1{migraphx::shape::float_type, {64, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {8, 8, 2, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s1);
        auto y    = mm->add_parameter("y", s2);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
        auto rsumr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), rsumb);
        auto add = add_pointwise(p1, "main:pointwise0", {rsumr, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto xr  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), x);
        auto add = add_reduce(
            p2,
            "main:reduce_sum0_reshape:main:pointwise0",
            {xr, y},
            {2, 3},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), rsum);
                return add_pointwise(
                    p2, rm, "main:pointwise0", {rsumb, inputs[1]}, single_pointwise("add"));
            });
        mm->add_return({add});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_reshape_pointwise2)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 32, 40960}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 320, 64, 64}};
    migraphx::shape s3{migraphx::shape::float_type, {2, 32, 10, 64, 64}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s1);
        auto y     = mm->add_parameter("y", s2);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
        auto rsumr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), rsumb);
        auto add = add_pointwise(p1, "main:pointwise0", {rsumr, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto xr  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), x);
        auto yr  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), y);
        auto add = add_reduce(
            p2,
            "main:reduce_sum0_reshape:main:pointwise0",
            {xr, yr},
            {2, 3, 4},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum  = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), rsum);
                return add_pointwise(
                    p2, rm, "main:pointwise0", {rsumb, inputs[1]}, single_pointwise("add"));
            });
        auto addr = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), add);
        mm->add_return({addr});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reduce_reshape_reduce)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 32, 4096}};
    migraphx::shape s1r{migraphx::shape::float_type, {2, 32, 1}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 16, 64, 64}};
    migraphx::shape s2r{migraphx::shape::float_type, {4, 16, 1, 1}};
    migraphx::shape s3{migraphx::shape::float_type, {2, 2, 16, 64, 64}};
    migraphx::shape s3r{migraphx::shape::float_type, {2, 2, 16, 1, 1}};
    migraphx::program p1;
    {
        auto* mm       = p1.get_main_module();
        auto x1        = mm->add_parameter("x1", s1);
        auto x2        = mm->add_parameter("x2", s1r);
        auto y         = mm->add_parameter("y", s2);
        auto rsum1     = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x1);
        auto rsum1_add = add_pointwise(p1, "main:pointwise0", {rsum1, x2}, single_pointwise("add"));

        auto rsum1_addb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum1_add);
        auto rsum1_sub =
            add_pointwise(p1, "main:pointwise1", {rsum1_addb, x1}, single_pointwise("sub"));
        auto rsum1_subr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), rsum1_sub);
        auto rsum2 =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3}}}), rsum1_subr);
        auto rsum2b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), rsum2);
        auto rsum2_add = add_pointwise(p1, "main:pointwise2", {rsum2b, y}, single_pointwise("add"));
        mm->add_return({rsum2_add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x1  = mm->add_parameter("x1", s1);
        auto x2  = mm->add_parameter("x2", s1r);
        auto y   = mm->add_parameter("y", s2);
        auto x1r = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), x1);
        auto x2r = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3r.lens()}}), x2);
        auto freduce = add_reduce(
            p2,
            "main:pointwise1:main:reduce_sum1_reshape_reshape:main:reduce_sum0:main:pointwise0_"
            "reshape",
            {x1r, x2r},
            {3, 4},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto add   = add_pointwise(
                    p2, rm, "main:pointwise0", {rsum1, inputs[1]}, single_pointwise("add"));
                auto addb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), add);
                auto sub = add_pointwise(
                    p2, rm, "main:pointwise1", {addb, inputs[0]}, single_pointwise("sub"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), sub);
            });
        auto freducer =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2r.lens()}}), freduce);
        // TODO: Fuse the last add as well
        auto freducerb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), freducer);
        auto add = add_pointwise(p2, "main:pointwise2", {freducerb, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    EXPECT(p1.sort() == p2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
