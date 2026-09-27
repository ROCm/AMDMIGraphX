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
#include <migraphx/fuse_topk.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>
#include <reduce.hpp>
#include <pointwise.hpp>

static void run_pass(migraphx::program& p, migraphx::fuse_topk pass = {})
{
    migraphx::run_passes(p, {pass, migraphx::dead_code_elimination{}});
}

// x / sum(x) over the axes, keeping the shape of x
static auto normalize(migraphx::program& p, const std::string& pointwise_name)
{
    return [&p, pointwise_name](auto* rm, const auto& inputs, const auto& axes) {
        auto x     = inputs[0];
        auto rsum  = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), x);
        auto rsumb = rm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x->get_shape().lens()}}), rsum);
        return add_pointwise(p, rm, pointwise_name, {x, rsumb}, single_pointwise("div"));
    };
}

// (x / sum(x)) * y
static auto normalize_scale(migraphx::program& p, const std::string& pointwise_name)
{
    return [&p, pointwise_name](auto* rm, const auto& inputs, const auto& axes) {
        auto x     = inputs[0];
        auto rsum  = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), x);
        auto rsumb = rm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x->get_shape().lens()}}), rsum);
        return add_pointwise(
            p, rm, pointwise_name, {x, rsumb, inputs[1]}, [](auto* pm, const auto& xs) {
                auto div = pm->add_instruction(migraphx::make_op("div"), xs[0], xs[1]);
                return pm->add_instruction(migraphx::make_op("mul"), div, xs[2]);
            });
    };
}

// sum(float(x))
static auto convert_sum(migraphx::program& p, const std::string& pointwise_name)
{
    return [&p, pointwise_name](auto* rm, const auto& inputs, const auto& axes) {
        auto c = add_pointwise(p, rm, pointwise_name, {inputs[0]}, [](auto* pm, const auto& xs) {
            return pm->add_instruction(
                migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}),
                xs[0]);
        });
        return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), c);
    };
}

static std::vector<migraphx::instruction_ref>
add_topk(migraphx::module_ref m, migraphx::instruction_ref x, std::size_t k)
{
    auto topk    = m->add_instruction(migraphx::make_op("topk", {{"axis", 1}, {"k", k}}), x);
    auto values  = m->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), topk);
    auto indices = m->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), topk);
    return {values, indices};
}

TEST_CASE(reduce_topk)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto norm = add_reduce(p1, "main:reduce0", {x}, {1}, normalize(p1, "main:pointwise0"));
        mm->add_return(add_topk(mm, norm, 2));
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto topk = add_reduce(
            p2, "main:reduce0:topk", {x}, {1}, [&](auto* rm, const auto& inputs, const auto& axes) {
                auto norm = normalize(p2, "main:pointwise0")(rm, inputs, axes);
                return add_topk(rm, norm, 2);
            });
        auto values =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), topk);
        auto indices =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), topk);
        mm->add_return({values, indices});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(topk_reduce)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto outs = add_topk(mm, x, 2);
        auto norm =
            add_reduce(p1, "main:reduce0", {outs[0]}, {1}, normalize(p1, "main:pointwise0"));
        mm->add_return({norm, outs[1]});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto fused = add_reduce(p2,
                                "main:topk:main:reduce0",
                                {x},
                                {1},
                                [&](auto* rm, const auto& inputs, const auto& axes) {
                                    auto outs = add_topk(rm, inputs[0], 2);
                                    auto norm = normalize(p2, "main:pointwise0")(
                                        rm, std::vector<migraphx::instruction_ref>{outs[0]}, axes);
                                    return std::vector<migraphx::instruction_ref>{norm, outs[1]};
                                });
        auto indices =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto values =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        mm->add_return({values, indices});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_topk_reduce)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto norm1 = add_reduce(p1, "main:reduce0", {x}, {1}, normalize(p1, "main:pointwise0"));
        auto outs  = add_topk(mm, norm1, 2);
        auto norm2 =
            add_reduce(p1, "main:reduce1", {outs[0]}, {1}, normalize(p1, "main:pointwise1"));
        mm->add_return({norm2, outs[1]});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto fused = add_reduce(p2,
                                "main:reduce0:topk:main:reduce1",
                                {x},
                                {1},
                                [&](auto* rm, const auto& inputs, const auto& axes) {
                                    auto norm1 = normalize(p2, "main:pointwise0")(rm, inputs, axes);
                                    auto outs  = add_topk(rm, norm1, 2);
                                    auto norm2 = normalize(p2, "main:pointwise1")(
                                        rm, std::vector<migraphx::instruction_ref>{outs[0]}, axes);
                                    return std::vector<migraphx::instruction_ref>{norm2, outs[1]};
                                });
        auto indices =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto values =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        mm->add_return({values, indices});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(topk_reduce_indices)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto outs = add_topk(mm, x, 2);
        auto rsum =
            add_reduce(p1, "main:reduce0", {outs[1]}, {1}, convert_sum(p1, "main:pointwise0"));
        mm->add_return({rsum, outs[0]});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto fused = add_reduce(p2,
                                "main:topk:main:reduce0",
                                {x},
                                {1},
                                [&](auto* rm, const auto& inputs, const auto& axes) {
                                    auto outs = add_topk(rm, inputs[0], 2);
                                    auto rsum = convert_sum(p2, "main:pointwise0")(
                                        rm, std::vector<migraphx::instruction_ref>{outs[1]}, axes);
                                    return std::vector<migraphx::instruction_ref>{rsum, outs[0]};
                                });
        auto values =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto rsum = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        mm->add_return({rsum, values});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(topk_reduce_extra_input)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};
    migraphx::shape ks{migraphx::shape::float_type, {2, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", ks);
        auto outs = add_topk(mm, x, 2);
        auto norm = add_reduce(
            p1, "main:reduce0", {outs[0], y}, {1}, normalize_scale(p1, "main:pointwise0"));
        mm->add_return({norm, outs[1]});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", ks);
        auto fused = add_reduce(
            p2,
            "main:topk:main:reduce0",
            {x, y},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto outs = add_topk(rm, inputs[0], 2);
                auto norm = normalize_scale(p2, "main:pointwise0")(
                    rm, std::vector<migraphx::instruction_ref>{outs[0], inputs[1]}, axes);
                return std::vector<migraphx::instruction_ref>{norm, outs[1]};
            });
        auto indices =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto values =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        mm->add_return({values, indices});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(topk_reduce_extra_input_after_topk)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};
    migraphx::shape ks{migraphx::shape::float_type, {2, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", ks);
        auto outs = add_topk(mm, x, 2);
        auto z    = add_pointwise(p1, "main:pointwise1", {y}, single_pointwise("relu"));
        auto norm = add_reduce(
            p1, "main:reduce0", {outs[0], z}, {1}, normalize_scale(p1, "main:pointwise0"));
        mm->add_return({norm, outs[1]});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

TEST_CASE(topk_reduce_values_used_twice)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto outs = add_topk(mm, x, 2);
        auto norm =
            add_reduce(p1, "main:reduce0", {outs[0]}, {1}, normalize(p1, "main:pointwise0"));
        mm->add_return({norm, outs[0], outs[1]});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_topk_used_twice)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto norm = add_reduce(p1, "main:reduce0", {x}, {1}, normalize(p1, "main:pointwise0"));
        auto outs = add_topk(mm, norm, 2);
        mm->add_return({outs[0], outs[1], norm});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_topk_axis_mismatch)
{
    migraphx::shape s{migraphx::shape::float_type, {8, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto norm = add_reduce(p1, "main:reduce0", {x}, {0}, normalize(p1, "main:pointwise0"));
        mm->add_return(add_topk(mm, norm, 2));
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_topk_with_indices_input)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};
    migraphx::shape is{migraphx::shape::int64_type, {2, 8}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto idx  = mm->add_parameter("idx", is);
        auto norm = add_reduce(p1, "main:reduce0", {x}, {1}, normalize(p1, "main:pointwise0"));
        auto topk =
            mm->add_instruction(migraphx::make_op("topk", {{"axis", 1}, {"k", 2}}), norm, idx);
        auto values =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), topk);
        auto indices =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), topk);
        mm->add_return({values, indices});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_topk_max_size)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto norm = add_reduce(p1, "main:reduce0", {x}, {1}, normalize(p1, "main:pointwise0"));
        mm->add_return(add_topk(mm, norm, 2));
    }
    migraphx::program p2 = p1;
    run_pass(p1, migraphx::fuse_topk{.max_size = 4});
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
