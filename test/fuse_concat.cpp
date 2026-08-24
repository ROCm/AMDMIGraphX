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
#include <migraphx/fuse_concat.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/functional.hpp>

#include <test.hpp>
#include <pointwise.hpp>

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::fuse_concat{}, migraphx::dead_code_elimination{}});
}

template <class F>
struct concat_arg
{
    std::string name;
    std::vector<migraphx::instruction_ref> inputs;
    F f;
};

template <class F>
static concat_arg<F> arg(std::string name, std::vector<migraphx::instruction_ref> inputs, F f)
{
    return {std::move(name), std::move(inputs), std::move(f)};
}

template <class Arg, class... Args>
static migraphx::instruction_ref
add_pointwise_concat(migraphx::program& p, std::size_t axis, Arg post_arg, const Args&... args)
{
    std::vector<migraphx::module_ref> module_inputs;
    std::vector<migraphx::instruction_ref> ins_inputs;
    migraphx::each_args(
        [&](auto arg) {
            module_inputs.push_back(create_pointwise_module(p, arg.name, arg.inputs, arg.f));
            ins_inputs.insert(ins_inputs.end(), arg.inputs.begin(), arg.inputs.end());
        },
        args...);
    module_inputs.push_back(create_pointwise_module(p, post_arg.name, {}, [&](auto* pm, auto&&) {
        std::vector<migraphx::instruction_ref> params;
        params.push_back(
            pm->add_parameter("!x0", migraphx::shape{ins_inputs.back()->get_shape().type()}));
        std::transform(post_arg.inputs.begin(),
                       post_arg.inputs.end(),
                       std::back_inserter(params),
                       [&](auto input) {
                           return pm->add_parameter("x" + std::to_string(params.size()),
                                                    migraphx::shape{input->get_shape().type()});
                       });
        return post_arg.f(pm, params);
    }));
    auto* mm = p.get_main_module();
    return mm->add_instruction(
        migraphx::make_op("fused_concat", {{"axis", axis}}), ins_inputs, module_inputs);
}

TEST_CASE(simple_concat_pointwise)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", s);
        auto y      = mm->add_parameter("y", s);
        auto add    = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto sub    = add_pointwise(p1, "main:pointwise1", {x, y}, single_pointwise("sub"));
        auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), add, sub);
        mm->add_return({concat});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto fused_concat =
            add_pointwise_concat(p2,
                                 1,
                                 arg("noop:concat0", {}, noop_pointwise()),
                                 arg("concat:main:pointwise0", {x, y}, single_pointwise("add")),
                                 arg("concat:main:pointwise1", {x, y}, single_pointwise("sub")));
        mm->add_return({fused_concat});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(partial_pointwise_concat)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 4, 8, 8}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 4, 16, 16}};
    migraphx::program p1;
    {
        auto* mm     = p1.get_main_module();
        auto x       = mm->add_parameter("x", s1);
        auto y       = mm->add_parameter("y", s1);
        auto z       = mm->add_parameter("z", s2);
        auto pooling = mm->add_instruction(
            migraphx::make_op("pooling", {{"lengths", {2, 2}}, {"stride", {2, 2}}}), z);
        auto add    = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), add, pooling);
        mm->add_return({concat});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm     = p2.get_main_module();
        auto x       = mm->add_parameter("x", s1);
        auto y       = mm->add_parameter("y", s1);
        auto z       = mm->add_parameter("z", s2);
        auto pooling = mm->add_instruction(
            migraphx::make_op("pooling", {{"lengths", {2, 2}}, {"stride", {2, 2}}}), z);
        auto fused_concat =
            add_pointwise_concat(p2,
                                 1,
                                 arg("noop:concat2", {}, noop_pointwise()),
                                 arg("concat:main:pointwise0", {x, y}, single_pointwise("add")),
                                 arg("concat:noop0", {pooling}, noop_pointwise()));
        mm->add_return({fused_concat});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(skip_pointwise_concat)
{
    // number of no-ops are two in this case and therefore pointwise+concat fusion wouldn't be
    // applicable.
    migraphx::shape s1{migraphx::shape::float_type, {1, 4, 8, 8}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 4, 16, 16}};
    migraphx::program p1;
    {
        auto* mm        = p1.get_main_module();
        auto w          = mm->add_parameter("w", s1);
        auto x          = mm->add_parameter("x", s1);
        auto y          = mm->add_parameter("y", s1);
        auto z          = mm->add_parameter("z", s2);
        auto reduce_ins = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), w);
        auto pooling    = mm->add_instruction(
            migraphx::make_op("pooling", {{"lengths", {2, 2}}, {"stride", {2, 2}}}), z);
        auto add    = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto concat = mm->add_instruction(
            migraphx::make_op("concat", {{"axis", 1}}), reduce_ins, add, pooling);
        mm->add_return({concat});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

TEST_CASE(simple_pointwise_concat_pointwise)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", s);
        auto y      = mm->add_parameter("y", s);
        auto add    = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto sub    = add_pointwise(p1, "main:pointwise1", {x, y}, single_pointwise("sub"));
        auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), add, sub);
        auto relu   = add_pointwise(p1, "main:pointwise2", {concat}, single_pointwise("relu"));
        mm->add_return({relu});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto fused_concat =
            add_pointwise_concat(p2,
                                 1,
                                 arg("main:pointwise2:concat", {}, single_pointwise("relu")),
                                 arg("concat:main:pointwise0", {x, y}, single_pointwise("add")),
                                 arg("concat:main:pointwise1", {x, y}, single_pointwise("sub")));
        mm->add_return({fused_concat});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(pointwise_concat_pointwise_multi_out)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 6}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", s1);
        auto y      = mm->add_parameter("y", s1);
        auto z      = mm->add_parameter("z", s2);
        auto add    = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto sub    = add_pointwise(p1, "main:pointwise1", {x, y}, single_pointwise("sub"));
        auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), add, sub);
        auto r      = add_pointwise(
            p1,
            "main:pointwise2",
            {concat, z},
            [=](auto* pm, const auto& inputs) -> std::vector<migraphx::instruction_ref> {
                auto mul  = pm->add_instruction(migraphx::make_op("mul"), inputs[0], inputs[1]);
                auto relu = pm->add_instruction(migraphx::make_op("relu"), mul);
                return {mul, relu};
            });
        auto mul  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), r);
        auto relu = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), r);
        mm->add_return({mul, relu});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s1);
        auto z   = mm->add_parameter("z", s2);
        auto fused_concat =
            add_pointwise_concat(p2,
                                 1,
                                 arg("noop:concat0", {}, noop_pointwise()),
                                 arg("concat:main:pointwise0", {x, y}, single_pointwise("add")),
                                 arg("concat:main:pointwise1", {x, y}, single_pointwise("sub")));
        auto r = add_pointwise(
            p2,
            "main:pointwise2",
            {fused_concat, z},
            [=](auto* pm, const auto& inputs) -> std::vector<migraphx::instruction_ref> {
                auto mul  = pm->add_instruction(migraphx::make_op("mul"), inputs[0], inputs[1]);
                auto relu = pm->add_instruction(migraphx::make_op("relu"), mul);
                return {mul, relu};
            });
        auto mul  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), r);
        auto relu = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), r);
        mm->add_return({mul, relu});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(partial_pointwise_concat_pointwise)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 4, 8, 8}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 4, 16, 16}};
    migraphx::program p1;
    {
        auto* mm     = p1.get_main_module();
        auto x       = mm->add_parameter("x", s1);
        auto y       = mm->add_parameter("y", s1);
        auto z       = mm->add_parameter("z", s2);
        auto pooling = mm->add_instruction(
            migraphx::make_op("pooling", {{"lengths", {2, 2}}, {"stride", {2, 2}}}), z);
        auto add    = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), add, pooling);
        auto relu   = add_pointwise(p1, "main:pointwise2", {concat}, single_pointwise("relu"));
        mm->add_return({relu});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm     = p2.get_main_module();
        auto x       = mm->add_parameter("x", s1);
        auto y       = mm->add_parameter("y", s1);
        auto z       = mm->add_parameter("z", s2);
        auto pooling = mm->add_instruction(
            migraphx::make_op("pooling", {{"lengths", {2, 2}}, {"stride", {2, 2}}}), z);
        auto fused_concat =
            add_pointwise_concat(p2,
                                 1,
                                 arg("main:pointwise2:concat", {}, single_pointwise("relu")),
                                 arg("concat:main:pointwise0", {x, y}, single_pointwise("add")),
                                 arg("concat:noop1", {pooling}, noop_pointwise()));
        mm->add_return({fused_concat});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(multiple_use_pointwise_concat_pointwise)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", s);
        auto y      = mm->add_parameter("y", s);
        auto add    = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto sub    = add_pointwise(p1, "main:pointwise1", {x, y}, single_pointwise("sub"));
        auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), add, sub);
        auto relu   = add_pointwise(p1, "main:pointwise2", {concat}, single_pointwise("relu"));
        auto slice  = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {4}}}), relu);
        auto mul = add_pointwise(p1, "main:pointwise3", {slice, sub}, single_pointwise({"mul"}));
        mm->add_return({mul});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto sub = add_pointwise(p2, "main:pointwise1", {x, y}, single_pointwise("sub"));
        auto fused_concat =
            add_pointwise_concat(p2,
                                 1,
                                 arg("main:pointwise2:concat", {}, single_pointwise("relu")),
                                 arg("concat:main:pointwise0", {x, y}, single_pointwise("add")),
                                 arg("concat:noop1", {sub}, noop_pointwise()));
        auto slice = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {4}}}),
            fused_concat);
        auto mul = add_pointwise(p2, "main:pointwise3", {slice, sub}, single_pointwise({"mul"}));
        mm->add_return({mul});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(pointwise_concat_fusion)
{
    migraphx::shape s1{migraphx::shape::half_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {2, 3}, {1, 2}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", s1);
        auto y      = mm->add_parameter("y", s2);
        auto yc     = mm->add_instruction(migraphx::make_op("contiguous"), y);
        auto sins   = add_pointwise(p1, "main:pointwise0", {x}, single_pointwise("sigmoid"));
        auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), sins, yc);
        auto relu   = add_pointwise(p1, "main:pointwise2", {concat}, single_pointwise("relu"));
        mm->add_return({relu});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto yc  = mm->add_instruction(migraphx::make_op("contiguous"), y);
        auto fused_concat =
            add_pointwise_concat(p2,
                                 1,
                                 arg("main:pointwise2:concat", {}, single_pointwise("relu")),
                                 arg("concat:main:pointwise0", {x}, single_pointwise("sigmoid")),
                                 arg("concat:noop1", {yc}, noop_pointwise()));
        mm->add_return({fused_concat});
    }
    EXPECT(p1 == p2);
}

// The pointwise input y is sliced to match each concat segment and the mul is
// split across the fused_concat's input submodules:
//
// Before:
//            x
//      +-----+-----+
//      |           |
//  slice[4:8] slice[0:4]
//    (xb)        (xa)
//      |           |
//      +-----+-----+
//            |
//     concat(axis=1)
//            |             y
//            +------+------+
//                   |
//                  mul
//                   |
//                 return
//
// After:
//            x                        y
//      +-----+-----+            +-----+-----+
//      |           |            |           |
//  slice[4:8] slice[0:4]    slice[0:4] slice[4:8]
//    (xb)        (xa)         (y0)        (y1)
//      |           |            |           |
//  +---+-----------+------------+-----------+---+
//  |            fused_concat(axis=1)            |
//  |  split0: mul(xb, y0), split1: mul(xa, y1)  |
//  +---------------------+----------------------+
//                        |
//                      return
TEST_CASE(pointwise_concat_of_slices_split)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto xb  = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {4}}, {"ends", {8}}}), x);
        auto xa = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {4}}}), x);
        auto rot = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), xb, xa);
        auto mul = add_pointwise(p1, "main:pointwise0", {rot, y}, single_pointwise("mul"));
        mm->add_return({mul});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto xb  = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {4}}, {"ends", {8}}}), x);
        auto xa = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {4}}}), x);
        auto y0 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {4}}}), y);
        auto y1 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {4}}, {"ends", {8}}}), y);
        auto fused_concat = add_pointwise_concat(
            p2,
            1,
            arg("noop:concat0", {}, noop_pointwise()),
            arg("concat:main:pointwise0:split0", {xb, y0}, single_pointwise("mul")),
            arg("concat:main:pointwise0:split1", {xa, y1}, single_pointwise("mul")));
        mm->add_return({fused_concat});
    }
    EXPECT(p1.sort() == p2.sort());
}

// The split mul segments and the outer concat's extra input all become inputs
// of a single fused_concat:
//
// Before:
//                   x
//      +------------+------------+
//      |            |            |
//  slice[4:8]   slice[0:4]   slice[8:12]
//    (xb)         (xa)        (xtail)
//      |            |            |
//      +-----+------+            |
//            |                   |
//     concat(axis=1)       y     |
//            |             |     |
//            +------+------+     |
//                   |            |
//                  mul           |
//                   |            |
//                   +-----+------+
//                         |
//                  concat(axis=1)
//                         |
//                       return
//
// After:
//                   x                             y
//      +------------+------------+          +-----+-----+
//      |            |            |          |           |
//  slice[4:8]   slice[0:4]   slice[8:12] slice[0:4] slice[4:8]
//    (xb)         (xa)        (xtail)      (y0)        (y1)
//      |            |            |          |           |
//  +---+------------+------------+----------+-----------+---+
//  |                  fused_concat(axis=1)                   |
//  |  split0: mul(xb, y0)  split1: mul(xa, y1)  noop(xtail)  |
//  +----------------------------+----------------------------+
//                               |
//                             return
TEST_CASE(pointwise_concat_of_slices_split_outer_concat)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 12}};
    migraphx::shape sr{migraphx::shape::float_type, {2, 8}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", sr);
        auto xb  = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {4}}, {"ends", {8}}}), x);
        auto xa = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {4}}}), x);
        auto xtail = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {8}}, {"ends", {12}}}), x);
        auto rot   = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), xb, xa);
        auto mul   = add_pointwise(p1, "main:pointwise0", {rot, y}, single_pointwise("mul"));
        auto outer = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), mul, xtail);
        mm->add_return({outer});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", sr);
        auto xb  = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {4}}, {"ends", {8}}}), x);
        auto xa = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {4}}}), x);
        auto xtail = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {8}}, {"ends", {12}}}), x);
        auto y0 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {4}}}), y);
        auto y1 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {4}}, {"ends", {8}}}), y);
        auto fused_concat = add_pointwise_concat(
            p2,
            1,
            arg("noop:concat2", {}, noop_pointwise()),
            arg("concat:main:pointwise0:split0", {xb, y0}, single_pointwise("mul")),
            arg("concat:main:pointwise0:split1", {xa, y1}, single_pointwise("mul")),
            arg("concat:noop0", {xtail}, noop_pointwise()));
        mm->add_return({fused_concat});
    }
    EXPECT(p1.sort() == p2.sort());
}

// The concat has a second use besides the pointwise (it is returned directly),
// so splitting the pointwise would duplicate work and no fusion happens:
//
//            x
//      +-----+-----+
//      |           |
//  slice[4:8] slice[0:4]
//    (xb)        (xa)
//      |           |
//      +-----+-----+
//            |
//     concat(axis=1)
//        |       |         y
//        |       +----+----+
//        |            |
//        |           mul
//        |            |
//        +-----+------+
//              |
//            return
TEST_CASE(pointwise_concat_of_slices_multi_use)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto xb  = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {4}}, {"ends", {8}}}), x);
        auto xa = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {4}}}), x);
        auto rot = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), xb, xa);
        auto mul = add_pointwise(p1, "main:pointwise0", {rot, y}, single_pointwise("mul"));
        mm->add_return({mul, rot});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

// The concat feeds both arguments of the pointwise, so the segments can't be
// paired one-to-one with slices of another input and no fusion happens:
//
//            x
//      +-----+-----+
//      |           |
//  slice[4:8] slice[0:4]
//    (xb)        (xa)
//      |           |
//      +-----+-----+
//            |
//     concat(axis=1)
//        |       |
//        +---+---+
//            |
//           mul
//            |
//          return
TEST_CASE(pointwise_concat_of_slices_repeated_arg)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto xb  = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {4}}, {"ends", {8}}}), x);
        auto xa = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {4}}}), x);
        auto rot = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), xb, xa);
        auto mul = add_pointwise(p1, "main:pointwise0", {rot, rot}, single_pointwise("mul"));
        mm->add_return({mul});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

// A single-input concat is a no-op copy; there is nothing to split, so no
// fusion happens:
//
//        x
//        |
//  slice[0:4]
//     (xa)
//        |
//   concat(axis=1)
//          |           y
//          +-----+-----+
//                |
//               mul
//                |
//              return
TEST_CASE(pointwise_concat_of_slices_single_input)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};
    migraphx::shape sr{migraphx::shape::float_type, {2, 4}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", sr);
        auto xa  = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {4}}}), x);
        auto cat = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), xa);
        auto mul = add_pointwise(p1, "main:pointwise0", {cat, y}, single_pointwise("mul"));
        mm->add_return({mul});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
