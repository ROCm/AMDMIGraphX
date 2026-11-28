/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/standardize_output.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/program.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/ranges.hpp>
#include <algorithm>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::gpu::standardize_output{}});
}

static migraphx::instruction_ref add_packed(migraphx::module& m, migraphx::instruction_ref input)
{
    const auto& s = input->get_shape();
    auto alloc    = m.add_instruction(migraphx::make_op(
        "allocate", {{"shape", migraphx::to_value(migraphx::shape{s.type(), s.lens()})}}));
    return m.add_instruction(migraphx::make_op("gpu::contiguous"), input, alloc);
}

TEST_CASE(nonstandard_return_gets_one_contiguous)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::float_type, {1, 128, 498}});
    auto t = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), x);
    m.add_return({t});

    run_pass(m);

    migraphx::module expected;
    auto ex = expected.add_parameter("x", {migraphx::shape::float_type, {1, 128, 498}});
    auto et =
        expected.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), ex);
    expected.add_return({add_packed(expected, et)});
    EXPECT(m == expected);
}

TEST_CASE(packed_return_is_unchanged)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::float_type, {1, 128, 498}});
    m.add_return({x});
    auto before = m;

    run_pass(m);

    EXPECT(m == before);
}

TEST_CASE(only_the_nonstandard_output)
{
    migraphx::module m;
    auto a = m.add_parameter("a", {migraphx::shape::float_type, {2, 3}});
    auto b = m.add_parameter("b", {migraphx::shape::float_type, {2, 3}});
    auto t = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), a);
    m.add_return({t, b});

    run_pass(m);

    migraphx::module expected;
    auto ea = expected.add_parameter("a", {migraphx::shape::float_type, {2, 3}});
    auto eb = expected.add_parameter("b", {migraphx::shape::float_type, {2, 3}});
    auto et =
        expected.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), ea);
    expected.add_return({add_packed(expected, et), eb});
    EXPECT(m == expected);
}

TEST_CASE(single_element_return_is_unchanged)
{
    migraphx::module m;
    // Every static shape whose dimensions are all 1 is standard: strides on size-1 axes
    // are ignored. An empty tensor can still be nonstandard, and elements() == 0 is what
    // keeps it from being packed.
    auto x = m.add_parameter("x", {migraphx::shape::float_type, {2, 0}, {1, 1}});
    m.add_return({x});
    auto before = m;

    run_pass(m);

    EXPECT(m == before);
    EXPECT(x->get_shape().elements() == 0);
    EXPECT(not x->get_shape().standard());
}

TEST_CASE(dynamic_return_is_unchanged)
{
    migraphx::module m;
    auto x = m.add_parameter(
        "x", {migraphx::shape::float_type, {migraphx::shape::dynamic_dimension{1, 4}}});
    m.add_return({x});
    auto before = m;

    run_pass(m);

    EXPECT(m == before);
}

TEST_CASE(offload_copy_packs_the_gpu_value)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto t = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x);
    auto c = m.add_instruction(migraphx::make_op("hip::copy_from_gpu"), t);
    m.add_return({c});

    run_pass(m);

    migraphx::module expected;
    auto ex = expected.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto et =
        expected.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), ex);
    auto packed = add_packed(expected, et);
    auto ec     = expected.add_instruction(migraphx::make_op("hip::copy_from_gpu"), packed);
    expected.add_return({ec});
    EXPECT(m == expected);
}

TEST_CASE(returned_layout_gets_one_contiguous)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::float_type, {1, 3, 4, 4}});
    auto layout =
        m.add_instruction(migraphx::make_op("layout", {{"permutation", {0, 2, 3, 1}}}), x);
    m.add_return({layout});

    run_pass(m);

    migraphx::module expected;
    auto ex = expected.add_parameter("x", {migraphx::shape::float_type, {1, 3, 4, 4}});
    auto elayout =
        expected.add_instruction(migraphx::make_op("layout", {{"permutation", {0, 2, 3, 1}}}), ex);
    expected.add_return({add_packed(expected, elayout)});
    EXPECT(m == expected);
}

TEST_CASE(implicit_return_is_packed)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x);

    run_pass(m);

    migraphx::module expected;
    auto ex = expected.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto et =
        expected.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), ex);
    add_packed(expected, et);
    EXPECT(m == expected);
    EXPECT(std::prev(m.end())->name() != "@return");
}

TEST_CASE(tuple_return_packs_leaves)
{
    migraphx::module m;
    auto x      = m.add_parameter("x", {migraphx::shape::float_type, {4, 6}});
    auto layout = m.add_instruction(migraphx::make_op("layout", {{"permutation", {1, 0}}}), x);
    auto top    = m.add_instruction(migraphx::make_op("topk", {{"k", 2}, {"axis", 0}}), layout);
    m.add_return({top});

    run_pass(m);

    migraphx::module expected;
    auto ex = expected.add_parameter("x", {migraphx::shape::float_type, {4, 6}});
    auto elayout =
        expected.add_instruction(migraphx::make_op("layout", {{"permutation", {1, 0}}}), ex);
    auto etop =
        expected.add_instruction(migraphx::make_op("topk", {{"k", 2}, {"axis", 0}}), elayout);
    auto values =
        expected.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), etop);
    auto packed_values = add_packed(expected, values);
    auto indices =
        expected.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), etop);
    auto packed_indices = add_packed(expected, indices);
    auto grouped        = expected.add_instruction(
        migraphx::make_op("gpu::group_tuple"), packed_values, packed_indices);
    expected.add_return({grouped});
    EXPECT(m == expected);
}

TEST_CASE(control_flow_submodule_is_not_packed)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto cond    = mm->add_parameter("cond", migraphx::shape{migraphx::shape::bool_type});
    auto x       = mm->add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto* branch = p.create_module("branch");
    auto y       = branch->add_instruction(migraphx::make_op("relu"), x);
    auto t = branch->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), y);
    branch->add_return({t});
    auto chosen = mm->add_instruction(migraphx::make_op("if"), {cond}, {branch, branch});
    auto elem   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), chosen);
    mm->add_return({elem});
    auto branch_before = *branch;

    migraphx::run_passes(p, {migraphx::gpu::standardize_output{}});

    EXPECT(*branch == branch_before);

    migraphx::module expected;
    auto econd = expected.add_parameter("cond", migraphx::shape{migraphx::shape::bool_type});
    expected.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto echosen = expected.add_instruction(migraphx::make_op("if"), {econd}, {branch, branch});
    auto eelem =
        expected.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), echosen);
    expected.add_return({add_packed(expected, eelem)});
    EXPECT(*mm == expected);
}

static bool has_contiguous_kernel(const migraphx::program& p)
{
    auto mods = p.get_modules();
    return std::any_of(mods.begin(), mods.end(), [](const migraphx::module* mod) {
        return std::any_of(mod->begin(), mod->end(), [](const migraphx::instruction& ins) {
            if(ins.name() != "gpu::code_object")
                return false;
            return migraphx::contains(
                ins.get_operator().to_value().at("symbol_name").to<std::string>(), "contiguous");
        });
    });
}

static migraphx::program compile_returned_layout(bool standardize, bool offload)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {1, 3, 4, 4}});
    auto layout =
        mm->add_instruction(migraphx::make_op("layout", {{"permutation", {0, 2, 3, 1}}}), x);
    mm->add_return({layout});
    migraphx::compile_options options;
    options.offload_copy                           = offload;
    options.backend_options["standardize_outputs"] = standardize;
    p.compile(migraphx::make_target("gpu"), options);
    return p;
}

TEST_CASE(compile_packs_layout_by_default)
{
    auto p     = compile_returned_layout(true, false);
    auto shape = p.get_main_module()->get_output_shapes().front();
    EXPECT(shape.standard());
    EXPECT(has_contiguous_kernel(p));
}

TEST_CASE(compile_keeps_layout_when_disabled)
{
    auto p     = compile_returned_layout(false, false);
    auto shape = p.get_main_module()->get_output_shapes().front();
    EXPECT(not shape.standard());
    EXPECT(not has_contiguous_kernel(p));
}

TEST_CASE(compile_offload_copy_is_packed)
{
    auto p     = compile_returned_layout(true, true);
    auto shape = p.get_main_module()->get_output_shapes().front();
    EXPECT(shape.standard());
}

TEST_CASE(compile_fused_pointwise_adds_no_contiguous)
{
    auto compile_add = [](bool standardize) {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {2, 4}});
        auto y   = mm->add_parameter("y", {migraphx::shape::float_type, {2, 4}});
        auto sum = mm->add_instruction(migraphx::make_op("add"), x, y);
        mm->add_return({sum});
        migraphx::compile_options options;
        options.offload_copy                           = false;
        options.backend_options["standardize_outputs"] = standardize;
        p.compile(migraphx::make_target("gpu"), options);
        return p;
    };
    auto on  = compile_add(true);
    auto off = compile_add(false);
    EXPECT(on.get_main_module()->get_output_shapes().front().standard());
    EXPECT(not has_contiguous_kernel(on));
    EXPECT(not has_contiguous_kernel(off));
    EXPECT(on.get_main_module()->get_output_shapes() == off.get_main_module()->get_output_shapes());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
