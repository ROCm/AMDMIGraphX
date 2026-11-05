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
#include <migraphx/program.hpp>
#include <migraphx/builtin.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/compile_ops.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device_description.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <test.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

static migraphx::program make_program(migraphx::shape::type_t type = migraphx::shape::float_type)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{type, {4, 256}};
    auto x    = mm->add_parameter("x", s);
    auto y    = mm->add_parameter("y", s);
    auto add  = mm->add_instruction(migraphx::make_op("add"), x, y);
    auto relu = mm->add_instruction(migraphx::make_op("relu"), add);
    auto sum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), relu);
    mm->add_return({relu, sum});
    return p;
}

static migraphx::target cross_target(const std::string& arch)
{
    return migraphx::make_target("gpu", migraphx::value{{"gpu_arch", arch}});
}

static migraphx::compile_options recipe_options()
{
    migraphx::compile_options options;
    options.recompilable = true;
    options.offload_copy = true;
    return options;
}

static migraphx::program compile_program(const migraphx::target& t,
                                         migraphx::compile_options options = recipe_options(),
                                         migraphx::program p               = make_program())
{
    p.compile(t, std::move(options));
    return p;
}

static std::vector<migraphx::gpu::code_object_op> get_kernels(const migraphx::program& p)
{
    std::vector<migraphx::gpu::code_object_op> result;
    for(const auto* mod : p.get_modules())
    {
        for(const auto& ins : *mod)
        {
            if(ins.name() == "gpu::code_object")
                result.push_back(
                    migraphx::any_cast<migraphx::gpu::code_object_op>(ins.get_operator()));
        }
    }
    return result;
}

static std::vector<migraphx::instruction_ref> kernel_instructions(migraphx::program& p)
{
    std::vector<migraphx::instruction_ref> result;
    for(auto* mod : p.get_modules())
    {
        for(auto ins : migraphx::iterator_for(*mod))
        {
            if(ins->name() == "gpu::code_object")
                result.push_back(ins);
        }
    }
    return result;
}

static bool all_have_recipes(migraphx::program& p)
{
    auto kernels = kernel_instructions(p);
    return not kernels.empty() and std::all_of(kernels.begin(), kernels.end(), [](auto ins) {
        return migraphx::gpu::find_recipe(ins).has_value();
    });
}

static std::string compiled_arch(const migraphx::program& p)
{
    auto device = migraphx::gpu::get_compiled_for(*p.get_main_module());
    return device.has_value() ? device->arch : "";
}

static migraphx::recompile_options force_options()
{
    migraphx::recompile_options options;
    options.force = true;
    return options;
}

static std::vector<migraphx::argument> run_program(const migraphx::program& p,
                                                   const migraphx::parameter_map& params)
{
    return p.eval(params);
}

static migraphx::parameter_map make_params()
{
    migraphx::shape s{migraphx::shape::float_type, {4, 256}};
    return {{"x", migraphx::generate_argument(s, 0)}, {"y", migraphx::generate_argument(s, 1)}};
}

static std::vector<float> to_vector(const migraphx::argument& arg)
{
    std::vector<float> result;
    arg.visit([&](auto v) { result.assign(v.begin(), v.end()); });
    return result;
}

static void expect_matches_ref(const migraphx::program& p)
{
    auto ref = make_program();
    ref.compile(migraphx::make_target("ref"));
    auto params   = make_params();
    auto expected = run_program(ref, params);
    auto results  = run_program(p, params);
    EXPECT(results.size() == expected.size());
    for(std::size_t i = 0; i < results.size(); i++)
        EXPECT(migraphx::verify::verify_rms_range(to_vector(results[i]), to_vector(expected[i])));
}

// Lowering fills the output of nms_compact before the kernel runs
static migraphx::program make_nms_program()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};
    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    migraphx::shape scalar_s{migraphx::shape::float_type, {1}};
    migraphx::shape int_scalar_s{migraphx::shape::int64_type, {1}};
    auto boxes   = mm->add_parameter("boxes", boxes_s);
    auto scores  = mm->add_parameter("scores", scores_s);
    auto max_out = mm->add_literal(migraphx::literal{int_scalar_s, std::vector<std::int64_t>{4}});
    auto iou     = mm->add_literal(migraphx::literal{scalar_s, std::vector<float>{0.5f}});
    auto score   = mm->add_literal(migraphx::literal{scalar_s, std::vector<float>{0.0f}});
    auto nms =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                            boxes,
                            scores,
                            max_out,
                            iou,
                            score);
    auto idx = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nms);
    auto cnt = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nms);
    mm->add_return({idx, cnt});
    return p;
}

static void expect_nms_result(const migraphx::program& p)
{
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};
    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> boxes  = {0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                 0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};
    std::vector<float> scores = {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f};
    auto results = run_program(p,
                               {{"boxes", migraphx::argument(boxes_s, boxes.data())},
                                {"scores", migraphx::argument(scores_s, scores.data())}});
    EXPECT(results.size() == 2);
    std::vector<std::int64_t> indices;
    results[0].visit([&](auto v) { indices.assign(v.begin(), v.end()); });
    std::int64_t num_selected = 0;
    results[1].visit([&](auto v) { num_selected = static_cast<std::int64_t>(v[0]); });
    EXPECT(num_selected == 3);
    indices.resize(9);
    EXPECT(indices == std::vector<std::int64_t>{0, 0, 3, 0, 0, 0, 0, 0, 5});
}

static bool has_kernel(const migraphx::program& p, const std::string& symbol)
{
    auto kernels = get_kernels(p);
    return std::any_of(
        kernels.begin(), kernels.end(), [&](const auto& co) { return co.symbol_name == symbol; });
}

TEST_CASE(gfx_generation)
{
    EXPECT(migraphx::gpu::get_gfx_generation("gfx90a") == 9);
    EXPECT(migraphx::gpu::get_gfx_generation("gfx942:sramecc+:xnack-") == 9);
    EXPECT(migraphx::gpu::get_gfx_generation("gfx1030") == 10);
    EXPECT(migraphx::gpu::get_gfx_generation("gfx1151") == 11);
    EXPECT(migraphx::gpu::get_gfx_generation("gfx1201") == 12);
    EXPECT(migraphx::gpu::get_gfx_generation("sm_80") == 0);
}

TEST_CASE(code_object_tuning_level_round_trip)
{
    migraphx::gpu::code_object_op co;
    co.symbol_name  = "kernel";
    co.tuning_level = "quick";
    auto op         = migraphx::make_op("gpu::code_object", migraphx::to_value(co));
    auto op2        = migraphx::make_op("gpu::code_object", op.to_value());
    auto co2        = migraphx::any_cast<migraphx::gpu::code_object_op>(op2);
    EXPECT(co2.tuning_level == "quick");
}

TEST_CASE(recipes_kept)
{
    auto p = compile_program(migraphx::make_target("gpu"));
    EXPECT(all_have_recipes(p));
    for(auto ins : kernel_instructions(p))
    {
        const auto& co = migraphx::any_cast<migraphx::gpu::code_object_op>(ins->get_operator());
        EXPECT(co.tuning_level == "quick" or co.tuning_level == "none");
        // The recipe takes the same arguments as the kernel and is skipped by passes
        auto recipe = *migraphx::gpu::find_recipe(ins);
        EXPECT(recipe->inputs().size() == ins->inputs().size());
        EXPECT(not migraphx::gpu::recipe_op_name(recipe).empty());
        EXPECT(ins->module_inputs().front()->bypass());
    }
    expect_matches_ref(p);
}

TEST_CASE(recipes_survive_save_and_load)
{
    auto p = compile_program(migraphx::make_target("gpu"));
    migraphx::file_options fo;
    fo.finalize = false;
    auto q      = migraphx::load_buffer(migraphx::save_buffer(p), fo);
    EXPECT(all_have_recipes(q));
    EXPECT(kernel_instructions(q).size() == kernel_instructions(p).size());
}

TEST_CASE(no_recipes_by_default)
{
    migraphx::compile_options options;
    options.offload_copy = true;
    auto p               = compile_program(migraphx::make_target("gpu"), options);
    for(auto ins : kernel_instructions(p))
    {
        EXPECT(not migraphx::gpu::find_recipe(ins).has_value());
        EXPECT(ins->module_inputs().empty());
        EXPECT(migraphx::any_cast<migraphx::gpu::code_object_op>(ins->get_operator())
                   .tuning_level.empty());
    }
    EXPECT(not migraphx::gpu::get_compiled_for(*p.get_main_module()).has_value());
}

TEST_CASE(compiled_for_recorded_as_comment)
{
    auto p   = compile_program(cross_target("gfx1100"));
    auto* mm = p.get_main_module();
    auto it  = std::find_if(mm->begin(), mm->end(), [](const auto& ins) {
        return ins.name() == "@comment" and
               migraphx::any_cast<migraphx::builtin::comment>(ins.get_operator()).tag ==
                   "gpu_target";
    });
    EXPECT(it != mm->end());
    EXPECT(compiled_arch(p) == "gfx1100");
}

TEST_CASE(load_without_finalize)
{
    auto p = compile_program(migraphx::make_target("gpu"));
    migraphx::file_options fo;
    fo.finalize = false;
    auto q      = migraphx::load_buffer(migraphx::save_buffer(p), fo);
    EXPECT(q.is_compiled());
    EXPECT(compiled_arch(q) == compiled_arch(p));
    EXPECT(test::throws<migraphx::exception>([&] { q.eval(make_params()); }, "without finalizing"));
    // Saving an unfinalized program keeps the device it was compiled for, and the contexts are
    // created when it is finalized
    auto r = migraphx::load_buffer(migraphx::save_buffer(q), fo);
    EXPECT(compiled_arch(r) == compiled_arch(p));
    r.finalize();
    expect_matches_ref(r);
}

TEST_CASE(recompile_same_target_is_noop)
{
    auto p = compile_program(migraphx::make_target("gpu"));
    migraphx::file_options fo;
    fo.finalize = false;
    auto q      = migraphx::load_buffer(migraphx::save_buffer(p), fo);
    q.recompile(migraphx::make_target("gpu"));
    auto before = get_kernels(p);
    auto after  = get_kernels(q);
    EXPECT(before.size() == after.size());
    for(std::size_t i = 0; i < before.size(); i++)
        EXPECT(before[i].code_object == after[i].code_object);
    expect_matches_ref(q);
}

TEST_CASE(recompile_same_target_force)
{
    auto p = compile_program(migraphx::make_target("gpu"));
    migraphx::file_options fo;
    fo.finalize = false;
    auto q      = migraphx::load_buffer(migraphx::save_buffer(p), fo);
    q.recompile(migraphx::make_target("gpu"), {}, force_options());
    EXPECT(all_have_recipes(q));
    expect_matches_ref(q);
}

TEST_CASE(recompile_shares_modules)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {4, 256}};
    auto x = mm->add_parameter("x", s);
    auto y = mm->add_parameter("y", s);
    mm->add_return({mm->add_instruction(migraphx::make_op("relu"), x),
                    mm->add_instruction(migraphx::make_op("relu"), y)});
    p = compile_program(migraphx::make_target("gpu"), recipe_options(), p);
    p.recompile(migraphx::make_target("gpu"), {}, force_options());
    auto kernels = get_kernels(p);
    EXPECT(kernels.size() == 2);
    EXPECT(kernels[0].code_object == kernels[1].code_object);
    EXPECT(kernels[0].module_id != -1);
    EXPECT(kernels[0].module_id == kernels[1].module_id);
    EXPECT(all_have_recipes(p));
}

TEST_CASE(recompile_exhaustive_level)
{
    auto p = compile_program(migraphx::make_target("gpu"));
    migraphx::compile_options options;
    options.exhaustive_tune = true;
    p.recompile(migraphx::make_target("gpu"), options);
    // Only tunable kernels are recompiled at the higher level
    for(const auto& co : get_kernels(p))
        EXPECT(co.tuning_level == "none" or co.tuning_level == "exhaustive");
    expect_matches_ref(p);
}

TEST_CASE(retarget_cross_compile_newer_generation)
{
    auto p = compile_program(cross_target("gfx1100"));
    EXPECT(compiled_arch(p) == "gfx1100");
    auto before = get_kernels(p);
    p.recompile(cross_target("gfx1201"));
    EXPECT(compiled_arch(p) == "gfx1201");
    auto after = get_kernels(p);
    EXPECT(before.size() == after.size());
    EXPECT(all_have_recipes(p));
    for(std::size_t i = 0; i < before.size(); i++)
    {
        EXPECT(before[i].code_object != after[i].code_object);
        EXPECT(before[i].expected_inputs == after[i].expected_inputs);
    }
    // The retargeted program round trips through a file
    migraphx::file_options fo;
    fo.finalize = false;
    auto q      = migraphx::load_buffer(migraphx::save_buffer(p), fo);
    EXPECT(compiled_arch(q) == "gfx1201");
}

TEST_CASE(retarget_cross_compile_same_generation)
{
    auto p = compile_program(cross_target("gfx1151"));
    p.recompile(cross_target("gfx1100"));
    EXPECT(compiled_arch(p) == "gfx1100");
}

TEST_CASE(retarget_older_generation_refused)
{
    auto p = compile_program(cross_target("gfx1151"));
    EXPECT(test::throws<migraphx::exception>([&] { p.recompile(cross_target("gfx942")); },
                                             "older generation"));
    // The program is unchanged
    EXPECT(compiled_arch(p) == "gfx1151");
}

TEST_CASE(retarget_without_recipes_refused)
{
    migraphx::compile_options options;
    options.offload_copy = true;
    auto p               = compile_program(cross_target("gfx1100"), options);
    EXPECT(test::throws<migraphx::exception>([&] { p.recompile(cross_target("gfx1201")); },
                                             "compile the model again"));
}

TEST_CASE(retarget_fp8_fnuz_refused)
{
    auto p = compile_program(
        cross_target("gfx942"), recipe_options(), make_program(migraphx::shape::fp8e4m3fnuz_type));
    EXPECT(test::throws<migraphx::exception>([&] { p.recompile(cross_target("gfx1151")); },
                                             "fp8 fnuz"));
}

TEST_CASE(load_version_8_program_refused)
{
    auto p       = compile_program(migraphx::make_target("gpu"));
    auto v       = p.to_value();
    v["version"] = 8;
    migraphx::program q;
    EXPECT(test::throws<migraphx::exception>([&] { q.from_value(v); }, "version mismatch"));
}

TEST_CASE(recompile_kernel_with_lowering_fill)
{
    auto p = compile_program(migraphx::make_target("gpu"), recipe_options(), make_nms_program());
    EXPECT(has_kernel(p, "nms_compact_kernel"));
    migraphx::file_options fo;
    fo.finalize = false;
    auto q      = migraphx::load_buffer(migraphx::save_buffer(p), fo);
    q.recompile(migraphx::make_target("gpu"), {}, force_options());
    EXPECT(has_kernel(q, "nms_compact_kernel"));
    expect_nms_result(q);
}

TEST_CASE(retarget_kernel_with_lowering_fill)
{
    auto p = compile_program(cross_target("gfx1151"), recipe_options(), make_nms_program());
    p.recompile(cross_target("gfx1100"));
    EXPECT(compiled_arch(p) == "gfx1100");
    EXPECT(has_kernel(p, "nms_compact_kernel"));
}

TEST_CASE(recompile_requires_compiled_program)
{
    auto p = make_program();
    EXPECT(test::throws<migraphx::exception>([&] { p.recompile(migraphx::make_target("gpu")); },
                                             "not compiled"));
}

TEST_CASE(recompile_unsupported_target_refused)
{
    auto p = make_program();
    p.compile(migraphx::make_target("ref"));
    EXPECT(test::throws<migraphx::exception>([&] { p.recompile(migraphx::make_target("ref")); },
                                             "does not support recompile"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
