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
#include <migraphx/gpu/recompile.hpp>
#include <migraphx/gpu/record_compiled_for.hpp>
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

static std::size_t count_comments(const migraphx::program& p, const std::string& tag)
{
    const auto* mm = p.get_main_module();
    return static_cast<std::size_t>(std::count_if(mm->begin(), mm->end(), [&](const auto& ins) {
        return ins.name() == "@comment" and
               migraphx::any_cast<migraphx::builtin::comment>(ins.get_operator()).tag == tag;
    }));
}

static std::string compiled_arch(const migraphx::program& p)
{
    auto device = migraphx::gpu::get_compiled_for(*p.get_main_module());
    return device.has_value() ? device->arch : "";
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

static migraphx::compile_options options_with_staging(std::size_t percent)
{
    auto options                                            = recipe_options();
    options.backend_options["literal_staging_peak_percent"] = percent;
    return options;
}

static migraphx::value recorded_options(const migraphx::program& p)
{
    return migraphx::gpu::get_compile_options(*p.get_main_module());
}

static std::size_t recorded_staging(const migraphx::program& p)
{
    return recorded_options(p)
        .at("backend_options")
        .at("literal_staging_peak_percent")
        .to<std::size_t>();
}

TEST_CASE(compile_options_recorded_as_comment)
{
    auto p   = compile_program(migraphx::make_target("gpu"), options_with_staging(7));
    auto rec = recorded_options(p);
    EXPECT(count_comments(p, "gpu_compile_options") == 1);
    EXPECT(rec.at("offload_copy").to<bool>());
    EXPECT(rec.at("recompilable").to<bool>());
    EXPECT(not rec.at("exhaustive_tune").to<bool>());
    EXPECT(rec.at("compile_mode").to<std::string>() == "balanced");
    EXPECT(recorded_staging(p) == 7);
    migraphx::file_options fo;
    fo.finalize = false;
    auto q      = migraphx::load_buffer(migraphx::save_buffer(p), fo);
    EXPECT(recorded_options(q) == rec);
}

TEST_CASE(merge_compile_options_rules)
{
    migraphx::value recorded = {{"exhaustive_tune", false},
                                {"compile_mode", "balanced"},
                                {"fast_math", true},
                                {"backend_options", {{"a", 1}, {"b", 2}}}};
    migraphx::value user     = {{"exhaustive_tune", false},
                                {"compile_mode", "max"},
                                {"fast_math", false},
                                {"backend_options", {{"b", 3}}}};
    auto merged              = migraphx::gpu::merge_compile_options(recorded, user);
    // compile_mode max asks for exhaustive tuning, but the recorded mode is kept
    EXPECT(migraphx::gpu::is_exhaustive(merged));
    EXPECT(merged.at("compile_mode").to<std::string>() == "balanced");
    EXPECT(merged.at("fast_math").to<bool>());
    EXPECT(merged.at("backend_options").at("a").to<int>() == 1);
    EXPECT(merged.at("backend_options").at("b").to<int>() == 3);

    // Exhaustive tuning that was recorded is not lowered
    recorded["exhaustive_tune"] = true;
    user["compile_mode"]        = "balanced";
    EXPECT(migraphx::gpu::is_exhaustive(migraphx::gpu::merge_compile_options(recorded, user)));

    // Without a record the user's options are used
    EXPECT(migraphx::gpu::merge_compile_options({}, user) == user);
}

TEST_CASE(check_recompiled_memory_refuses_allocation)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {4}};
    auto x = m.add_parameter("x", s);
    m.add_return({m.add_instruction(migraphx::make_op("relu"), x)});
    migraphx::gpu::check_recompiled_memory{}.apply(m);

    auto alloc =
        m.add_instruction(migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(s)}}));
    m.add_instruction(migraphx::make_op("identity"), alloc);
    EXPECT(test::throws<migraphx::exception>(
        [&] { migraphx::gpu::check_recompiled_memory{}.apply(m); }, "need memory"));
}

TEST_CASE(recipes_kept)
{
    auto p = compile_program(migraphx::make_target("gpu"));
    EXPECT(all_have_recipes(p));
    for(auto ins : kernel_instructions(p))
    {
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
    }
    EXPECT(count_comments(p, "gpu_target") == 0);
    EXPECT(count_comments(p, "gpu_compile_options") == 0);
}

TEST_CASE(compiled_for_recorded_as_comment)
{
    auto p = compile_program(cross_target("gfx1100"));
    EXPECT(count_comments(p, "gpu_target") == 1);
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

TEST_CASE(recompile_same_target)
{
    auto p = compile_program(migraphx::make_target("gpu"));
    migraphx::file_options fo;
    fo.finalize = false;
    auto q      = migraphx::load_buffer(migraphx::save_buffer(p), fo);
    q.recompile(migraphx::make_target("gpu"));
    EXPECT(get_kernels(q).size() == get_kernels(p).size());
    EXPECT(all_have_recipes(q));
    EXPECT(compiled_arch(q) == compiled_arch(p));
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
    p.recompile(migraphx::make_target("gpu"));
    auto kernels = get_kernels(p);
    EXPECT(kernels.size() == 2);
    EXPECT(kernels[0].code_object == kernels[1].code_object);
    EXPECT(kernels[0].module_id != -1);
    EXPECT(kernels[0].module_id == kernels[1].module_id);
    EXPECT(all_have_recipes(p));
}

TEST_CASE(recompile_exhaustive_level)
{
    auto p = compile_program(migraphx::make_target("gpu"), options_with_staging(7));
    EXPECT(not migraphx::gpu::is_exhaustive(recorded_options(p)));
    migraphx::compile_options options;
    options.exhaustive_tune                                 = true;
    options.backend_options["literal_staging_peak_percent"] = 9;
    p.recompile(migraphx::make_target("gpu"), options);
    EXPECT(migraphx::gpu::is_exhaustive(recorded_options(p)));
    EXPECT(recorded_staging(p) == 9);
    EXPECT(count_comments(p, "gpu_compile_options") == 1);
    // A later recompile without options keeps the recorded ones
    p.recompile(migraphx::make_target("gpu"));
    EXPECT(migraphx::gpu::is_exhaustive(recorded_options(p)));
    EXPECT(recorded_staging(p) == 9);
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
    q.recompile(migraphx::make_target("gpu"));
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

static void expect_same_as_ref(const migraphx::program& p,
                               migraphx::program ref,
                               const migraphx::parameter_map& params)
{
    ref.compile(migraphx::make_target("ref"));
    auto expected = run_program(ref, params);
    auto results  = run_program(p, params);
    EXPECT(results.size() == expected.size());
    for(std::size_t i = 0; i < results.size(); i++)
        EXPECT(migraphx::verify::verify_rms_range(to_vector(results[i]), to_vector(expected[i])));
}

// The compiler of scatter copies the data into the output before the kernel, so the kernel does
// not take the data as an argument
static migraphx::program make_scatter_program()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape data_s{migraphx::shape::float_type, {4, 4}};
    migraphx::shape indices_s{migraphx::shape::int64_type, {2, 1}};
    migraphx::shape updates_s{migraphx::shape::float_type, {2, 4}};
    auto data    = mm->add_parameter("data", data_s);
    auto indices = mm->add_literal(migraphx::literal{indices_s, std::vector<std::int64_t>{0, 2}});
    auto updates = mm->add_parameter("updates", updates_s);
    mm->add_return(
        {mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates)});
    return p;
}

TEST_CASE(recompile_scatter)
{
    auto p =
        compile_program(migraphx::make_target("gpu"), recipe_options(), make_scatter_program());
    auto kernels = kernel_instructions(p);
    // The recipe finds the data through the copy
    EXPECT(std::any_of(kernels.begin(), kernels.end(), [](auto ins) {
        auto inputs = migraphx::gpu::recipe_inputs(ins);
        return inputs.has_value() and inputs->size() == 4 and ins->inputs().size() == 3;
    }));
    p.recompile(migraphx::make_target("gpu"));
    EXPECT(all_have_recipes(p));
    expect_same_as_ref(
        p,
        make_scatter_program(),
        {{"data", migraphx::generate_argument({migraphx::shape::float_type, {4, 4}}, 0)},
         {"updates", migraphx::generate_argument({migraphx::shape::float_type, {2, 4}}, 1)}});
}

// Instructions of the main module that nothing uses, other than the last one
static std::size_t count_unused(const migraphx::program& p)
{
    const auto* mm = p.get_main_module();
    auto last      = std::prev(mm->end());
    return static_cast<std::size_t>(std::count_if(mm->begin(), last, [](const auto& ins) {
        return ins.outputs().empty() and ins.get_shape().ndim() != 0 and ins.name() != "@param" and
               ins.name() != "@literal";
    }));
}

// The concat is eliminated, so the reductions write into parts of its buffer and nothing uses
// them
static migraphx::program make_concat_program()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {4, 256}};
    auto x = mm->add_parameter("x", s);
    auto y = mm->add_parameter("y", s);
    auto a = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), x);
    auto b = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), y);
    auto c = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), a, b);
    mm->add_return({mm->add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), c)});
    return p;
}

TEST_CASE(recompile_keeps_unused_writers)
{
    auto p = compile_program(migraphx::make_target("gpu"), recipe_options(), make_concat_program());
    auto unused = count_unused(p);
    EXPECT(unused > 0);
    auto kernels = get_kernels(p).size();
    p.recompile(migraphx::make_target("gpu"));
    EXPECT(count_unused(p) == unused);
    EXPECT(get_kernels(p).size() == kernels);
    expect_same_as_ref(p, make_concat_program(), make_params());
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
