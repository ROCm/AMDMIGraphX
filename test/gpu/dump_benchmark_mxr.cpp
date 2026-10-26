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
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/json.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/process.hpp>
#include <migraphx/tmp_dir.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/value.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_ops.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <test.hpp>
#include <reduce.hpp>
#include <algorithm>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

static migraphx::fs::path executable; // NOLINT
static const char* const dump_flag = "--dump-benchmarks";

// A reduce has several exhaustive tuning candidates on every device
static migraphx::program reduce_program()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {4, 16}});
    mm->add_return({add_reduce(p, "main:reduce_sum0", {x}, {1}, single_reduce("reduce_sum"))});
    return p;
}

// Compile the reduce with every candidate saved to dir. The pass exits the process after
// saving, so returning at all means nothing was dumped.
static int dump_benchmarks(const migraphx::fs::path& dir)
{
    migraphx::log::set_severity(migraphx::log::severity::warn);
    migraphx::gpu::context ctx{0, 1};
    auto p = reduce_program();
    migraphx::run_passes(
        p,
        {migraphx::gpu::lowering{&ctx, false},
         migraphx::gpu::compile_ops{.ctx = &ctx, .exhaustive_tune = true, .dump_mxr_dir = dir}});
    return 1;
}

static migraphx::instruction_ref find_precompile_op(migraphx::module& m)
{
    auto ins = std::find_if(
        m.begin(), m.end(), [](const auto& i) { return i.name() == "gpu::precompile_op"; });
    EXPECT(ins != m.end());
    return ins;
}

static std::string json(const migraphx::value& v) { return migraphx::to_json_string(v); }

static std::vector<migraphx::fs::path> mxr_files(const migraphx::fs::path& dir)
{
    std::vector<migraphx::fs::path> files;
    std::copy_if(
        migraphx::fs::directory_iterator{dir},
        migraphx::fs::directory_iterator{},
        std::back_inserter(files),
        [](const migraphx::fs::directory_entry& e) { return e.path().extension() == ".mxr"; });
    return files;
}

// A benchmark MXR is a standalone program holding the compiled kernel and a json comment naming
// the op, problem and solution it was built for
static migraphx::value benchmark_comment(const migraphx::fs::path& file)
{
    auto p   = migraphx::load(file.string());
    auto* mm = p.get_main_module();
    EXPECT(std::any_of(
        mm->begin(), mm->end(), [](const auto& ins) { return ins.name() == "gpu::code_object"; }));
    auto comment = std::find_if(
        mm->begin(), mm->end(), [](const auto& ins) { return ins.name() == "@comment"; });
    EXPECT(comment != mm->end());
    return migraphx::from_json_string(
        comment->get_operator().to_value().at("text").to<std::string>());
}

// Dumping runs in a child process because the pass exits once the files are saved. Every
// candidate the pass would benchmark has to come back as its own file, including the ones whose
// kernels needed a nested compile of the device ops inserted when they were placed.
TEST_CASE(dumps_one_mxr_per_candidate)
{
    migraphx::gpu::context ctx{0, 1};
    auto p = reduce_program();
    migraphx::run_passes(p, {migraphx::gpu::lowering{&ctx, false}});
    auto ins   = find_precompile_op(*p.get_main_module());
    auto preop = migraphx::from_value<migraphx::operation>(ins->get_operator().to_value().at("op"));
    auto config = migraphx::gpu::get_tuning_config(ctx, ins, preop, true);
    EXPECT(config.has_value());
    EXPECT(config->solutions.size() > 1);

    migraphx::tmp_dir td{"benchmark-mxr"};
    migraphx::process{executable, {dump_flag, td.path.string()}}.exec();

    auto files = mxr_files(td.path);
    EXPECT(files.size() == config->solutions.size());

    std::vector<std::string> dumped;
    std::transform(files.begin(),
                   files.end(),
                   std::back_inserter(dumped),
                   [&](const migraphx::fs::path& file) {
                       auto comment = benchmark_comment(file);
                       EXPECT(comment.at("op").to<std::string>() == preop.name());
                       EXPECT(json(comment.at("problem").without_key()) == json(config->problem));
                       return json(comment.at("solution").without_key());
                   });
    std::vector<std::string> expected;
    std::transform(config->solutions.begin(),
                   config->solutions.end(),
                   std::back_inserter(expected),
                   [](const auto& solution) { return json(solution); });
    EXPECT(std::is_permutation(dumped.begin(), dumped.end(), expected.begin(), expected.end()));
}

int main(int argc, const char* argv[])
{
    if(argc > 2 and std::string_view{argv[1]} == dump_flag)
        return dump_benchmarks(argv[2]);
    // The child is spawned by path, so a bare-name argv[0] has to be resolved here.
    executable = migraphx::fs::absolute(argv[0]);
    test::run(argc, argv);
    return 0;
}
