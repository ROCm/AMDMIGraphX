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
#include <migraphx/make_op.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/tmp_dir.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/binary_cache.hpp>
#include <migraphx/gpu/compiled_code.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <test.hpp>

static migraphx::program pointwise_program()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {4, 8}};
    auto x    = mm->add_parameter("x", s);
    auto y    = mm->add_parameter("y", s);
    auto add  = mm->add_instruction(migraphx::make_op("add"), x, y);
    auto relu = mm->add_instruction(migraphx::make_op("relu"), add);
    auto mul  = mm->add_instruction(migraphx::make_op("mul"), relu, x);
    mm->add_return({mul});
    return p;
}

// Independent branches of the same shape lower to separate instructions that cannot fuse into
// each other, but they all produce the same kernel, so only one of them should be compiled.
static migraphx::program repeated_reduce(std::size_t n)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {4, 8}};
    std::vector<migraphx::instruction_ref> branches;
    for(std::size_t i = 0; i < n; i++)
    {
        auto x   = mm->add_parameter("x" + std::to_string(i), s);
        auto mul = mm->add_instruction(migraphx::make_op("mul"), x, x);
        branches.push_back(
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul));
    }
    mm->add_return({mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), branches)});
    return p;
}

static migraphx::compile_options cache_options(const migraphx::fs::path& dir, bool verify = false)
{
    migraphx::compile_options options;
    migraphx::set_backend_options(
        options, {{"binary_cache", dir.string()}, {"binary_cache_verify", verify}});
    return options;
}

static migraphx::gpu::binary_cache_stats
compile_and_get_stats(migraphx::program p, const migraphx::compile_options& options)
{
    p.compile(migraphx::make_target("gpu"), options);
    return migraphx::any_cast<migraphx::gpu::context>(p.get_context())
        .get_binary_cache()
        .get_stats();
}

TEST_CASE(identical_kernels_compile_once)
{
    migraphx::tmp_dir td{"binary-cache"};
    const std::size_t n = 6;
    auto stats          = compile_and_get_stats(repeated_reduce(n), cache_options(td.path));
    EXPECT(stats.reused >= n - 1);
    EXPECT(stats.compiled < n);
}

// The whole point of the cache: a program compiled once should not compile again. The second
// compile gets a fresh context, so an empty in-memory cache, and everything it reuses has to
// come back off disk.
TEST_CASE(second_compile_reads_from_disk)
{
    migraphx::tmp_dir td{"binary-cache"};
    auto options = cache_options(td.path);

    auto cold = compile_and_get_stats(pointwise_program(), options);
    EXPECT(cold.compiled > 0);
    EXPECT(cold.misses > 0);
    EXPECT(cold.hits == 0);

    auto warm = compile_and_get_stats(pointwise_program(), options);
    EXPECT(warm.hits == cold.misses);
    EXPECT(warm.compiled == 0);
    EXPECT(warm.misses == 0);
}

// Without a directory nothing is written, but results are still shared within the compile.
TEST_CASE(no_directory_keeps_results_in_memory)
{
    migraphx::tmp_dir td{"binary-cache"};
    auto stats = compile_and_get_stats(repeated_reduce(4), cache_options(""));
    EXPECT(stats.reused > 0);
    EXPECT(migraphx::fs::is_empty(td.path));
}

// A result taken from the cache has to produce the same numbers as one that was just compiled.
TEST_CASE(cached_results_match_reference)
{
    migraphx::tmp_dir td{"binary-cache"};
    auto options = cache_options(td.path);

    auto p_ref = pointwise_program();
    p_ref.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {4, 8}};
    auto x = migraphx::generate_argument(s, 0);
    auto y = migraphx::generate_argument(s, 1);

    auto ref_result = p_ref.eval({{"x", x}, {"y", y}}).back();

    // Compile once to fill the cache, then again so the result comes back out of it.
    auto warmup = pointwise_program();
    warmup.compile(migraphx::make_target("gpu"), options);

    auto t = migraphx::make_target("gpu");
    auto p = pointwise_program();
    p.compile(t, options);

    migraphx::parameter_map params;
    for(auto&& [name, shape] : p.get_parameter_shapes())
    {
        if(name == "x")
            params[name] = t.copy_to(x);
        else if(name == "y")
            params[name] = t.copy_to(y);
        else
            params[name] = t.allocate(shape);
    }
    auto gpu_result = t.copy_from(p.eval(params).back());

    std::vector<float> ref_vec;
    std::vector<float> gpu_vec;
    ref_result.visit([&](auto v) { ref_vec.assign(v.begin(), v.end()); });
    gpu_result.visit([&](auto v) { gpu_vec.assign(v.begin(), v.end()); });

    EXPECT(migraphx::verify::verify_rms_range(ref_vec, gpu_vec));
}

// With verification on, every reused result is checked against a fresh compile. A clean run
// means the keys really do capture what the compilers depend on.
TEST_CASE(verified_reuse_matches_fresh_compiles)
{
    migraphx::tmp_dir td{"binary-cache"};
    auto options = cache_options(td.path, /* verify */ true);

    compile_and_get_stats(pointwise_program(), options);
    auto stats = compile_and_get_stats(pointwise_program(), options);
    EXPECT(stats.hits > 0);
}

// A damaged entry must cost a recompile and nothing more.
TEST_CASE(corrupt_entry_is_ignored)
{
    migraphx::tmp_dir td{"binary-cache"};
    auto options = cache_options(td.path);

    auto p = pointwise_program();
    p.compile(migraphx::make_target("gpu"), options);

    std::size_t truncated = 0;
    for(const auto& entry : migraphx::fs::recursive_directory_iterator(td.path))
    {
        if(entry.path().extension() != ".mxr")
            continue;
        migraphx::write_buffer(entry.path(), std::vector<char>(8, 0));
        truncated++;
    }
    EXPECT(truncated > 0);

    auto stats = compile_and_get_stats(pointwise_program(), options);
    EXPECT(stats.hits == 0);
    EXPECT(stats.compiled > 0);
}

TEST_CASE(entry_round_trip)
{
    migraphx::gpu::compiled_code code;
    {
        auto* fm = code.fragment.get_main_module();
        auto x   = fm->add_parameter(migraphx::gpu::compiled_code::input_name(0),
                                     {migraphx::shape::float_type, {2}});
        fm->add_return({fm->add_instruction(migraphx::make_op("abs"), x)});
    }
    code.fill_map["float_type{2}"] = 3.0;

    migraphx::gpu::binary_cache_entry entry;
    entry.key      = "some-key";
    entry.op_name  = "pointwise";
    entry.solution = migraphx::value{{"algo", "block"}};
    entry.code     = code;

    auto buffer = migraphx::to_msgpack(migraphx::to_value(entry));
    migraphx::gpu::binary_cache_entry loaded;
    migraphx::from_value(migraphx::from_msgpack(buffer), loaded);

    EXPECT(loaded.key == entry.key);
    EXPECT(loaded.op_name == entry.op_name);
    EXPECT(loaded.solution == entry.solution);
    EXPECT(loaded.code.fill_map == entry.code.fill_map);
    EXPECT(*loaded.code.fragment.get_main_module() == *entry.code.fragment.get_main_module());
}

// The key has to cover everything handed to the compiler, not just the source text. Two
// kernels that differ only in their tensor views or launch bounds must not share an entry.
TEST_CASE(key_covers_more_than_the_source)
{
    migraphx::gpu::context ctx;
    migraphx::shape s{migraphx::shape::float_type, {4, 8}};

    migraphx::gpu::hip_src src;
    src.content             = "// kernel source";
    src.options.inputs      = {s, s};
    src.options.output      = s;
    src.options.global      = 1024;
    src.options.local       = 256;
    src.options.kernel_name = "kernel";
    auto base               = migraphx::gpu::hip_compile_key(ctx, src);

    EXPECT(base == migraphx::gpu::hip_compile_key(ctx, src));

    auto changed_source    = src;
    changed_source.content = "// different source";
    EXPECT(migraphx::gpu::hip_compile_key(ctx, changed_source) != base);

    auto changed_name                = src;
    changed_name.options.kernel_name = "other_kernel";
    EXPECT(migraphx::gpu::hip_compile_key(ctx, changed_name) != base);

    auto changed_global           = src;
    changed_global.options.global = 2048;
    EXPECT(migraphx::gpu::hip_compile_key(ctx, changed_global) != base);

    auto changed_local          = src;
    changed_local.options.local = 128;
    EXPECT(migraphx::gpu::hip_compile_key(ctx, changed_local) != base);

    auto changed_params = src;
    changed_params.options.emplace_param("-DEXTRA=1");
    EXPECT(migraphx::gpu::hip_compile_key(ctx, changed_params) != base);

    // Same source, but the generated tensor views differ.
    auto changed_views                   = src;
    changed_views.options.virtual_inputs = {{migraphx::shape::float_type, {32}},
                                            {migraphx::shape::float_type, {32}}};
    EXPECT(migraphx::gpu::hip_compile_key(ctx, changed_views) != base);
}

TEST_CASE(version_dir_is_stable)
{
    migraphx::gpu::context ctx;
    EXPECT(migraphx::gpu::binary_cache::version_dir(ctx) ==
           migraphx::gpu::binary_cache::version_dir(ctx));
    EXPECT(not migraphx::gpu::binary_cache::version_dir(ctx).empty());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
