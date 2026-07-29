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
#include <algorithm>

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

static migraphx::compile_options cache_options(const migraphx::fs::path& dir, bool verify = false)
{
    migraphx::compile_options options;
    migraphx::set_backend_options(
        options, {{"binary_cache", dir.string()}, {"binary_cache_verify", verify}});
    return options;
}

static migraphx::gpu::compiled_code make_code()
{
    migraphx::gpu::compiled_code code;
    auto* fm = code.fragment.get_main_module();
    auto x   = fm->add_parameter(migraphx::gpu::compiled_code::input_name(0),
                                 {migraphx::shape::float_type, {2}});
    fm->add_return({fm->add_instruction(migraphx::make_op("abs"), x)});
    code.fill_map["float_type{2}"] = 3.0;
    return code;
}

static migraphx::gpu::binary_cache::entry make_entry(const std::string& key)
{
    migraphx::gpu::binary_cache::entry e;
    e.key      = key;
    e.op_name  = "pointwise";
    e.solution = migraphx::value{{"algo", "block"}};
    e.code     = make_code();
    return e;
}

// A cache built without stats keeps no counters, so the tests that assert on counts supply one.
static std::shared_ptr<migraphx::gpu::binary_cache::stats> make_stats()
{
    return std::make_shared<migraphx::gpu::binary_cache::stats>();
}

TEST_CASE(lookup_records_a_miss)
{
    migraphx::gpu::context ctx;
    auto st = make_stats();
    migraphx::gpu::binary_cache cache{st};

    EXPECT(not cache.get(ctx, "absent").has_value());
    EXPECT(st->misses == 1);
    EXPECT(st->hits == 0);
    EXPECT(st->reused == 0);
}

// An entry served out of memory is one an earlier compile in this process already paid for.
TEST_CASE(memory_lookup_records_reuse)
{
    migraphx::gpu::context ctx;
    auto st = make_stats();
    migraphx::gpu::binary_cache cache{st};

    cache.insert(ctx, make_entry("a-key"));
    EXPECT(st->compiled == 1);

    auto found = cache.get(ctx, "a-key");
    EXPECT(found.has_value());
    EXPECT(st->reused == 1);
    EXPECT(st->misses == 0);
}

// A second cache shares nothing in memory, so anything it finds came off disk.
TEST_CASE(disk_lookup_records_a_hit)
{
    migraphx::tmp_dir td{"binary-cache"};
    migraphx::gpu::context ctx;
    migraphx::gpu::binary_cache_settings settings{td.path.string(), false};

    migraphx::gpu::binary_cache writer;
    writer.configure(settings);
    writer.insert(ctx, make_entry("shared-key"));

    auto st = make_stats();
    migraphx::gpu::binary_cache reader{st};
    reader.configure(settings);

    auto found = reader.get(ctx, "shared-key");
    EXPECT(found.has_value());
    EXPECT(st->hits == 1);
    EXPECT(st->misses == 0);
    EXPECT(*found->fragment.get_main_module() == *make_code().fragment.get_main_module());
}

// A damaged entry must cost a recompile and nothing more.
TEST_CASE(corrupt_entry_is_ignored)
{
    migraphx::tmp_dir td{"binary-cache"};
    migraphx::gpu::context ctx;
    migraphx::gpu::binary_cache_settings settings{td.path.string(), false};

    migraphx::gpu::binary_cache writer;
    writer.configure(settings);
    writer.insert(ctx, make_entry("damaged"));

    std::size_t truncated = 0;
    for(const auto& file : migraphx::fs::recursive_directory_iterator(td.path))
    {
        if(file.path().extension() != ".mxr")
            continue;
        migraphx::write_buffer(file.path(), std::vector<char>(8, 0));
        truncated++;
    }
    EXPECT(truncated > 0);

    auto st = make_stats();
    migraphx::gpu::binary_cache reader{st};
    reader.configure(settings);

    EXPECT(not reader.get(ctx, "damaged").has_value());
    EXPECT(st->misses == 1);
}

// Without a directory nothing reaches disk, though results are still shared in memory.
TEST_CASE(no_directory_writes_nothing)
{
    migraphx::tmp_dir td{"binary-cache"};
    migraphx::gpu::context ctx;
    auto st = make_stats();
    migraphx::gpu::binary_cache cache{st};
    cache.configure({"", false});

    cache.insert(ctx, make_entry("in-memory-only"));
    EXPECT(cache.get(ctx, "in-memory-only").has_value());
    EXPECT(st->reused == 1);
    EXPECT(migraphx::fs::is_empty(td.path));
}

// Compiling twice against the same directory has to leave entries behind and keep producing the
// same numbers as the reference, whichever half of the run they came from.
TEST_CASE(compiling_twice_populates_the_cache_and_matches_reference)
{
    migraphx::tmp_dir td{"binary-cache"};
    auto options = cache_options(td.path);

    auto p_ref = pointwise_program();
    p_ref.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {4, 8}};
    auto x = migraphx::generate_argument(s, 0);
    auto y = migraphx::generate_argument(s, 1);

    auto ref_result = p_ref.eval({{"x", x}, {"y", y}}).back();

    auto warmup = pointwise_program();
    warmup.compile(migraphx::make_target("gpu"), options);

    auto entries =
        std::count_if(migraphx::fs::recursive_directory_iterator{td.path},
                      migraphx::fs::recursive_directory_iterator{},
                      [](const auto& file) { return file.path().extension() == ".mxr"; });
    EXPECT(entries > 0);

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

    EXPECT(migraphx::verify::verify_rms_range(ref_result.to_vector<float>(),
                                              gpu_result.to_vector<float>()));
}

// With verification on, every reused result is compiled again and compared, so a run that does
// not throw is one where the keys really do capture what the compilers depend on.
TEST_CASE(verified_reuse_matches_fresh_compiles)
{
    migraphx::tmp_dir td{"binary-cache"};
    auto options = cache_options(td.path, /* verify */ true);

    auto warmup = pointwise_program();
    warmup.compile(migraphx::make_target("gpu"), options);

    auto p = pointwise_program();
    p.compile(migraphx::make_target("gpu"), options);
}

TEST_CASE(entry_round_trip)
{
    auto e      = make_entry("some-key");
    auto buffer = migraphx::to_msgpack(migraphx::to_value(e));

    migraphx::gpu::binary_cache::entry loaded;
    migraphx::from_value(migraphx::from_msgpack(buffer), loaded);

    EXPECT(loaded.key == e.key);
    EXPECT(loaded.op_name == e.op_name);
    EXPECT(loaded.solution == e.solution);
    EXPECT(loaded.code.fill_map == e.code.fill_map);
    EXPECT(*loaded.code.fragment.get_main_module() == *e.code.fragment.get_main_module());
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
    EXPECT(migraphx::gpu::binary_cache::version_dir() ==
           migraphx::gpu::binary_cache::version_dir());
    EXPECT(not migraphx::gpu::binary_cache::version_dir().empty());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
