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

#include <migraphx/gpu/problem_cache.hpp>
#include <migraphx/tmp_dir.hpp>
#include <migraphx/value.hpp>
#include <migraphx/file_buffer.hpp>

#include "test.hpp"

namespace {

migraphx::gpu::cache_device_key make_key()
{
    migraphx::gpu::cache_device_key k;
    k.device_name    = "test_device_gfx1201";
    k.gfx_name       = "gfx1201";
    k.cu_count       = 64;
    k.wavefront_size = 32;
    return k;
}

migraphx::value make_problem(std::size_t variant) { return migraphx::value{{"variant", variant}}; }

} // namespace

// --------------------------------------------------------------------------
// load(path) writes to the same path on save() -- the path is "remembered".
// --------------------------------------------------------------------------
TEST_CASE(problem_cache_path_override_round_trip)
{
    migraphx::tmp_dir td{"problem_cache_path_override"};
    auto path = (td.path / "explicit.json").string();

    {
        migraphx::gpu::problem_cache writer;
        writer.set_device_key(make_key());
        writer.load(path); // path doesn't exist yet -- no-op load, but path is remembered
        writer.insert("gemm", make_problem(0), migraphx::value{{"kernel", "kA"}});
        writer.save(); // must write to `path` because path_override was set
    }

    EXPECT(migraphx::fs::exists(path));

    {
        migraphx::gpu::problem_cache reader;
        reader.set_device_key(make_key());
        reader.load(path);
        EXPECT(reader.has("gemm", make_problem(0)));
        auto v = reader.get("gemm", make_problem(0));
        EXPECT(bool(v));
        EXPECT((*v).at("kernel").to<std::string>() == "kA");
    }
}

// --------------------------------------------------------------------------
// load(empty path) is a no-op and leaves the cache empty (no env-var fallback
// from the path-arg overload).
// --------------------------------------------------------------------------
TEST_CASE(problem_cache_path_override_empty_is_noop)
{
    migraphx::gpu::problem_cache c;
    c.set_device_key(make_key());
    c.load(std::string{});
    EXPECT(not c.has("gemm", make_problem(7)));

    // Subsequent save() with no path-override and no env var must be a no-op
    // (no exception, no file written). We can't easily verify the negative
    // here, but we can verify the cache still works in-memory.
    c.insert("gemm", make_problem(7), migraphx::value{{"kernel", "kZ"}});
    EXPECT(c.has("gemm", make_problem(7)));
}

// --------------------------------------------------------------------------
// load(paths) with multiple files is a read-only priority list: has()/get()
// search the layers in order and the first hit wins (highest-priority first).
// This is the layered search that lives inside problem_cache (not context).
// --------------------------------------------------------------------------
TEST_CASE(problem_cache_layered_priority_first_hit_wins)
{
    migraphx::tmp_dir td{"problem_cache_layered"};
    auto high = (td.path / "high.json").string();
    auto low  = (td.path / "low.json").string();

    // High-priority file: solution kHigh for problem 0.
    {
        migraphx::gpu::problem_cache w;
        w.set_device_key(make_key());
        w.load(high);
        w.insert("gemm", make_problem(0), migraphx::value{{"kernel", "kHigh"}});
        w.save();
    }
    // Low-priority file: a *different* solution for problem 0, plus a problem 1
    // that exists only here.
    {
        migraphx::gpu::problem_cache w;
        w.set_device_key(make_key());
        w.load(low);
        w.insert("gemm", make_problem(0), migraphx::value{{"kernel", "kLow"}});
        w.insert("gemm", make_problem(1), migraphx::value{{"kernel", "kOnlyLow"}});
        w.save();
    }

    migraphx::gpu::problem_cache c;
    c.set_device_key(make_key());
    c.load(std::vector<std::string>{high, low}); // priority order: high first

    // Problem 0 is in both files -> the higher-priority file wins.
    EXPECT(c.has("gemm", make_problem(0)));
    auto s0 = c.get("gemm", make_problem(0));
    EXPECT(bool(s0));
    EXPECT((*s0).at("kernel").to<std::string>() == "kHigh");

    // Problem 1 exists only in the lower-priority file -> still found.
    EXPECT(c.has("gemm", make_problem(1)));
    auto s1 = c.get("gemm", make_problem(1));
    EXPECT(bool(s1));
    EXPECT((*s1).at("kernel").to<std::string>() == "kOnlyLow");

    // A problem in neither file is not found.
    EXPECT(not c.has("gemm", make_problem(2)));
    EXPECT(not bool(c.get("gemm", make_problem(2))));

    // Multiple files are a read-only list: save() must be a no-op (no writable
    // path is configured) and must not throw.
    c.save();
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
