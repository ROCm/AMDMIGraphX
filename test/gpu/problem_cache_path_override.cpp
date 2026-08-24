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
// load({}, {empty path}) is a no-op: an empty writable path configures no
// cache and leaves it empty (no env-var fallback from the path arg).
// --------------------------------------------------------------------------
TEST_CASE(problem_cache_path_override_empty_is_noop)
{
    migraphx::gpu::problem_cache c;
    c.set_device_key(make_key());
    c.load(std::vector<std::string>{}, std::vector<std::string>{std::string{}});
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
        w.load(std::vector<std::string>{}, std::vector<std::string>{high});
        w.insert("gemm", make_problem(0), migraphx::value{{"kernel", "kHigh"}});
        w.save();
    }
    // Low-priority file: a *different* solution for problem 0, plus a problem 1
    // that exists only here.
    {
        migraphx::gpu::problem_cache w;
        w.set_device_key(make_key());
        w.load(std::vector<std::string>{}, std::vector<std::string>{low});
        w.insert("gemm", make_problem(0), migraphx::value{{"kernel", "kLow"}});
        w.insert("gemm", make_problem(1), migraphx::value{{"kernel", "kOnlyLow"}});
        w.save();
    }

    migraphx::gpu::problem_cache c;
    c.set_device_key(make_key());
    c.load(std::vector<std::string>{high, low},
           std::vector<std::string>{}); // read-only, high first

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

// --------------------------------------------------------------------------
// load(read_only, writable): both tiers together. The writable cache is
// searched first (a locally tuned solution wins over a read-only one), inserts
// and save() go to the writable cache only, and the read-only cache stays a
// never-written fallback.
// --------------------------------------------------------------------------
TEST_CASE(problem_cache_writable_over_read_only)
{
    migraphx::tmp_dir td{"problem_cache_two_options"};
    auto ro = (td.path / "read_only.json").string();
    auto rw = (td.path / "writable.json").string();

    // Seed the read-only file: a solution for problem 0 and a problem 1 that
    // lives only here.
    {
        migraphx::gpu::problem_cache w;
        w.set_device_key(make_key());
        w.load(std::vector<std::string>{}, std::vector<std::string>{ro});
        w.insert("gemm", make_problem(0), migraphx::value{{"kernel", "kReadOnly"}});
        w.insert("gemm", make_problem(1), migraphx::value{{"kernel", "kOnlyRO"}});
        w.save();
    }

    migraphx::gpu::problem_cache c;
    c.set_device_key(make_key());
    c.load(std::vector<std::string>{ro}, std::vector<std::string>{rw});

    // Snapshot the read-only file's raw bytes to prove save() never rewrites it.
    const auto ro_bytes_before = migraphx::read_string(ro);

    // Writable is empty at first, so a read-only-only problem is still found.
    EXPECT(c.has("gemm", make_problem(1)));
    EXPECT((*c.get("gemm", make_problem(1))).at("kernel").to<std::string>() == "kOnlyRO");

    // Insert for problem 0 goes to the writable cache and now wins over the
    // read-only entry for the same problem.
    c.insert("gemm", make_problem(0), migraphx::value{{"kernel", "kWritable"}});
    EXPECT((*c.get("gemm", make_problem(0))).at("kernel").to<std::string>() == "kWritable");
    c.save();

    // The read-only file is byte-for-byte unchanged after the writable save.
    EXPECT(migraphx::read_string(ro) == ro_bytes_before);

    // The read-only file still resolves to its original solution; the writable
    // file holds the new solution.
    {
        migraphx::gpu::problem_cache ro_reader;
        ro_reader.set_device_key(make_key());
        ro_reader.load(std::vector<std::string>{}, std::vector<std::string>{ro});
        EXPECT((*ro_reader.get("gemm", make_problem(0))).at("kernel").to<std::string>() ==
               "kReadOnly");
    }
    {
        migraphx::gpu::problem_cache rw_reader;
        rw_reader.set_device_key(make_key());
        rw_reader.load(std::vector<std::string>{}, std::vector<std::string>{rw});
        EXPECT(rw_reader.has("gemm", make_problem(0)));
        EXPECT((*rw_reader.get("gemm", make_problem(0))).at("kernel").to<std::string>() ==
               "kWritable");
    }
}

// --------------------------------------------------------------------------
// load({}, {writable}): writable-only configuration (no read-only layers).
// Lookup, insert, and save() all use the single writable cache.
// --------------------------------------------------------------------------
TEST_CASE(problem_cache_writable_only)
{
    migraphx::tmp_dir td{"problem_cache_writable_only"};
    auto rw = (td.path / "writable.json").string();

    migraphx::gpu::problem_cache c;
    c.set_device_key(make_key());
    c.load(std::vector<std::string>{}, std::vector<std::string>{rw});

    EXPECT(not c.has("gemm", make_problem(0)));
    c.insert("gemm", make_problem(0), migraphx::value{{"kernel", "kW"}});
    EXPECT(c.has("gemm", make_problem(0)));
    EXPECT((*c.get("gemm", make_problem(0))).at("kernel").to<std::string>() == "kW");
    c.save();

    // The solution persists to the writable file and reloads.
    migraphx::gpu::problem_cache reader;
    reader.set_device_key(make_key());
    reader.load(std::vector<std::string>{}, std::vector<std::string>{rw});
    EXPECT(reader.has("gemm", make_problem(0)));
    EXPECT((*reader.get("gemm", make_problem(0))).at("kernel").to<std::string>() == "kW");
}

// --------------------------------------------------------------------------
// load({}, {}): no cache configured. Lookup works in-memory, newly generated
// solutions stay in-memory, and save() is a no-op (no writable path) that must
// not throw or create a file.
// --------------------------------------------------------------------------
TEST_CASE(problem_cache_no_cache_config_is_noop)
{
    migraphx::gpu::problem_cache c;
    c.set_device_key(make_key());
    c.load(std::vector<std::string>{}, std::vector<std::string>{});

    EXPECT(not c.has("gemm", make_problem(0)));
    c.insert("gemm", make_problem(0), migraphx::value{{"kernel", "kMem"}});
    EXPECT(c.has("gemm", make_problem(0)));
    c.save(); // no writable cache -> no-op, must not throw
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
