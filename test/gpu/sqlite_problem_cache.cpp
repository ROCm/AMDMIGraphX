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
#include <migraphx/gpu/problem_cache_backend.hpp>
#include <migraphx/gpu/sqlite_problem_cache.hpp>
#include <migraphx/tmp_dir.hpp>
#include <migraphx/value.hpp>

#include "test.hpp"

namespace {

migraphx::gpu::cache_device_key make_key(const std::string& gfx, std::size_t cu)
{
    migraphx::gpu::cache_device_key k;
    k.device_name    = "test_device_" + gfx;
    k.gfx_name       = gfx;
    k.cu_count       = cu;
    k.wavefront_size = 32;
    return k;
}

migraphx::value make_problem(const std::string& name, std::size_t variant)
{
    return {{"name", name}, {"problem", migraphx::value{{"variant", variant}}}};
}

} // namespace

// --------------------------------------------------------------------------
// Round-trip: write entries, save, load into a fresh instance, verify.
// --------------------------------------------------------------------------
TEST_CASE(sqlite_problem_cache_round_trip)
{
    migraphx::tmp_dir td{"sqlite_problem_cache_round_trip"};
    auto path = (td.path / "cache.db").string();

    auto dk1   = make_key("gfx1201", 64);
    auto dk2   = make_key("gfx1100", 96);
    auto key_a = make_problem("gemm", 0);
    auto key_b = make_problem("conv", 1);
    auto sol_a = migraphx::value{{"block", std::size_t{256}}, {"kernel", "kA"}};
    auto sol_b = migraphx::value{{"block", std::size_t{128}}, {"kernel", "kB"}};

    migraphx::gpu::sqlite_problem_cache writer;
    writer.insert(dk1, key_a, sol_a);
    writer.insert(dk2, key_b, sol_b);
    writer.mark(dk1, key_b); // sentinel: tried, no solution
    writer.save(path);

    migraphx::gpu::sqlite_problem_cache reader;
    reader.load(path);

    EXPECT(reader.has(dk1, key_a));
    EXPECT(reader.has(dk2, key_b));
    EXPECT(reader.has(dk1, key_b)); // sentinel still "present"

    auto got_a = reader.get(dk1, key_a);
    EXPECT(bool(got_a));
    EXPECT(*got_a == sol_a);

    auto got_b = reader.get(dk2, key_b);
    EXPECT(bool(got_b));
    EXPECT(*got_b == sol_b);

    // Sentinel: get() returns a present optional whose value is null.
    auto sentinel = reader.get(dk1, key_b);
    EXPECT(bool(sentinel));
    EXPECT(sentinel->is_null());

    // Miss across buckets: dk2's bucket has no key_a.
    EXPECT(not reader.has(dk2, key_a));
    EXPECT(not bool(reader.get(dk2, key_a)));
}

// --------------------------------------------------------------------------
// load() from a non-existent path is a no-op (typical first-run case).
// --------------------------------------------------------------------------
TEST_CASE(sqlite_problem_cache_missing_file_is_noop)
{
    migraphx::tmp_dir td{"sqlite_problem_cache_missing"};
    auto path = (td.path / "does_not_exist.db").string();

    migraphx::gpu::sqlite_problem_cache c;
    c.load(path); // must not throw
    EXPECT(c.cache.empty());

    // Subsequent save() must create the file and round-trip cleanly.
    auto dk = make_key("gfx1201", 64);
    c.insert(dk, make_problem("gemm", 0), migraphx::value{{"kernel", "kX"}});
    c.save(path);

    migraphx::gpu::sqlite_problem_cache c2;
    c2.load(path);
    EXPECT(c2.has(dk, make_problem("gemm", 0)));
}

// --------------------------------------------------------------------------
// Save-overwrite: a second save() of a smaller cache must replace the
// previous file's rows (DELETE FROM solutions semantics), not accumulate.
// --------------------------------------------------------------------------
TEST_CASE(sqlite_problem_cache_save_overwrite)
{
    migraphx::tmp_dir td{"sqlite_problem_cache_overwrite"};
    auto path = (td.path / "cache.db").string();

    auto dk    = make_key("gfx1201", 64);
    auto key_a = make_problem("gemm", 0);
    auto key_b = make_problem("conv", 1);

    migraphx::gpu::sqlite_problem_cache c;
    c.insert(dk, key_a, migraphx::value{{"kernel", "kA"}});
    c.insert(dk, key_b, migraphx::value{{"kernel", "kB"}});
    c.save(path);

    // Reload, drop one entry, save again.
    migraphx::gpu::sqlite_problem_cache c2;
    c2.load(path);
    EXPECT(c2.has(dk, key_a));
    EXPECT(c2.has(dk, key_b));
    c2.cache[dk].erase(key_b);
    c2.save(path);

    // Third instance must see the smaller set.
    migraphx::gpu::sqlite_problem_cache c3;
    c3.load(path);
    EXPECT(c3.has(dk, key_a));
    EXPECT(not c3.has(dk, key_b));
}

// --------------------------------------------------------------------------
// Type-erasure round-trip: wrap sqlite_problem_cache in problem_cache_backend
// and exercise every concept member. Verifies the te.py-generated wrapper
// forwards correctly to a *different* concrete backend than json.
// --------------------------------------------------------------------------
TEST_CASE(problem_cache_backend_sqlite_type_erasure)
{
    migraphx::tmp_dir td{"problem_cache_backend_sqlite_te"};
    auto path = (td.path / "cache.db").string();

    migraphx::gpu::problem_cache_backend backend = migraphx::gpu::sqlite_problem_cache{};

    auto dk   = make_key("gfx1201", 64);
    auto key  = make_problem("gemm", 7);
    auto sol  = migraphx::value{{"block", std::size_t{512}}, {"kernel", "kY"}};
    auto miss = make_problem("never_inserted", 99);

    EXPECT(not backend.has(dk, key));
    EXPECT(not bool(backend.get(dk, key)));

    backend.insert(dk, key, sol);
    EXPECT(backend.has(dk, key));
    auto got = backend.get(dk, key);
    EXPECT(bool(got));
    EXPECT(*got == sol);

    backend.mark(dk, miss);
    auto m = backend.get(dk, miss);
    EXPECT(bool(m));
    EXPECT(m->is_null());

    backend.save(path);

    migraphx::gpu::problem_cache_backend reloaded = migraphx::gpu::sqlite_problem_cache{};
    reloaded.load(path);
    EXPECT(reloaded.has(dk, key));
    auto got2 = reloaded.get(dk, key);
    EXPECT(bool(got2));
    EXPECT(*got2 == sol);
}

// --------------------------------------------------------------------------
// any_cast lets callers recover the concrete sqlite backend.
// --------------------------------------------------------------------------
TEST_CASE(problem_cache_backend_sqlite_any_cast)
{
    migraphx::gpu::problem_cache_backend backend = migraphx::gpu::sqlite_problem_cache{};

    auto* concrete = backend.any_cast<migraphx::gpu::sqlite_problem_cache>();
    EXPECT(concrete != nullptr);

    // Casting to a sibling concrete type returns nullptr (no cross-type
    // recovery via the type-erased interface).
    auto* wrong = backend.any_cast<int>();
    EXPECT(wrong == nullptr);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
