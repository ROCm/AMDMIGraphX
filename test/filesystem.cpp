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

#include <migraphx/filesystem.hpp>
#include <migraphx/tmp_dir.hpp>
#include <test.hpp>

namespace fs = migraphx::fs;

TEST_CASE(weakly_canonical_existing_path)
{
    migraphx::tmp_dir td{};
    fs::create_directories(td.path / "a" / "b");
    EXPECT(migraphx::weakly_canonical(td.path / "a" / "b") == fs::canonical(td.path) / "a" / "b");
    // The . and .. of a path that exists are resolved
    EXPECT(migraphx::weakly_canonical(td.path / "a" / "." / "b" / "..") ==
           fs::canonical(td.path) / "a");
}

TEST_CASE(weakly_canonical_missing_file)
{
    migraphx::tmp_dir td{};
    // The directory is resolved and the name of the file that does not exist is kept
    EXPECT(migraphx::weakly_canonical(td.path / "missing.txt") ==
           fs::canonical(td.path) / "missing.txt");
}

TEST_CASE(weakly_canonical_missing_directories)
{
    migraphx::tmp_dir td{};
    EXPECT(migraphx::weakly_canonical(td.path / "x" / "y" / "z.txt") ==
           fs::canonical(td.path) / "x" / "y" / "z.txt");
}

TEST_CASE(weakly_canonical_normalizes_the_missing_part)
{
    migraphx::tmp_dir td{};
    fs::create_directories(td.path / "a");
    auto base = fs::canonical(td.path);
    EXPECT(migraphx::weakly_canonical(td.path / "a" / "x" / "." / "y.txt") ==
           base / "a" / "x" / "y.txt");
    EXPECT(migraphx::weakly_canonical(td.path / "a" / "x" / ".." / "y.txt") ==
           base / "a" / "y.txt");
    // A .. can take the path back above the part that exists
    EXPECT(migraphx::weakly_canonical(td.path / "a" / "x" / ".." / ".." / "y.txt") ==
           base / "y.txt");
}

TEST_CASE(weakly_canonical_two_names_for_a_file)
{
    migraphx::tmp_dir td{};
    fs::create_directories(td.path / "a");
    auto direct   = td.path / "a" / "model.mxr";
    auto indirect = td.path / "a" / ".." / "b" / ".." / "a" / "." / "model.mxr";
    EXPECT(migraphx::weakly_canonical(direct) == migraphx::weakly_canonical(indirect));
    EXPECT(migraphx::weakly_canonical(direct) !=
           migraphx::weakly_canonical(td.path / "a" / "weights.db"));
}

TEST_CASE(weakly_canonical_relative_path_with_nothing_existing)
{
    // No leading part exists to resolve, so the path stays relative and is only normalized
    fs::path missing{"migraphx-weakly-canonical-missing-dir"};
    EXPECT(migraphx::weakly_canonical(missing / "file.txt") == missing / "file.txt");
    EXPECT(migraphx::weakly_canonical(missing / "x" / ".." / "file.txt") == missing / "file.txt");
    EXPECT(migraphx::weakly_canonical(missing / "..") == ".");
    EXPECT(migraphx::weakly_canonical("").empty());
}

#ifndef _WIN32
TEST_CASE(weakly_canonical_resolves_symlinks)
{
    migraphx::tmp_dir td{};
    fs::create_directories(td.path / "real");
    fs::create_directory_symlink(td.path / "real", td.path / "link");
    EXPECT(migraphx::weakly_canonical(td.path / "link" / "missing.txt") ==
           fs::canonical(td.path) / "real" / "missing.txt");
}
#endif

int main(int argc, const char* argv[]) { test::run(argc, argv); }
