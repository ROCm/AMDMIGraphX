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

#include <migraphx/fileutils.hpp>
#include <migraphx/tmp_dir.hpp>
#include <test.hpp>
#include <string_view>

namespace fs = migraphx::fs;

constexpr std::string_view baze_name{"test"};
constexpr std::string_view txt{".txt"};
constexpr std::string_view bz2{".bz2"};
constexpr std::string_view separator{": "};

#ifdef _WIN32
constexpr std::string_view executable_postfix{".exe"};
constexpr std::string_view library_prefix{""};
constexpr std::string_view shared_object_postfix{".dll"};
constexpr std::string_view static_library_postfix{".lib"};
constexpr std::string_view object_file_postfix{".obj"};
#else
constexpr std::string_view executable_postfix{""};
constexpr std::string_view library_prefix{"lib"};
constexpr std::string_view shared_object_postfix{".so"};
constexpr std::string_view static_library_postfix{".a"};
constexpr std::string_view object_file_postfix{".o"};
#endif

TEST_CASE(executable_filename)
{
    auto name = migraphx::make_executable_filename(baze_name);
    EXPECT(name == std::string{baze_name}.append(executable_postfix));
}

TEST_CASE(shared_object_filename)
{
    auto name = migraphx::make_shared_object_filename(baze_name);
    EXPECT(name == std::string{library_prefix}.append(baze_name).append(shared_object_postfix));
}

TEST_CASE(object_filename)
{
    auto name = migraphx::make_object_file_filename(baze_name);
    EXPECT(name == std::string{baze_name}.append(object_file_postfix));
}

TEST_CASE(static_library_filename)
{
    auto name = migraphx::make_static_library_filename(baze_name);
    EXPECT(name == std::string{library_prefix}.append(baze_name).append(static_library_postfix));
}

TEST_CASE(append_to_string)
{
    // 'using namespace' required for '+' operator
    using namespace migraphx::MIGRAPHX_INLINE_NS; // NOLINT
    auto cwd = fs::current_path();
    std::string prefix{baze_name};
    auto s1 = prefix + separator + cwd;
    EXPECT(s1 == prefix + separator + cwd.string());
    auto s2 = cwd + std::string{separator} + prefix;
    EXPECT(s2 == cwd.string() + separator + prefix);
}

TEST_CASE(append_file_extension)
{
    auto name    = fs::path{baze_name}.replace_extension(txt);
    auto updated = migraphx::MIGRAPHX_INLINE_NS::append_extension(name, bz2);
    EXPECT(updated == std::string{baze_name}.append(txt).append(bz2));
}

TEST_CASE(sanitize_filename_replaces_invalid_chars)
{
    // Each Windows-invalid character maps to a single '_'.
    EXPECT(migraphx::sanitize_filename("a:b") == "a_b");
    EXPECT(migraphx::sanitize_filename("a*b") == "a_b");
    EXPECT(migraphx::sanitize_filename("a?b") == "a_b");
    EXPECT(migraphx::sanitize_filename("a\\b") == "a_b");
    EXPECT(migraphx::sanitize_filename("a/b") == "a_b");
    EXPECT(migraphx::sanitize_filename("a<b>c|d\"e") == "a_b_c_d_e");
}

TEST_CASE(sanitize_filename_double_colon)
{
    // "::" in op names (e.g. "gpu::mlir_op") becomes "__".
    EXPECT(migraphx::sanitize_filename("gpu::mlir_op") == "gpu__mlir_op");
}

TEST_CASE(sanitize_filename_no_invalid_chars)
{
    // Valid names are returned unchanged.
    EXPECT(migraphx::sanitize_filename("gpu_mlir_op_8_123.mxr") == "gpu_mlir_op_8_123.mxr");
    EXPECT(migraphx::sanitize_filename("") == "");
}

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
