/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <test.hpp>

namespace fs = migraphx::fs;

// NOLINTBEGIN
// Explicit use of #define to concatenate strings
// at compilation time instead of at runtime.
#define MIGRAPHX_TEST "test"
#define MIGRAPHX_TXT ".txt"
#define MIGRAPHX_BZ2 ".bz2"

#ifdef _WIN32
#define MIGRAPHX_EXE_POSTFIX ".exe"
#define MIGRAPHX_LIB_PREFIX
#define MIGRAPHX_DYNAMIC_POSTFIX ".dll"
#define MIGRAPHX_STATIC_POSTFIX ".lib"
#define MIGRAPHX_OBJECT_POSTFIX ".obj"
#else
#define MIGRAPHX_EXE_POSTFIX
#define MIGRAPHX_LIB_PREFIX "lib"
#define MIGRAPHX_DYNAMIC_POSTFIX ".so"
#define MIGRAPHX_STATIC_POSTFIX ".a"
#define MIGRAPHX_OBJECT_POSTFIX ".o"
#endif
// NOLINTEND

TEST_CASE(executable_filename)
{
    auto name = migraphx::make_executable_filename(MIGRAPHX_TEST);
    EXPECT(name == MIGRAPHX_TEST MIGRAPHX_EXE_POSTFIX);
}

TEST_CASE(shared_object_filename)
{
    auto name = migraphx::make_shared_object_filename(MIGRAPHX_TEST);
    EXPECT(name == MIGRAPHX_LIB_PREFIX MIGRAPHX_TEST MIGRAPHX_DYNAMIC_POSTFIX);
}

TEST_CASE(object_filename)
{
    auto name = migraphx::make_object_file_filename(MIGRAPHX_TEST);
    EXPECT(name == MIGRAPHX_TEST MIGRAPHX_OBJECT_POSTFIX);
}

TEST_CASE(static_library_filename)
{
    auto name = migraphx::make_static_library_filename(MIGRAPHX_TEST);
    EXPECT(name == MIGRAPHX_LIB_PREFIX MIGRAPHX_TEST MIGRAPHX_STATIC_POSTFIX);
}

TEST_CASE(append_to_string)
{
    using namespace migraphx::MIGRAPHX_INLINE_NS; // NOLINT
    auto cwd = fs::current_path();
    auto str = MIGRAPHX_TEST ": " + cwd;
    EXPECT(str == std::string{MIGRAPHX_TEST ": "} + cwd.string());
}

TEST_CASE(append_file_extension)
{
    using namespace migraphx::MIGRAPHX_INLINE_NS; // NOLINT
    fs::path name{MIGRAPHX_TEST MIGRAPHX_TXT};
    auto updated = append_extension(name, MIGRAPHX_BZ2);
    EXPECT(updated == MIGRAPHX_TEST MIGRAPHX_TXT MIGRAPHX_BZ2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
