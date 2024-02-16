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

// Based on https://stackoverflow.com/a/59448568

namespace impl {
template <const std::string_view&, typename, const std::string_view&, typename>
struct concat;

template <const std::string_view& S1, std::size_t... I1,
          const std::string_view& S2, std::size_t... I2>
struct concat<S1, std::index_sequence<I1...>, S2, std::index_sequence<I2...>>
{
    static constexpr const char value[]{S1[I1]..., S2[I2]..., 0};
};
} // namespace impl

template <const std::string_view&...> struct join_strings_compile_time;
template <> struct join_strings_compile_time<>
{
    static constexpr std::string_view value{""};
};

template <const std::string_view& S1, const std::string_view& S2>
struct join_strings_compile_time<S1, S2>
{
    static constexpr std::string_view value =
        impl::concat<S1, std::make_index_sequence<S1.size()>,
                     S2, std::make_index_sequence<S2.size()>>::value;

};

template <const std::string_view& S, const std::string_view&... R>
struct join_strings_compile_time<S, R...>
{
    static constexpr std::string_view value =
        join_strings_compile_time<S, join_strings_compile_time<R...>::value>::value;
};

template <const std::string_view&... Strings>
static constexpr auto join_strings_v = join_strings_compile_time<Strings...>::value;

TEST_CASE(executable_filename)
{
    auto name = migraphx::make_executable_filename(baze_name);
    EXPECT(name == join_strings_v<baze_name, executable_postfix>);
}

TEST_CASE(shared_object_filename)
{
    auto name = migraphx::make_shared_object_filename(baze_name);
    EXPECT(name == join_strings_v<library_prefix, baze_name, shared_object_postfix>);
}

TEST_CASE(object_filename)
{
    auto name = migraphx::make_object_file_filename(baze_name);
    EXPECT(name == join_strings_v<baze_name, object_file_postfix>);
}

TEST_CASE(static_library_filename)
{
    auto name = migraphx::make_static_library_filename(baze_name);
    EXPECT(name == join_strings_v<library_prefix, baze_name, static_library_postfix>);
}

TEST_CASE(append_to_string)
{
    // 'using namespace' required for '+' operator
    using namespace migraphx::MIGRAPHX_INLINE_NS; // NOLINT
    auto cwd = fs::current_path();
    std::string prefix{join_strings_v<baze_name, separator>};
    auto str = prefix + cwd;
    EXPECT(str == prefix + cwd.string());
}

TEST_CASE(append_file_extension)
{
    fs::path name{join_strings_v<baze_name, txt>};
    auto updated = migraphx::MIGRAPHX_INLINE_NS::append_extension(name, bz2);
    EXPECT(updated == join_strings_v<baze_name, txt, bz2>);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
