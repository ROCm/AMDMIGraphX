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
#include <algorithm>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#ifdef _WIN32
constexpr std::string_view executable_postfix{".exe"};
constexpr std::string_view library_prefix{""};
constexpr std::string_view library_postfix{".dll"};
constexpr std::string_view static_library_postfix{".lib"};
constexpr std::string_view object_file_postfix{".obj"};
#else
constexpr std::string_view executable_postfix{""};
constexpr std::string_view library_prefix{"lib"};
constexpr std::string_view library_postfix{".so"};
constexpr std::string_view static_library_postfix{".a"};
constexpr std::string_view object_file_postfix{".o"};
#endif

fs::path make_executable_filename(std::string_view name)
{
    return std::string{name}.append(executable_postfix);
}

fs::path make_shared_object_filename(std::string_view name)
{
    return std::string{library_prefix}.append(name).append(library_postfix);
}

fs::path make_object_file_filename(std::string_view name)
{
    return std::string{name}.append(object_file_postfix);
}

fs::path make_static_library_filename(std::string_view name)
{
    return std::string{library_prefix}.append(name).append(static_library_postfix);
}

fs::path append_extension(const fs::path& path, std::string_view ext)
{
    return fs::path{path}.replace_extension(path.extension().string().append(ext));
}

std::string sanitize_filename(std::string s)
{
    static constexpr std::string_view invalid = "<>:\"/\\|?*";
    std::replace_if(
        s.begin(), s.end(), [](char c) { return invalid.find(c) != std::string_view::npos; }, '_');
    return s;
}

#if MIGRAPHX_HAS_FILESYSTEM

fs::path weakly_canonical(const fs::path& path) { return fs::weakly_canonical(path); }

#else

// path::lexically_normal, which std::experimental::filesystem does not have either
static fs::path lexically_normal(const fs::path& path)
{
    auto parts = std::accumulate(
        path.begin(), path.end(), std::vector<fs::path>{}, [](auto acc, const fs::path& part) {
            // A trailing separator is an empty or a "." element
            if(part.empty() or part == ".")
                return acc;
            if(part != ".." or acc.empty() or acc.back() == "..")
                acc.push_back(part);
            // The parent of the root is the root, so a ".." after it is dropped
            else if(not acc.back().has_root_path())
                acc.pop_back();
            return acc;
        });
    if(parts.empty())
        return path.empty() ? "" : ".";
    return std::accumulate(parts.begin(), parts.end(), fs::path{}, [](fs::path acc, const auto& p) {
        return acc /= p;
    });
}

fs::path weakly_canonical(const fs::path& path)
{
    if(fs::exists(fs::status(path)))
        return fs::canonical(path);
    // The leading elements of the path that exist are resolved, and the rest appended to them
    fs::path head;
    auto missing  = std::find_if(path.begin(), path.end(), [&](const fs::path& part) {
        auto next = head / part;
        if(not fs::exists(fs::status(next)))
            return true;
        head = next;
        return false;
    });
    auto resolved = head.empty() ? head : fs::canonical(head);
    return lexically_normal(std::accumulate(
        missing, path.end(), resolved, [](fs::path acc, const auto& p) { return acc /= p; }));
}

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
