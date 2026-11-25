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
#include <migraphx/errors.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/tmp_dir.hpp>
#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>
#include "test.hpp"

TEST_CASE(map_file_contents)
{
    migraphx::tmp_dir td{};
    auto filename = td.path / "data.bin";
    // Larger than a page so the mapping spans several
    std::vector<char> data(10000);
    std::iota(data.begin(), data.end(), 0);
    migraphx::write_buffer(filename, data);

    auto view = migraphx::map_file(filename);
    EXPECT(view.size() == data.size());
    EXPECT(view.begin() == view.data());
    EXPECT(std::equal(view.begin(), view.end(), data.begin(), data.end()));
}

TEST_CASE(map_file_outlives_view)
{
    migraphx::tmp_dir td{};
    auto filename = td.path / "data.bin";
    std::vector<char> data(10000);
    std::iota(data.begin(), data.end(), 0);
    migraphx::write_buffer(filename, data);

    // The mapping lasts as long as any pointer into it, not only the view that made it
    auto view = migraphx::map_file(filename);
    std::shared_ptr<const char> tail{view.ptr(), view.data() + 5000};
    view = {};
    EXPECT(std::equal(data.begin() + 5000, data.end(), tail.get()));
}

TEST_CASE(map_file_empty)
{
    migraphx::tmp_dir td{};
    auto filename = td.path / "empty.bin";
    migraphx::write_buffer(filename, std::vector<char>{});

    auto view = migraphx::map_file(filename);
    EXPECT(view.size() == 0);
    EXPECT(view.data() == nullptr);
    EXPECT(view.begin() == view.end());
}

TEST_CASE(map_file_missing)
{
    migraphx::tmp_dir td{};
    EXPECT(test::throws<migraphx::exception>([&] { migraphx::map_file(td.path / "missing.bin"); }));
}

TEST_CASE(map_file_directory)
{
    migraphx::tmp_dir td{};
    EXPECT(test::throws<migraphx::exception>([&] { migraphx::map_file(td.path); }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
