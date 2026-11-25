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
#include <migraphx/mapped_file.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/tmp_dir.hpp>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>
#include "test.hpp"

TEST_CASE(maps_contents)
{
    migraphx::tmp_dir td{"mapped-file"};
    auto p = td.path / "data.bin";
    std::vector<char> content{'a', 'b', 'c', 'd', 'e'};
    migraphx::write_buffer(p, content);

    migraphx::mapped_file f{p};
    EXPECT(f.size() == content.size());
    EXPECT(f.path() == p);
    EXPECT(f.data() != nullptr);
    EXPECT(std::vector<char>(f.data(), f.data() + f.size()) == content);
}

TEST_CASE(maps_empty_file)
{
    migraphx::tmp_dir td{"mapped-file"};
    auto p = td.path / "empty.bin";
    migraphx::write_buffer(p, std::vector<char>{});

    migraphx::mapped_file f{p};
    EXPECT(f.size() == 0);
    EXPECT(f.data() == nullptr);
}

TEST_CASE(missing_file_throws)
{
    migraphx::tmp_dir td{"mapped-file"};
    EXPECT(test::throws([&] { migraphx::mapped_file f{td.path / "missing.bin"}; }));
    EXPECT(not migraphx::mapped_file::is_mapped(td.path / "missing.bin"));
}

// The mapping is read-only, so its data cannot be written through
static_assert(
    std::is_same<decltype(std::declval<const migraphx::mapped_file&>().data()), const char*>{});

TEST_CASE(replace_unless_mapped_refuses_mapped_file)
{
    migraphx::tmp_dir td{"mapped-file"};
    auto p = td.path / "data.bin";
    std::vector<char> content{'a', 'b', 'c'};
    migraphx::write_buffer(p, content);
    bool replaced = false;
    {
        migraphx::mapped_file f{p};
        EXPECT(test::throws(
            [&] { migraphx::mapped_file::replace_unless_mapped(p, [&] { replaced = true; }); }));
        EXPECT(not replaced);
    }
    migraphx::mapped_file::replace_unless_mapped(p, [&] { replaced = true; });
    EXPECT(replaced);
    EXPECT(migraphx::read_buffer(p) == content);
}

TEST_CASE(replace_unless_mapped_missing_file)
{
    migraphx::tmp_dir td{"mapped-file"};
    auto p = td.path / "new.bin";
    migraphx::mapped_file::replace_unless_mapped(
        p, [&] { migraphx::write_buffer(p, std::vector<char>{'a'}); });
    EXPECT(migraphx::read_buffer(p) == std::vector<char>{'a'});
}

#ifdef __linux__
TEST_CASE(write_buffer_reports_full_disk)
{
    // The write is buffered, so only the flush on close can fail
    EXPECT(test::throws([] { migraphx::write_buffer("/dev/full", std::vector<char>(16, 'x')); }));
}
#endif

TEST_CASE(is_mapped_tracks_lifetime)
{
    migraphx::tmp_dir td{"mapped-file"};
    auto p = td.path / "data.bin";
    migraphx::write_buffer(p, std::vector<char>{'a'});
    EXPECT(not migraphx::mapped_file::is_mapped(p));
    {
        migraphx::mapped_file f1{p};
        EXPECT(migraphx::mapped_file::is_mapped(p));
        EXPECT(migraphx::mapped_file::is_mapped(td.path / "." / "data.bin"));
        {
            migraphx::mapped_file f2{p};
            EXPECT(migraphx::mapped_file::is_mapped(p));
        }
        EXPECT(migraphx::mapped_file::is_mapped(p));
    }
    EXPECT(not migraphx::mapped_file::is_mapped(p));
}

TEST_CASE(is_mapped_other_file)
{
    migraphx::tmp_dir td{"mapped-file"};
    auto p1 = td.path / "a.bin";
    auto p2 = td.path / "b.bin";
    migraphx::write_buffer(p1, std::vector<char>{'a'});
    migraphx::write_buffer(p2, std::vector<char>{'b'});
    migraphx::mapped_file f{p1};
    EXPECT(migraphx::mapped_file::is_mapped(p1));
    EXPECT(not migraphx::mapped_file::is_mapped(p2));
}

TEST_CASE(mapping_is_page_aligned)
{
    migraphx::tmp_dir td{"mapped-file"};
    auto p = td.path / "data.bin";
    migraphx::write_buffer(p, std::vector<char>(100, 'x'));
    migraphx::mapped_file f{p};
    EXPECT(reinterpret_cast<std::uintptr_t>(f.data()) % 4096 == 0);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
