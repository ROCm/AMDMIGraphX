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
#include <migraphx/external_data.hpp>
#include <migraphx/mapped_file.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/md5.hpp>
#include <migraphx/tmp_dir.hpp>
#include <algorithm>
#include <bitset>
#include <fstream>
#include <numeric>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static thread_local scoped_external_data_writer* active_writer = nullptr;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static thread_local scoped_external_data_reader* active_reader = nullptr;

// The Filesystem TS (std::experimental::filesystem, used on some older toolchains) lacks
// weakly_canonical and path::lexically_relative, so normalize by hand.
fs::path lexically_normalize(const fs::path& p)
{
    auto parts = std::accumulate(
        p.begin(), p.end(), std::vector<fs::path>{}, [](auto acc, const fs::path& part) {
            if(part == "." or part.empty())
                return acc;
            if(part == ".." and not acc.empty() and acc.back() != "..")
                acc.pop_back();
            else
                acc.push_back(part);
            return acc;
        });
    return std::accumulate(parts.begin(),
                           parts.end(),
                           fs::path{},
                           [](fs::path acc, const auto& part) { return acc /= part; });
}

fs::path resolve_external_data_path(const fs::path& base_dir, const fs::path& relative)
{
    const fs::path base     = lexically_normalize(base_dir);
    const fs::path resolved = lexically_normalize(base_dir / relative);
    // resolved must stay within base: base's components must be a strict prefix
    // of resolved's. A '..' that climbs out of base breaks the prefix match.
    const auto m = std::mismatch(base.begin(), base.end(), resolved.begin(), resolved.end());
    if(m.first != base.end() or m.second == resolved.end())
        MIGRAPHX_THROW("External data path escapes base directory: " + relative.string());
    return resolved;
}

std::string relative_external_data_path(const fs::path& base_dir, const fs::path& p)
{
    const fs::path target = lexically_normalize(fs::absolute(p));
    if(base_dir.empty())
        return target.generic_string();
    const fs::path base = lexically_normalize(fs::absolute(base_dir));
    const auto m        = std::mismatch(base.begin(), base.end(), target.begin(), target.end());
    if(m.first != base.end() or m.second == target.end())
        return target.generic_string();
    return std::accumulate(m.second,
                           target.end(),
                           fs::path{},
                           [](fs::path acc, const auto& part) { return acc /= part; })
        .generic_string();
}

struct external_data_writer_impl
{
    explicit external_data_writer_impl(external_data_save_options o) : options(std::move(o)) {}

    external_data_writer_impl(const external_data_writer_impl&)            = delete;
    external_data_writer_impl& operator=(const external_data_writer_impl&) = delete;

    ~external_data_writer_impl()
    {
        if(not os.is_open())
            return;
        os.close();
        std::error_code ec;
        fs::remove(tmp, ec);
    }

    // Offset of the n bytes, appended at the next aligned position after zero padding
    std::size_t append(const char* data, std::size_t n)
    {
        if(not os.is_open())
        {
            // Beside the destination since the rename is only atomic within one filesystem
            auto suffix = md5(unique_string("weights")).substr(0, 16);
            tmp         = options.weights_file.parent_path() /
                  (options.weights_file.filename().string() + "." + suffix + ".tmp");
            os.open(tmp, std::ios::out | std::ios::binary | std::ios::trunc);
            if(not os.is_open())
                MIGRAPHX_THROW("Failure opening file: " + tmp.string());
        }
        std::size_t offset = (pos + options.alignment - 1) / options.alignment * options.alignment;
        std::vector<char> padding(offset - pos, 0);
        os.write(padding.data(), padding.size());
        os.write(data, n);
        if(os.bad())
            MIGRAPHX_THROW("Error writing file: " + tmp.string());
        pos = offset + n;
        return offset;
    }

    void commit()
    {
        if(not os.is_open())
            return;
        os.close();
        try
        {
            if(os.fail())
                MIGRAPHX_THROW("Error writing file: " + tmp.string());
            fs::rename(tmp, options.weights_file);
        }
        catch(...)
        {
            std::error_code ec;
            fs::remove(tmp, ec);
            throw;
        }
    }

    external_data_save_options options;
    fs::path tmp;
    std::ofstream os;
    std::size_t pos     = 0;
    bool has_references = false;
};

scoped_external_data_writer::scoped_external_data_writer(external_data_save_options options)
    : impl(std::make_unique<external_data_writer_impl>(std::move(options))), previous(active_writer)
{
    const auto& o = impl->options;
    if(std::bitset<64>(o.alignment).count() != 1)
        MIGRAPHX_THROW("External data alignment must be a nonzero power of two: " +
                       std::to_string(o.alignment));
    if(o.threshold.has_value())
    {
        if(o.weights_file.empty())
            MIGRAPHX_THROW("External data requires a weights file");
        if(mapped_file::is_mapped(o.weights_file))
            MIGRAPHX_THROW("Cannot overwrite weights file " + o.weights_file.string() +
                           " while a loaded program maps it");
    }
    active_writer = this;
}

scoped_external_data_writer::~scoped_external_data_writer() { active_writer = previous; }

std::optional<external_data> scoped_external_data_writer::write(
    const char* data, std::size_t n, const std::optional<external_data>& original)
{
    const auto& o = impl->options;
    if(n == 0)
        return std::nullopt;
    if(o.reuse_external_data and original.has_value())
    {
        external_data result = *original;
        result.file_path     = relative_external_data_path(o.base_dir, original->file_path);
        if(result.checksum.empty())
            result.checksum = md5(std::string_view{data, n});
        impl->has_references = true;
        return result;
    }
    if(not o.threshold.has_value() or n < *o.threshold)
        return std::nullopt;
    external_data result;
    result.file_path     = relative_external_data_path(o.base_dir, o.weights_file);
    result.offset        = impl->append(data, n);
    result.size          = n;
    result.checksum      = md5(std::string_view{data, n});
    impl->has_references = true;
    return result;
}

void scoped_external_data_writer::commit() { impl->commit(); }

bool scoped_external_data_writer::has_references() const { return impl->has_references; }

scoped_external_data_writer* scoped_external_data_writer::active() { return active_writer; }

struct external_data_reader_impl
{
    external_data_load_options options;
    std::unordered_map<std::string, std::shared_ptr<mapped_file>> files;
};

scoped_external_data_reader::scoped_external_data_reader(external_data_load_options options)
    : impl(std::make_unique<external_data_reader_impl>()), previous(active_reader)
{
    impl->options = std::move(options);
    active_reader = this;
}

scoped_external_data_reader::~scoped_external_data_reader() { active_reader = previous; }

const external_data_load_options& scoped_external_data_reader::get_options() const
{
    return impl->options;
}

std::shared_ptr<mapped_file> scoped_external_data_reader::map(const fs::path& resolved)
{
    auto& file = impl->files[resolved.string()];
    if(file == nullptr)
        file = std::make_shared<mapped_file>(resolved);
    return file;
}

scoped_external_data_reader* scoped_external_data_reader::active() { return active_reader; }

std::pair<std::shared_ptr<mapped_file>, external_data> map_external_data(const external_data& ref)
{
    auto* reader = scoped_external_data_reader::active();
    const external_data_load_options defaults{};
    const auto& options = reader == nullptr ? defaults : reader->get_options();
    const fs::path p{ref.file_path};
    fs::path resolved;
    if(p.is_absolute())
    {
        resolved = lexically_normalize(p);
    }
    else
    {
        auto base = options.base_dir.empty() ? fs::current_path() : fs::absolute(options.base_dir);
        resolved  = resolve_external_data_path(base, p);
    }
    auto file = reader == nullptr ? std::make_shared<mapped_file>(resolved) : reader->map(resolved);
    if(ref.offset > file->size() or ref.size > file->size() - ref.offset)
        MIGRAPHX_THROW("External data of " + std::to_string(ref.size) + " bytes at offset " +
                       std::to_string(ref.offset) + " exceeds size of file: " + resolved.string());
    if(options.verify_checksum and not ref.checksum.empty() and
       md5(std::string_view{file->data() + ref.offset, ref.size}) != ref.checksum)
        MIGRAPHX_THROW("External data checksum mismatch for " + std::to_string(ref.size) +
                       " bytes at offset " + std::to_string(ref.offset) +
                       " in file: " + resolved.string());
    external_data result = ref;
    result.file_path     = resolved.string();
    return {file, result};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
