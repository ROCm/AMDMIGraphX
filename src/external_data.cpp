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
#include <migraphx/logger.hpp>
#include <migraphx/md5.hpp>
#include <migraphx/tmp_dir.hpp>
#include <algorithm>
#include <bitset>
#include <cctype>
#include <fstream>
#include <iterator>
#include <numeric>
#include <set>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

constexpr std::size_t generation_id_size = 16;

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

// Whether p lies strictly within dir, comparing whole components
static bool is_within(const fs::path& dir, const fs::path& p)
{
    const auto m = std::mismatch(dir.begin(), dir.end(), p.begin(), p.end());
    return m.first == dir.end() and m.second != p.end();
}

fs::path resolve_external_data_path(const fs::path& base_dir, const fs::path& relative)
{
    const fs::path base     = lexically_normalize(base_dir);
    const fs::path resolved = lexically_normalize(base_dir / relative);
    // A '..' that climbs out of base breaks the prefix match
    if(not is_within(base, resolved))
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

fs::path weights_generation_path(const fs::path& base, const std::string& id)
{
    return base.parent_path() / (base.stem().string() + "." + id + base.extension().string());
}

bool is_weights_generation(const fs::path& base, const fs::path& p)
{
    if(p.parent_path() != base.parent_path())
        return false;
    const std::string name   = p.filename().string();
    const std::string prefix = base.stem().string() + ".";
    const std::string suffix = base.extension().string();
    if(name.size() != prefix.size() + generation_id_size + suffix.size())
        return false;
    if(name.compare(0, prefix.size(), prefix) != 0 or
       name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
        return false;
    auto id = name.begin() + prefix.size();
    return std::all_of(id, id + generation_id_size, [](unsigned char c) {
        return std::isdigit(c) or (c >= 'a' and c <= 'f');
    });
}

struct external_data_writer_impl
{
    explicit external_data_writer_impl(external_data_save_options o) : options(std::move(o))
    {
        if(options.weights_base.empty())
            return;
        generation = lexically_normalize(fs::absolute(weights_generation_path(
            options.weights_base, md5(unique_string("weights")).substr(0, generation_id_size))));
        // Beside the destination since the rename is only atomic within one filesystem
        tmp = generation.string() + ".tmp";
    }

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

    // The reference to p written into the program, recording that the program names it
    std::string reference(const fs::path& p)
    {
        auto target = lexically_normalize(fs::absolute(p));
        if(std::find(references.begin(), references.end(), target) == references.end())
            references.push_back(target);
        has_references = true;
        auto result    = relative_external_data_path(options.base_dir, target);
        // A reused reference holds the canonical path it was loaded from
        std::error_code ec;
        if(not options.base_dir.empty() and fs::path{result}.is_absolute())
        {
            auto canonical_base = fs::canonical(options.base_dir, ec);
            if(not ec)
                result = relative_external_data_path(canonical_base, target);
        }
        if(not options.base_dir.empty() and fs::path{result}.is_absolute() and
           warned_dirs.insert(target.parent_path().string()).second)
            log::warn() << "External data " << target.string() << " is outside "
                        << options.base_dir.string() << "; loading the program requires allowing "
                        << target.parent_path().string()
                        << " (external_data_allowed_paths, --external-data-allow)";
        return result;
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
            if(fs::exists(generation))
                MIGRAPHX_THROW("Weights file already exists: " + generation.string());
            fs::rename(tmp, generation);
            committed = true;
        }
        catch(...)
        {
            std::error_code ec;
            fs::remove(tmp, ec);
            throw;
        }
    }

    void discard()
    {
        if(not committed)
            return;
        std::error_code ec;
        fs::remove(generation, ec);
        committed = false;
    }

    external_data_save_options options;
    fs::path generation;
    fs::path tmp;
    std::ofstream os;
    std::size_t pos     = 0;
    bool has_references = false;
    bool committed      = false;
    std::vector<fs::path> references;
    std::set<std::string> warned_dirs;
};

// A reference can only be kept while its file still holds the bytes, which another process may
// have removed or truncated since it was loaded
static bool still_present(const external_data& ref)
{
    std::error_code ec;
    auto size = fs::file_size(ref.file_path, ec);
    return not ec and ref.offset <= size and ref.size <= size - ref.offset;
}

external_data_writer::external_data_writer(external_data_save_options options)
    : impl(std::make_unique<external_data_writer_impl>(std::move(options)))
{
    const auto& o = impl->options;
    if(std::bitset<64>(o.alignment).count() != 1)
        MIGRAPHX_THROW("External data alignment must be a nonzero power of two: " +
                       std::to_string(o.alignment));
    if(o.threshold.has_value() and o.weights_base.empty())
        MIGRAPHX_THROW("External data requires a weights file");
}

external_data_writer::~external_data_writer() = default;

std::optional<external_data>
external_data_writer::write(const char* data, std::size_t n, const external_data* original)
{
    const auto& o = impl->options;
    if(n == 0)
        return std::nullopt;
    if(o.reuse_external_data and original != nullptr and still_present(*original))
    {
        external_data result = *original;
        result.file_path     = impl->reference(original->file_path);
        if(result.checksum.empty())
            result.checksum = md5(std::string_view{data, n});
        return result;
    }
    if(not o.threshold.has_value() or n < *o.threshold)
        return std::nullopt;
    external_data result;
    result.offset    = impl->append(data, n);
    result.file_path = impl->reference(impl->generation);
    result.size      = n;
    result.checksum  = md5(std::string_view{data, n});
    return result;
}

void external_data_writer::commit() { impl->commit(); }

void external_data_writer::discard() { impl->discard(); }

bool external_data_writer::has_references() const { return impl->has_references; }

const std::vector<fs::path>& external_data_writer::referenced_files() const
{
    return impl->references;
}

static fs::path canonical_path(const fs::path& p)
{
    std::error_code ec;
    auto result = fs::canonical(p, ec);
    if(ec)
        MIGRAPHX_THROW("External data file does not exist: " + p.string());
    return result;
}

struct external_data_reader_impl
{
    external_data_load_options options;
    fs::path base;
    // Canonical forms, so a symlink cannot lead out of them; root is empty when base is missing
    fs::path root;
    std::vector<fs::path> allowed;
    std::unordered_map<std::string, std::shared_ptr<mapped_file>> files;
};

external_data_reader::external_data_reader(external_data_load_options options)
    : impl(std::make_unique<external_data_reader_impl>())
{
    impl->options = std::move(options);
    const auto& o = impl->options;
    impl->base    = o.base_dir.empty() ? fs::current_path() : fs::absolute(o.base_dir);
    std::error_code ec;
    impl->root = fs::canonical(impl->base, ec);
    if(ec)
        impl->root.clear();
    std::transform(o.allowed_dirs.begin(),
                   o.allowed_dirs.end(),
                   std::back_inserter(impl->allowed),
                   [](const fs::path& dir) {
                       std::error_code dir_ec;
                       auto result = fs::canonical(dir, dir_ec);
                       if(dir_ec)
                           MIGRAPHX_THROW("External data allowed path does not exist: " +
                                          dir.string());
                       return result;
                   });
}

external_data_reader::~external_data_reader() = default;

const external_data_load_options& external_data_reader::get_options() const
{
    return impl->options;
}

fs::path external_data_reader::resolve(const std::string& file_path) const
{
    const fs::path p{file_path};
    const fs::path target = p.is_absolute() ? p : impl->base / p;
    if(impl->root.empty())
        MIGRAPHX_THROW("External data directory does not exist: " + impl->base.string());
    auto resolved = canonical_path(target);
    if(is_within(impl->root, resolved) or
       std::any_of(impl->allowed.begin(), impl->allowed.end(), [&](const fs::path& dir) {
           return is_within(dir, resolved);
       }))
        return resolved;
    MIGRAPHX_THROW("External data file " + resolved.string() + " is outside " +
                   impl->root.string() + "; allow " + resolved.parent_path().string() +
                   " with external_data_allowed_paths (--external-data-allow) to load it");
}

std::shared_ptr<mapped_file> external_data_reader::map(const fs::path& resolved)
{
    auto& file = impl->files[resolved.string()];
    if(file == nullptr)
        file = std::make_shared<mapped_file>(resolved);
    return file;
}

std::pair<std::shared_ptr<mapped_file>, external_data>
external_data_reader::read(const external_data& ref)
{
    auto resolved = resolve(ref.file_path);
    auto file     = map(resolved);
    if(ref.offset > file->size() or ref.size > file->size() - ref.offset)
        MIGRAPHX_THROW("External data of " + std::to_string(ref.size) + " bytes at offset " +
                       std::to_string(ref.offset) + " exceeds size of file: " + resolved.string());
    if(impl->options.verify_checksum)
    {
        if(ref.checksum.empty())
            MIGRAPHX_THROW("External data of " + std::to_string(ref.size) + " bytes at offset " +
                           std::to_string(ref.offset) + " in file " + resolved.string() +
                           " has no checksum to verify");
        if(md5(std::string_view{file->data() + ref.offset, ref.size}) != ref.checksum)
            MIGRAPHX_THROW("External data checksum mismatch for " + std::to_string(ref.size) +
                           " bytes at offset " + std::to_string(ref.offset) +
                           " in file: " + resolved.string());
    }
    external_data result = ref;
    result.file_path     = resolved.string();
    return {file, result};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
