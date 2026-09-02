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
 *
 */
#include <migraphx/gpu/file_binary_cache.hpp>
#include <migraphx/gpu/binary_cache_backend.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/tmp_dir.hpp>
#include <type_traits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

static_assert(std::is_constructible<binary_cache_backend, file_binary_cache>{},
              "file_binary_cache must satisfy the binary_cache_backend concept");

/// Where an entry lives. The caller guarantees a non-empty version, so entries compiled by
/// different toolchains can never land on the same path.
static fs::path entry_path(const fs::path& root,
                           const std::string& version,
                           const std::string& device,
                           const std::string& key_hash)
{
    return root / version / device / (key_hash + ".mxr");
}

/// Publish by rename so a reader never sees a half-written file. The temporary stays beside
/// the destination since the rename is only atomic within one filesystem.
static void write_atomically(const fs::path& dest, const std::vector<char>& content)
{
    tmp_dir td{"cache", dest.parent_path()};
    auto tmp = td.path / dest.filename();
    write_buffer(tmp, content);
    fs::rename(tmp, dest);
}

/// Record what this build is, so a directory full of hashes can be identified later.
static void write_stamp(const fs::path& dir, const std::string& stamp)
{
    auto path = dir / "cache.info";
    if(fs::exists(path))
        return;
    write_atomically(path, std::vector<char>(stamp.begin(), stamp.end()));
}

optional<std::vector<char>> file_binary_cache::load(const std::string& version,
                                                    const std::string& device,
                                                    const std::string& key_hash)
{
    auto path = entry_path(root, version, device, key_hash);
    try
    {
        if(not fs::exists(path))
            return nullopt;
        return read_buffer(path);
    }
    catch(const std::exception& ex)
    {
        // An unreadable entry is a miss, which costs a recompile and nothing else.
        log::warn() << "Failed to read binary cache entry " << path << ": " << ex.what();
        return nullopt;
    }
}

void file_binary_cache::store(const std::string& version,
                              const std::string& device,
                              const std::string& key_hash,
                              const binary_cache_entry&,
                              const std::vector<char>& blob)
{
    auto path = entry_path(root, version, device, key_hash);
    // The content is decided entirely by the key, so a writer that loses the publish race
    // replaces the file with the same bytes and no locking is needed.
    try
    {
        fs::create_directories(path.parent_path());
        write_stamp(root / version, stamp);
        write_atomically(path, blob);
    }
    catch(const std::exception& ex)
    {
        log::warn() << "Failed to write binary cache entry " << path << ": " << ex.what();
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
