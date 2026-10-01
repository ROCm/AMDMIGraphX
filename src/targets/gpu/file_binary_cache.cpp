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
#include <migraphx/md5.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/tmp_dir.hpp>
#include <system_error>
#include <type_traits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

static_assert(std::is_constructible<binary_cache_backend, file_binary_cache>{},
              "file_binary_cache must satisfy the binary_cache_backend concept");

/// Where an entry lives. The key is the whole compile source, so it is hashed to keep the name
/// short. The caller guarantees a non-empty version, so entries compiled by different toolchains
/// can never land on the same path.
static fs::path entry_path(const fs::path& root,
                           const std::string& version,
                           const std::string& device,
                           const std::string& key)
{
    return root / version / device / (md5(key) + ".mxr");
}

/// Publish by rename so a reader never sees a half-written file. The temporary stays beside
/// the destination since the rename is only atomic within one filesystem. Its short random
/// suffix keeps concurrent writers of the same entry apart without lengthening an already deep
/// path, which on Windows must stay under MAX_PATH for std::ofstream to open it.
static void write_atomically(const fs::path& dest, const std::vector<char>& content)
{
    auto suffix = md5(unique_string("cache")).substr(0, 16);
    auto tmp    = dest.parent_path() / (dest.stem().string() + "." + suffix + ".tmp");
    try
    {
        write_buffer(tmp, content);
        fs::rename(tmp, dest);
    }
    catch(...)
    {
        std::error_code ec;
        fs::remove(tmp, ec);
        throw;
    }
}

optional<binary_cache_entry> file_binary_cache::load(const std::string& version,
                                                     const std::string& device,
                                                     const std::string& key) const
{
    auto path = entry_path(root, version, device, key);
    binary_cache_entry e;
    try
    {
        if(not fs::exists(path))
            return nullopt;
        migraphx::from_value(from_msgpack(read_buffer(path)), e);
    }
    catch(const std::exception& ex)
    {
        // An unreadable or damaged entry is a miss, which costs a recompile and nothing else.
        log::warn() << "Ignoring unreadable binary cache entry " << path << ": " << ex.what();
        return nullopt;
    }
    // Files are named by a hash of the key, so the full key is checked here to make a collision
    // a miss rather than a wrong kernel.
    if(e.key != key)
    {
        log::warn() << "Ignoring binary cache entry with mismatched key: " << path;
        return nullopt;
    }
    return e;
}

void file_binary_cache::store(const std::string& version,
                              const std::string& device,
                              const std::vector<binary_cache_entry>& entries) const
{
    for(const auto& e : entries)
    {
        auto path = entry_path(root, version, device, e.key);
        // The content is decided entirely by the key, so a writer that loses the publish race
        // replaces the file with the same bytes and no locking is needed.
        fs::create_directories(path.parent_path());
        write_atomically(path, to_msgpack(migraphx::to_value(e)));
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
