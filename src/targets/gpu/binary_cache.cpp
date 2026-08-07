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
#include <migraphx/gpu/binary_cache.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/md5.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/tmp_dir.hpp>
#include <migraphx_kernels.hpp>
#include <sstream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Bump when the shape of a stored fragment changes, which happens when a compiler changes the
// instructions it replaces with or when a serialized operator gains or loses a field. Such a
// change is invisible to the key, since the source handed to the backend is unaffected.
static constexpr const char* binary_cache_format = "v1";

#ifdef MIGRAPHX_ROCMLIR_ID
static constexpr const char* rocmlir_id = MIGRAPHX_ROCMLIR_ID;
#else
static constexpr const char* rocmlir_id = "nomlir";
#endif

std::shared_ptr<binary_cache> make_binary_cache() { return std::make_shared<binary_cache>(); }

void binary_cache::record_reused()
{
    if(st)
        st->reused++;
}

void binary_cache::record_hit()
{
    if(st)
        st->hits++;
}

optional<compiled_code> binary_cache::record_miss()
{
    if(st)
        st->misses++;
    return nullopt;
}

void binary_cache::record_compiled()
{
    if(st)
        st->compiled++;
}

bool binary_cache::verify() const { return settings.verify; }

/// A digest of the kernel headers compiled into this build. Taken from the embedded sources
/// rather than the files on disk, so it tracks what is actually compiled even when the build
/// system has not reconfigured.
static const std::string& kernels_digest()
{
    static const std::string digest = [] {
        std::stringstream ss;
        for(const auto& [path, content] : ::migraphx_kernels())
        {
            ss << path << "\n" << content << "\n";
        }
        return md5(ss.str()).substr(0, 12);
    }();
    return digest;
}

const std::string& binary_cache::version_dir()
{
    static const std::string dir = [] {
        const auto& compiler = hip_compiler_version();
        if(compiler.empty())
            return std::string{};
        // The version numbers make the directory readable; the hash of the full version string
        // separates builds that share them, since it also covers the source revision.
        return std::string{binary_cache_format} + "-hip" + compiler.major + "." + compiler.minor +
               "." + md5(compiler.version).substr(0, 12) + "-kernels" + kernels_digest() +
               "-rocmlir" + rocmlir_id;
    }();
    return dir;
}

/// Entries are grouped by the device they were compiled for. The architecture is part of this
/// because it is passed to the compiler rather than appearing in the source; the core count and
/// wavefront size are here to keep the directory self-describing, since they already reach the
/// key through the launch bounds and -D defines in the parameter list.
static std::string device_dir(const context& ctx)
{
    const auto& device = ctx.get_current_device();
    return to_c_id(device.get_device_name()) + "_cu" + std::to_string(device.get_cu_count()) +
           "_wf" + std::to_string(device.get_wavefront_size());
}

/// Where an entry lives, or an empty path when the toolchain cannot be identified and entries
/// from different toolchains would be indistinguishable.
static fs::path entry_path(const fs::path& root, const context& ctx, const std::string& key)
{
    const auto& version = binary_cache::version_dir();
    if(version.empty())
        return {};
    return root / version / device_dir(ctx) / (md5(key) + ".mxr");
}

/// Record what this build is, so a directory full of hashes can be identified later.
static void write_stamp(const fs::path& dir)
{
    auto stamp = dir / "cache.info";
    if(fs::exists(stamp))
        return;
    std::stringstream ss;
    ss << "format: " << binary_cache_format << "\n";
    ss << "hip: " << hip_compiler_version().version << "\n";
    ss << "kernels: " << kernels_digest() << "\n";
    ss << "rocmlir: " << rocmlir_id << "\n";
    write_string(stamp, ss.str());
}

optional<compiled_code> binary_cache::get(const context& ctx, const std::string& key)
{
    if(key.empty())
        return nullopt;
    auto it = memo.find(key);
    if(it != memo.end())
    {
        record_reused();
        return it->second;
    }
    const auto& root = settings.path;
    if(root.empty())
        return record_miss();

    auto path = entry_path(root, ctx, key);
    if(path.empty() or not fs::exists(path))
        return record_miss();

    entry e;
    try
    {
        migraphx::from_value(from_msgpack(read_buffer(path)), e);
    }
    catch(const std::exception& ex)
    {
        // A damaged or stale entry is only worth a recompile, so treat every failure as a miss
        // and let the result be written over the top.
        log::warn() << "Ignoring unreadable binary cache entry " << path << ": " << ex.what();
        return record_miss();
    }
    if(e.key != key)
    {
        log::warn() << "Ignoring binary cache entry with mismatched key: " << path;
        return record_miss();
    }

    record_hit();
    memo[key] = std::move(e.code);
    return memo[key];
}

void binary_cache::insert(const context& ctx, const entry& e)
{
    if(e.key.empty())
        return;
    record_compiled();
    memo[e.key]      = e.code;
    const auto& root = settings.path;
    if(root.empty())
        return;

    auto path = entry_path(root, ctx, e.key);
    if(path.empty())
        return;
    // Publish by rename so a reader never sees a half-written entry. The content is decided
    // entirely by the key, so a writer that loses the race replaces the file with the same bytes
    // and no locking is needed.
    auto tmp = path;
    tmp += "." + unique_string("tmp");
    try
    {
        fs::create_directories(path.parent_path());
        write_stamp(path.parent_path().parent_path());
        write_buffer(tmp, to_msgpack(migraphx::to_value(e)));
        fs::rename(tmp, path);
    }
    catch(const std::exception& ex)
    {
        // Leaving the temporary behind would accumulate in the cache directory.
        std::error_code ec;
        fs::remove(tmp, ec);
        log::warn() << "Failed to write binary cache entry " << path << ": " << ex.what();
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
