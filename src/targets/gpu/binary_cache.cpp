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
#include <migraphx/gpu/file_binary_cache.hpp>
#include <migraphx/gpu/sqlite_binary_cache.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/md5.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/stringutils.hpp>
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

/// An md5 digest, truncated for readability when a short one is requested.
static std::string digest(const std::string& s, bool use_short_digest)
{
    auto d = md5(s);
    if(use_short_digest)
        d.resize(12);
    return d;
}

/// The kernel headers compiled into this build. Taken from the embedded sources rather than
/// the files on disk, so it tracks what is actually compiled even when the build system has
/// not reconfigured.
static const std::string& kernels_source()
{
    static const std::string src = [] {
        std::stringstream ss;
        for(const auto& [path, content] : ::migraphx_kernels())
        {
            ss << path << "\n" << content << "\n";
        }
        return ss.str();
    }();
    return src;
}

static std::string make_version_id(bool use_short_digest)
{
    const auto& compiler = hip_compiler_version();
    if(compiler.empty())
        return {};
    // The version numbers make the id readable; the hash of the full version string
    // separates builds that share them, since it also covers the source revision.
    return std::string{binary_cache_format} + "-hip" + compiler.major + "." + compiler.minor +
           "." + digest(compiler.version, use_short_digest) + "-kernels" +
           digest(kernels_source(), use_short_digest) + "-rocmlir" + rocmlir_id;
}

const std::string& binary_cache::version_id(bool use_short_digest)
{
    static const std::string short_id = make_version_id(true);
    static const std::string long_id  = make_version_id(false);
    return use_short_digest ? short_id : long_id;
}

/// Entries are grouped by the device they were compiled for. This keeps the directory
/// self-describing; the arch, core count and wavefront size already reach the key through the
/// arch line, the launch bounds and the -D defines.
static std::string device_dir(const context& ctx)
{
    const auto& device = ctx.get_current_device();
    return to_c_id(device.get_device_name()) + "_cu" + std::to_string(device.get_cu_count()) +
           "_wf" + std::to_string(device.get_wavefront_size());
}

/// Turn a stored blob back into an entry. Any failure is a miss, so a damaged entry costs a
/// recompile.
static optional<binary_cache::entry>
decode_entry(const std::vector<char>& blob, const std::string& key, const std::string& key_hash)
{
    binary_cache::entry e;
    try
    {
        migraphx::from_value(from_msgpack(blob), e);
    }
    catch(const std::exception& ex)
    {
        log::warn() << "Ignoring unreadable binary cache entry " << key_hash << ": " << ex.what();
        return nullopt;
    }
    // Entries are addressed by a hash of the key, so the full key is checked here to make a
    // collision a miss rather than a wrong kernel.
    if(e.key != key)
    {
        log::warn() << "Ignoring binary cache entry with mismatched key: " << key_hash;
        return nullopt;
    }
    return e;
}

binary_cache::binary_cache(binary_cache_settings s) : settings(std::move(s)) {}

// The storage backend is selected by file type, the same rule make_problem_cache_backend applies
// in problem_cache.cpp: a ".db"/".sqlite" path is a SQLite database, anything else is a
// directory of entries. A directory is named with the short version id to keep paths short; a
// database records the full id, which is self-describing. Nothing is persisted when the compiler
// cannot be identified, since entries from different toolchains would be indistinguishable.
binary_cache_backend* binary_cache::get_backend()
{
    if(backend_opened)
        return backend.has_value() ? &*backend : nullptr;
    backend_opened   = true;
    const auto& path = settings.path;
    // Checked first so that a memory-only cache never compiles the version probe.
    if(path.empty())
        return nullptr;
    const bool database = ends_with(path, ".db") or ends_with(path, ".sqlite");
    version             = version_id(not database);
    if(version.empty())
        return nullptr;
    if(not database)
        backend = binary_cache_backend{file_binary_cache{path}};
    else if(auto db = sqlite_binary_cache::open(path))
        backend = binary_cache_backend{std::move(*db)};
    return backend.has_value() ? &*backend : nullptr;
}

binary_cache::store_batch::store_batch(binary_cache& c) : backend(c.get_backend())
{
    if(backend != nullptr)
        backend->begin_batch();
}

binary_cache::store_batch::~store_batch()
{
    if(backend != nullptr)
        backend->end_batch();
}

optional<compiled_code> binary_cache::get(const context& ctx, const std::string& key)
{
    if(key.empty())
        return nullopt;
    auto it = memo.find(key);
    if(it != memo.end())
    {
        counters.reused++;
        return it->second;
    }
    if(auto* b = get_backend())
    {
        // The key is the whole compile source, so it is hashed once for the lookup and any
        // diagnostics.
        auto key_hash = md5(key);
        auto blob     = b->load(version, device_dir(ctx), key_hash);
        if(blob.has_value())
        {
            auto e = decode_entry(*blob, key, key_hash);
            if(e.has_value())
            {
                counters.hits++;
                return memo.emplace(key, std::move(e->code)).first->second;
            }
        }
    }
    counters.misses++;
    return nullopt;
}

void binary_cache::insert(const context& ctx, entry e)
{
    if(e.key.empty())
        return;
    counters.compiled++;
    if(auto* b = get_backend())
    {
        auto key_hash = md5(e.key);
        try
        {
            // A failure to serialize or store is a warning, not a failed compile.
            auto blob = to_msgpack(migraphx::to_value(e));
            b->store(version, device_dir(ctx), key_hash, e, blob);
        }
        catch(const std::exception& ex)
        {
            log::warn() << "Failed to store binary cache entry " << key_hash << ": " << ex.what();
        }
    }
    memo[std::move(e.key)] = std::move(e.code);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
