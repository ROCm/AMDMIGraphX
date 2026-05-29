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
#ifndef MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_BACKEND_HPP
#define MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_BACKEND_HPP

#include <migraphx/config.hpp>
#include <migraphx/gpu/export.h>
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <utility>
#include <cstdint>
#include <functional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// ============================================================================
// Data types
// ============================================================================

/// A single cache entry for bulk operations (import/export/migration).
struct cache_entry
{
    std::string device_key; // stable device identifier (may be empty for legacy)
    std::string name;
    std::string problem;
    std::string solution; // empty string = marked/WIP
};

/// Stable device key for cache namespace separation.
/// Only includes hardware properties that do not change with power state.
/// Clock frequencies, VRAM size, etc. are metadata-only (stored but not keyed on).
struct cache_device_key
{
    std::string gpu_arch;   // e.g. "gfx1100"
    int cu_count       = 0; // compute units
    int wavefront_size = 0; // warp/wavefront width

    bool empty() const { return gpu_arch.empty(); }
    bool operator==(const cache_device_key& other) const
    {
        return gpu_arch == other.gpu_arch and cu_count == other.cu_count and
               wavefront_size == other.wavefront_size;
    }
    bool operator!=(const cache_device_key& other) const { return not(*this == other); }
};

/// Convert device key to a stable string representation for storage.
/// Format: "gpu_arch|cu_count|wavefront_size" (e.g. "gfx1100|48|32")
inline std::string to_string(const cache_device_key& dk)
{
    if(dk.empty())
        return {};
    return dk.gpu_arch + "|" + std::to_string(dk.cu_count) + "|" +
           std::to_string(dk.wavefront_size);
}

/// Parse a device key string back into a struct.
/// Returns empty key on malformed input.
MIGRAPHX_GPU_EXPORT cache_device_key parse_device_key(const std::string& s);

/// Hash for cache_device_key (for use in unordered containers).
struct cache_device_key_hash
{
    std::size_t operator()(const cache_device_key& dk) const
    {
        auto h1 = std::hash<std::string>{}(dk.gpu_arch);
        auto h2 = std::hash<int>{}(dk.cu_count);
        auto h3 = std::hash<int>{}(dk.wavefront_size);
        return h1 ^ (h2 << 1U) ^ (h3 << 2U);
    }
};

/// Hardware metadata for the GPU that produced cache entries.
/// Populated once per session by querying HIP device properties.
/// Backends store this alongside entries for analytics and provenance.
/// Only gpu_arch, cu_count, wavefront_size are part of the device key;
/// the rest are metadata stored for diagnostics but NOT used in lookup.
struct cache_hw_metadata
{
    std::string gpu_arch;
    int cu_count            = 0;
    int graphics_clock_mhz  = 0;
    int memory_clock_mhz    = 0;
    int memory_bus_bits     = 0;
    std::int64_t vram_bytes = 0;
    int wavefront_size      = 0;
    int regs_per_block      = 0;
    int max_threads_per_cu  = 0;

    bool empty() const { return gpu_arch.empty(); }

    /// Extract the stable device key from full metadata.
    cache_device_key device_key() const { return {gpu_arch, cu_count, wavefront_size}; }
};

/// Backend statistics for debugging/monitoring.
struct backend_stats
{
    std::size_t entry_count     = 0;
    std::size_t file_size_bytes = 0;
    std::string storage_path;
    std::string backend_type;
};

// ============================================================================
// Type-erased backend wrapper
// ============================================================================

/// Type-erased problem cache backend.
///
/// Concrete backends (json_cache_backend, sqlite_cache_backend, etc.) do NOT
/// inherit from a common base class. Instead they satisfy a concept by providing
/// the required methods, and this wrapper type-erases them.
///
/// Required methods on a concrete backend T:
///   void open(const std::string& path, const cache_device_key& current_device)
///   void close()
///   bool has(const std::string& device_key, const std::string& name, const std::string& problem)
///   const std::optional<std::string> get(const std::string& device_key, const std::string& name,
///   const std::string& problem) const void insert(const std::string& device_key, const
///   std::string& name, const std::string& problem, const std::string& solution) void mark(const
///   std::string& device_key, const std::string& name, const std::string& problem) void save()
///   std::vector<cache_entry> all_entries() const
///   void load_entries(const std::vector<cache_entry>& entries)
///   std::size_t size() const
///   std::string backend_name() const
///   backend_stats stats() const
///   void set_hw_metadata(const cache_hw_metadata& meta)
///   const cache_hw_metadata& get_hw_metadata() const
class MIGRAPHX_GPU_EXPORT problem_cache_backend
{
    public:
    problem_cache_backend() = default;

    template <class Backend>
    explicit problem_cache_backend(Backend backend)
        : self_(std::make_unique<model<Backend>>(std::move(backend)))
    {
    }

    problem_cache_backend(problem_cache_backend&&) noexcept            = default;
    problem_cache_backend& operator=(problem_cache_backend&&) noexcept = default;

    explicit operator bool() const { return self_ != nullptr; }

    // -- Lifecycle --
    void open(const std::string& path, const cache_device_key& current_device)
    {
        self_->do_open(path, current_device);
    }
    void close() { self_->do_close(); }

    // -- Read operations (device_key is the string form) --
    bool
    has(const std::string& device_key, const std::string& name, const std::string& problem) const
    {
        return self_->do_has(device_key, name, problem);
    }

    std::optional<std::string>
    get(const std::string& device_key, const std::string& name, const std::string& problem) const
    {
        return self_->do_get(device_key, name, problem);
    }

    // -- Write operations --
    void insert(const std::string& device_key,
                const std::string& name,
                const std::string& problem,
                const std::string& solution)
    {
        self_->do_insert(device_key, name, problem, solution);
    }

    void mark(const std::string& device_key, const std::string& name, const std::string& problem)
    {
        self_->do_mark(device_key, name, problem);
    }

    // -- Persistence --
    void save() { self_->do_save(); }

    // -- Bulk operations --
    std::vector<cache_entry> all_entries() const { return self_->do_all_entries(); }
    void load_entries(const std::vector<cache_entry>& entries) { self_->do_load_entries(entries); }

    // -- Metadata --
    std::size_t size() const { return self_->do_size(); }
    std::string backend_name() const { return self_->do_backend_name(); }
    backend_stats stats() const { return self_->do_stats(); }

    void set_hw_metadata(const cache_hw_metadata& meta) { self_->do_set_hw_metadata(meta); }
    const cache_hw_metadata& get_hw_metadata() const { return self_->do_get_hw_metadata(); }

    private:
    struct concept_t
    {
        concept_t()                                                               = default;
        concept_t(const concept_t&)                                               = default;
        concept_t(concept_t&&)                                                    = default;
        concept_t& operator=(const concept_t&)                                    = default;
        concept_t& operator=(concept_t&&)                                         = default;
        virtual ~concept_t()                                                      = default;
        virtual void do_open(const std::string& path, const cache_device_key& dk) = 0;
        virtual void do_close()                                                   = 0;
        virtual bool
        do_has(const std::string& dk, const std::string& n, const std::string& p) const = 0;
        virtual std::optional<std::string>
        do_get(const std::string& dk, const std::string& n, const std::string& p) const         = 0;
        virtual void do_insert(const std::string& dk,
                               const std::string& n,
                               const std::string& p,
                               const std::string& s)                                            = 0;
        virtual void do_mark(const std::string& dk, const std::string& n, const std::string& p) = 0;
        virtual void do_save()                                                                  = 0;
        virtual std::vector<cache_entry> do_all_entries() const                                 = 0;
        virtual void do_load_entries(const std::vector<cache_entry>& entries)                   = 0;
        virtual std::size_t do_size() const                                                     = 0;
        virtual std::string do_backend_name() const                                             = 0;
        virtual backend_stats do_stats() const                                                  = 0;
        virtual void do_set_hw_metadata(const cache_hw_metadata& meta)                          = 0;
        virtual const cache_hw_metadata& do_get_hw_metadata() const                             = 0;
    };

    template <class Backend>
    struct model final : concept_t
    {
        Backend backend_;
        explicit model(Backend b) : backend_(std::move(b)) {}

        void do_open(const std::string& path, const cache_device_key& dk) override
        {
            backend_.open(path, dk);
        }
        void do_close() override { backend_.close(); }
        bool
        do_has(const std::string& dk, const std::string& n, const std::string& p) const override
        {
            return backend_.has(dk, n, p);
        }
        std::optional<std::string>
        do_get(const std::string& dk, const std::string& n, const std::string& p) const override
        {
            return backend_.get(dk, n, p);
        }
        void do_insert(const std::string& dk,
                       const std::string& n,
                       const std::string& p,
                       const std::string& s) override
        {
            backend_.insert(dk, n, p, s);
        }
        void do_mark(const std::string& dk, const std::string& n, const std::string& p) override
        {
            backend_.mark(dk, n, p);
        }
        void do_save() override { backend_.save(); }
        std::vector<cache_entry> do_all_entries() const override { return backend_.all_entries(); }
        void do_load_entries(const std::vector<cache_entry>& entries) override
        {
            backend_.load_entries(entries);
        }
        std::size_t do_size() const override { return backend_.size(); }
        std::string do_backend_name() const override { return backend_.backend_name(); }
        backend_stats do_stats() const override { return backend_.stats(); }
        void do_set_hw_metadata(const cache_hw_metadata& meta) override
        {
            backend_.set_hw_metadata(meta);
        }
        const cache_hw_metadata& do_get_hw_metadata() const override
        {
            return backend_.get_hw_metadata();
        }
    };

    std::unique_ptr<concept_t> self_;
};

// ============================================================================
// Factory functions
// ============================================================================

/// Factory: create a backend by type name ("json", "sqlite", "lmdb", "memory").
/// Throws std::runtime_error if the requested backend is not available.
MIGRAPHX_GPU_EXPORT problem_cache_backend make_cache_backend(const std::string& type);

/// Factory: create the default backend based on environment variables.
/// Uses MIGRAPHX_CACHE_BACKEND (json|sqlite|lmdb|memory) with "json" as default.
MIGRAPHX_GPU_EXPORT problem_cache_backend make_default_cache_backend();

/// Factory: create a backend using an explicit type string with env-var fallback.
/// If explicit_backend is non-empty, uses it (falls back to JSON for unknown types).
/// If explicit_backend is empty, delegates to make_default_cache_backend() (env var path).
MIGRAPHX_GPU_EXPORT problem_cache_backend
make_cache_backend_with_fallback(const std::string& explicit_backend);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_BACKEND_HPP
