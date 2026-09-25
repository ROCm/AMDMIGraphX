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
//
// te.py DSL for migraphx::gpu::binary_cache_backend.
//
// The generated header lives at
// src/targets/gpu/include/migraphx/gpu/binary_cache_backend.hpp; regenerate it
// with `cd tools && python generate.py` (generate_all routes include/gpu/ inputs
// into the gpu target tree). Do not edit the generated header by hand.
//
// Any type T satisfies the binary_cache_backend concept if it provides the
// member functions listed below; begin_batch and end_batch are optional and
// default to doing nothing. The wrapper holds T by shared_ptr and forwards
// each call through a virtual dispatch, matching problem_cache_backend.
//
// Notes:
//   * binary_cache_entry is defined in <migraphx/gpu/binary_cache_entry.hpp>;
//     the include below pulls in its full definition.
//   * Backends typically own non-trivial resources (a cache directory, a SQLite
//     connection) and are not meaningfully copyable beyond shared ownership.
//   * The members are non-const: binary_cache::get and insert are themselves
//     non-const, so nothing forces a const qualifier here, and a backend that
//     tracks an open batch needs the mutability. A backend may still declare
//     them const.
//
#ifndef MIGRAPHX_GUARD_GPU_BINARY_CACHE_BACKEND_HPP
#define MIGRAPHX_GUARD_GPU_BINARY_CACHE_BACKEND_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <migraphx/config.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/gpu/export.h>
#include <migraphx/gpu/binary_cache_entry.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

#ifdef DOXYGEN

/// Type-erased interface for binary-cache storage backends.
///
/// A backend persists serialized binary_cache_entry blobs to some medium (a
/// directory of files, a SQLite database, an in-memory map for tests). Entries
/// are addressed by three strings the caller has already computed:
///
///   * `version` -- binary_cache::version_id(), identifying the toolchain and
///     the embedded kernel sources that produced the entry. Never empty; the
///     caller skips persistence entirely when it is. The short form is used
///     for directories and the full form for databases.
///   * `device`  -- the GPU the entry was compiled for.
///   * `key_hash` -- md5 of the compile key. A hash rather than the key itself
///     because a file backend needs a short name; a collision is harmless,
///     since the caller re-checks the full key against the decoded entry.
///
/// Together these three form the identity of an entry. A backend must keep
/// entries with different scopes distinct rather than overwriting across them.
struct binary_cache_backend
{
    /// Return the serialized entry for this key, or nullopt for a miss.
    ///
    /// nullopt also covers every failure: a missing file, an unreadable
    /// database, a permissions problem. A cache that cannot be read is not an
    /// error, it is a cache miss, and the caller recompiles.
    ///
    /// Must not throw.
    optional<std::vector<char>>
    load(const std::string& version, const std::string& device, const std::string& key_hash);

    /// Persist `blob`, the msgpack encoding of `e`, under this key.
    ///
    /// `e` is passed alongside `blob` so a backend may denormalize op_name,
    /// problem and solution into queryable columns. Those fields are also
    /// inside `blob`, which stays the authoritative record -- a backend that
    /// stores them separately must still be able to answer a load() with the
    /// blob alone.
    ///
    /// Overwriting an existing entry is expected and safe: the content is
    /// decided entirely by the key, so a writer that loses a race replaces the
    /// entry with equivalent bytes.
    ///
    /// Must not throw. A failure to store costs a recompile next run, nothing
    /// more, and the caller still keeps the result in memory.
    void store(const std::string& version,
               const std::string& device,
               const std::string& key_hash,
               const binary_cache_entry& e,
               const std::vector<char>& blob);

    /// Mark the start of a run of stores that may be committed together, such as
    /// a database transaction, rather than one at a time. Every begin_batch is
    /// followed by an end_batch, and the two are never nested. Optional: a
    /// backend without them stores each entry as it comes.
    ///
    /// Must not throw. A backend that cannot start a batch stores entries one
    /// at a time instead.
    void begin_batch();

    /// Commit the stores made since begin_batch.
    ///
    /// Must not throw. A failed commit costs those entries a recompile next
    /// run, nothing more, and must not leave anything locked.
    void end_batch();
};

#else

#ifdef TYPE_ERASED_DECLARATION

// Type-erased interface for:
struct MIGRAPHX_EXPORT binary_cache_backend
{
    //
    optional<std::vector<char>>
    load(const std::string& version, const std::string& device, const std::string& key_hash);
    //
    void store(const std::string& version,
               const std::string& device,
               const std::string& key_hash,
               const binary_cache_entry& e,
               const std::vector<char>& blob);
    // (optional)
    void begin_batch();
    // (optional)
    void end_batch();
};

#else
// NOLINTBEGIN(performance-unnecessary-value-param)
struct binary_cache_backend
{
    private:
    template <class T>
    static auto private_detail_te_default_begin_batch(char, T&& private_detail_te_self)
        -> decltype(private_detail_te_self.begin_batch())
    {
        private_detail_te_self.begin_batch();
    }

    template <class T>
    static void private_detail_te_default_begin_batch(float, T&& private_detail_te_self)
    {
        migraphx::nop(private_detail_te_self);
    }

    template <class T>
    static auto private_detail_te_default_end_batch(char, T&& private_detail_te_self)
        -> decltype(private_detail_te_self.end_batch())
    {
        private_detail_te_self.end_batch();
    }

    template <class T>
    static void private_detail_te_default_end_batch(float, T&& private_detail_te_self)
    {
        migraphx::nop(private_detail_te_self);
    }

    template <class PrivateDetailTypeErasedT>
    struct private_te_unwrap_reference
    {
        using type = PrivateDetailTypeErasedT;
    };
    template <class PrivateDetailTypeErasedT>
    struct private_te_unwrap_reference<std::reference_wrapper<PrivateDetailTypeErasedT>>
    {
        using type = PrivateDetailTypeErasedT;
    };
    template <class PrivateDetailTypeErasedT>
    using private_te_pure = typename std::remove_cv<
        typename std::remove_reference<PrivateDetailTypeErasedT>::type>::type;

    template <class PrivateDetailTypeErasedT>
    using private_te_constraints_impl =
        decltype(std::declval<PrivateDetailTypeErasedT>().load(std::declval<const std::string&>(),
                                                               std::declval<const std::string&>(),
                                                               std::declval<const std::string&>()),
                 std::declval<PrivateDetailTypeErasedT>().store(
                     std::declval<const std::string&>(),
                     std::declval<const std::string&>(),
                     std::declval<const std::string&>(),
                     std::declval<const binary_cache_entry&>(),
                     std::declval<const std::vector<char>&>()),
                 private_detail_te_default_begin_batch(char(0),
                                                       std::declval<PrivateDetailTypeErasedT>()),
                 private_detail_te_default_end_batch(char(0),
                                                     std::declval<PrivateDetailTypeErasedT>()),
                 void());

    template <class PrivateDetailTypeErasedT>
    using private_te_constraints = private_te_constraints_impl<
        typename private_te_unwrap_reference<private_te_pure<PrivateDetailTypeErasedT>>::type>;

    public:
    // Constructors
    binary_cache_backend() = default;

    template <typename PrivateDetailTypeErasedT,
              typename = private_te_constraints<PrivateDetailTypeErasedT>,
              typename = typename std::enable_if<
                  not std::is_same<private_te_pure<PrivateDetailTypeErasedT>,
                                   binary_cache_backend>{}>::type>
    binary_cache_backend(PrivateDetailTypeErasedT&& value)
        : private_detail_te_handle_mem_var(
              std::make_shared<
                  private_detail_te_handle_type<private_te_pure<PrivateDetailTypeErasedT>>>(
                  std::forward<PrivateDetailTypeErasedT>(value)))
    {
    }

    // Assignment
    template <typename PrivateDetailTypeErasedT,
              typename = private_te_constraints<PrivateDetailTypeErasedT>,
              typename = typename std::enable_if<
                  not std::is_same<private_te_pure<PrivateDetailTypeErasedT>,
                                   binary_cache_backend>{}>::type>
    binary_cache_backend& operator=(PrivateDetailTypeErasedT && value)
    {
        using std::swap;
        auto* derived = this->any_cast<private_te_pure<PrivateDetailTypeErasedT>>();
        if(derived and private_detail_te_handle_mem_var.use_count() == 1)
        {
            *derived = std::forward<PrivateDetailTypeErasedT>(value);
        }
        else
        {
            binary_cache_backend rhs(value);
            swap(private_detail_te_handle_mem_var, rhs.private_detail_te_handle_mem_var);
        }
        return *this;
    }

    // Cast
    template <typename PrivateDetailTypeErasedT>
    PrivateDetailTypeErasedT* any_cast()
    {
        return this->type_id() == typeid(PrivateDetailTypeErasedT)
                   ? std::addressof(static_cast<private_detail_te_handle_type<
                                        typename std::remove_cv<PrivateDetailTypeErasedT>::type>&>(
                                        private_detail_te_get_handle())
                                        .private_detail_te_value)
                   : nullptr;
    }

    template <typename PrivateDetailTypeErasedT>
    const typename std::remove_cv<PrivateDetailTypeErasedT>::type* any_cast() const
    {
        return this->type_id() == typeid(PrivateDetailTypeErasedT)
                   ? std::addressof(static_cast<const private_detail_te_handle_type<
                                        typename std::remove_cv<PrivateDetailTypeErasedT>::type>&>(
                                        private_detail_te_get_handle())
                                        .private_detail_te_value)
                   : nullptr;
    }

    const std::type_info& type_id() const
    {
        if(private_detail_te_handle_empty())
            return typeid(std::nullptr_t);
        else
            return private_detail_te_get_handle().type();
    }

    optional<std::vector<char>>
    load(const std::string& version, const std::string& device, const std::string& key_hash)
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().load(version, device, key_hash);
    }

    void store(const std::string& version,
               const std::string& device,
               const std::string& key_hash,
               const binary_cache_entry& e,
               const std::vector<char>& blob)
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().store(version, device, key_hash, e, blob);
    }

    void begin_batch()
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().begin_batch();
    }

    void end_batch()
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().end_batch();
    }

    friend bool is_shared(const binary_cache_backend& private_detail_x,
                          const binary_cache_backend& private_detail_y)
    {
        return private_detail_x.private_detail_te_handle_mem_var ==
               private_detail_y.private_detail_te_handle_mem_var;
    }

    private:
    struct private_detail_te_handle_base_type
    {
        virtual ~private_detail_te_handle_base_type() {}
        virtual std::shared_ptr<private_detail_te_handle_base_type> clone() const = 0;
        virtual const std::type_info& type() const                                = 0;

        virtual optional<std::vector<char>> load(const std::string& version,
                                                 const std::string& device,
                                                 const std::string& key_hash) = 0;
        virtual void store(const std::string& version,
                           const std::string& device,
                           const std::string& key_hash,
                           const binary_cache_entry& e,
                           const std::vector<char>& blob)                     = 0;
        virtual void begin_batch()                                            = 0;
        virtual void end_batch()                                              = 0;
    };

    template <typename PrivateDetailTypeErasedT>
    struct private_detail_te_handle_type : private_detail_te_handle_base_type
    {
        template <typename PrivateDetailTypeErasedU = PrivateDetailTypeErasedT>
        private_detail_te_handle_type(
            PrivateDetailTypeErasedT value,
            typename std::enable_if<std::is_reference<PrivateDetailTypeErasedU>{}>::type* = nullptr)
            : private_detail_te_value(value)
        {
        }

        template <typename PrivateDetailTypeErasedU = PrivateDetailTypeErasedT>
        private_detail_te_handle_type(
            PrivateDetailTypeErasedT value,
            typename std::enable_if<not std::is_reference<PrivateDetailTypeErasedU>{}, int>::type* =
                nullptr) noexcept
            : private_detail_te_value(std::move(value))
        {
        }

        std::shared_ptr<private_detail_te_handle_base_type> clone() const override
        {
            return std::make_shared<private_detail_te_handle_type>(private_detail_te_value);
        }

        const std::type_info& type() const override { return typeid(private_detail_te_value); }

        optional<std::vector<char>> load(const std::string& version,
                                         const std::string& device,
                                         const std::string& key_hash) override
        {

            return private_detail_te_value.load(version, device, key_hash);
        }

        void store(const std::string& version,
                   const std::string& device,
                   const std::string& key_hash,
                   const binary_cache_entry& e,
                   const std::vector<char>& blob) override
        {

            private_detail_te_value.store(version, device, key_hash, e, blob);
        }

        void begin_batch() override
        {

            private_detail_te_default_begin_batch(char(0), private_detail_te_value);
        }

        void end_batch() override
        {

            private_detail_te_default_end_batch(char(0), private_detail_te_value);
        }

        PrivateDetailTypeErasedT private_detail_te_value;
    };

    template <typename PrivateDetailTypeErasedT>
    struct private_detail_te_handle_type<std::reference_wrapper<PrivateDetailTypeErasedT>>
        : private_detail_te_handle_type<PrivateDetailTypeErasedT&>
    {
        private_detail_te_handle_type(std::reference_wrapper<PrivateDetailTypeErasedT> ref)
            : private_detail_te_handle_type<PrivateDetailTypeErasedT&>(ref.get())
        {
        }
    };

    bool private_detail_te_handle_empty() const
    {
        return private_detail_te_handle_mem_var == nullptr;
    }

    const private_detail_te_handle_base_type& private_detail_te_get_handle() const
    {
        assert(private_detail_te_handle_mem_var != nullptr);
        return *private_detail_te_handle_mem_var;
    }

    private_detail_te_handle_base_type& private_detail_te_get_handle()
    {
        assert(private_detail_te_handle_mem_var != nullptr);
        if(private_detail_te_handle_mem_var.use_count() > 1)
            private_detail_te_handle_mem_var = private_detail_te_handle_mem_var->clone();
        return *private_detail_te_handle_mem_var;
    }

    std::shared_ptr<private_detail_te_handle_base_type> private_detail_te_handle_mem_var;
};

template <typename ValueType>
inline const ValueType* any_cast(const binary_cache_backend* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType* any_cast(binary_cache_backend* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType& any_cast(binary_cache_backend& x)
{
    auto* y = x.any_cast<typename std::remove_reference<ValueType>::type>();
    if(y == nullptr)
        throw std::bad_cast();
    return *y;
}

template <typename ValueType>
inline const ValueType& any_cast(const binary_cache_backend& x)
{
    const auto* y = x.any_cast<typename std::remove_reference<ValueType>::type>();
    if(y == nullptr)
        throw std::bad_cast();
    return *y;
}
// NOLINTEND(performance-unnecessary-value-param)
#endif

#endif

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_BINARY_CACHE_BACKEND_HPP
