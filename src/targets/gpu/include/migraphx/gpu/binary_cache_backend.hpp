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
// member functions listed below. The wrapper holds T by shared_ptr and forwards
// each call through a virtual dispatch, matching problem_cache_backend.
//
// Notes:
//   * binary_cache_entry is defined in <migraphx/gpu/binary_cache_entry.hpp>;
//     the include below pulls in its full definition.
//   * Backends must be copyable: the wrapper shares T and clones it on a
//     non-const call while the handle is shared. sqlite_binary_cache shares its
//     connection across copies.
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
/// A backend persists binary_cache_entry values to some medium (a directory of
/// files or a SQLite database), and decides for itself how to serialize them.
/// Entries are addressed by their key, scoped by two strings the caller has
/// already computed:
///
///   * `version` -- binary_cache::version_id(), identifying the toolchain and
///     the embedded kernel sources that produced the entry. Never empty; the
///     caller skips persistence entirely when it is.
///   * `device`  -- the GPU the entry was compiled for.
///
/// A backend must keep entries with different scopes distinct rather than
/// overwriting across them. It may address entries by a hash of the key, for
/// instance to keep file names short, but must then check the full key when
/// loading so that a collision is a miss rather than a wrong kernel.
struct binary_cache_backend
{
    /// Return the entry stored for this key, or nullopt for a miss.
    ///
    /// nullopt also covers every failure: a missing file, an unreadable
    /// database, a damaged entry, a permissions problem. A cache that cannot be
    /// read is not an error, it is a cache miss, and the caller recompiles.
    ///
    /// Must not throw.
    optional<binary_cache_entry>
    load(const std::string& version, const std::string& device, const std::string& key);

    /// Persist every entry in `entries` under its key. The entries arrive
    /// together so a backend can commit them at once, such as in one database
    /// transaction, rather than one at a time.
    ///
    /// Overwriting an existing entry is expected and safe: the content is
    /// decided entirely by the key, so a writer that loses a race replaces the
    /// entry with an equivalent one.
    ///
    /// May throw: the caller reports a failed store as a warning. It costs a
    /// recompile next run, nothing more, and the caller still keeps the results
    /// in memory. A backend that throws must not leave anything locked.
    void store(const std::string& version,
               const std::string& device,
               const std::vector<binary_cache_entry>& entries);
};

#else

#ifdef TYPE_ERASED_DECLARATION

// Type-erased interface for:
struct MIGRAPHX_EXPORT binary_cache_backend
{
    //
    optional<binary_cache_entry>
    load(const std::string& version, const std::string& device, const std::string& key);
    //
    void store(const std::string& version,
               const std::string& device,
               const std::vector<binary_cache_entry>& entries);
};

#else
// NOLINTBEGIN(performance-unnecessary-value-param)
struct binary_cache_backend
{
    private:
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
                     std::declval<const std::vector<binary_cache_entry>&>()),
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

    optional<binary_cache_entry>
    load(const std::string& version, const std::string& device, const std::string& key)
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().load(version, device, key);
    }

    void store(const std::string& version,
               const std::string& device,
               const std::vector<binary_cache_entry>& entries)
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().store(version, device, entries);
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

        virtual optional<binary_cache_entry>
        load(const std::string& version, const std::string& device, const std::string& key) = 0;
        virtual void store(const std::string& version,
                           const std::string& device,
                           const std::vector<binary_cache_entry>& entries)                  = 0;
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

        optional<binary_cache_entry>
        load(const std::string& version, const std::string& device, const std::string& key) override
        {

            return private_detail_te_value.load(version, device, key);
        }

        void store(const std::string& version,
                   const std::string& device,
                   const std::vector<binary_cache_entry>& entries) override
        {

            private_detail_te_value.store(version, device, entries);
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
