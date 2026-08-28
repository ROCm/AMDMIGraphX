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
// te.py DSL for migraphx::gpu::problem_cache_backend.
//
// The generated header lives at
// src/targets/gpu/include/migraphx/gpu/problem_cache_backend.hpp; regenerate it
// with `cd tools && python generate.py` (generate_all routes include/gpu/ inputs
// into the gpu target tree). Do not edit the generated header by hand.
//
// Any type T satisfies the problem_cache_backend concept if it provides the
// member functions listed below. The wrapper holds T by shared_ptr (small
// object optimisation is irrelevant -- backends typically own non-trivial
// resources like file handles or SQLite connections) and forwards each call
// through a virtual dispatch.
//
// Notes:
//   * cache_device_key is defined in <migraphx/gpu/cache_device_key.hpp>;
//     the include below pulls in its full definition.
//   * Backends are NOT expected to be copyable in any meaningful sense
//     beyond what shared ownership provides; they may freely hold streams,
//     file handles, etc.
//   * load() is non-const (mutates internal state); save() is const.
//
#ifndef MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_BACKEND_HPP
#define MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_BACKEND_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/gpu/export.h>
#include <migraphx/gpu/cache_device_key.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

#ifdef DOXYGEN

/// Type-erased interface for problem-cache storage backends.
///
/// A backend persists (cache_device_key, key) -> solution entries to some
/// medium (JSON file, SQLite database, in-memory map for tests, ...). All
/// lookups and inserts are scoped by the device key so a single backend
/// instance can hold entries for multiple GPU configurations.
struct problem_cache_backend
{
    /// Load entries from `path` into the backend. Idempotent; calling
    /// load() twice with the same path leaves the backend in an equivalent
    /// state. Implementations should treat a missing file as "no entries"
    /// rather than an error (the typical first-run case).
    void load(const std::string& path);

    /// Persist all entries to `path`. Const by design: backends that need
    /// to flush write-behind buffers must do so internally without mutating
    /// observable state.
    void save(const std::string& path) const;

    /// Insert (or overwrite) the solution for `key` in the bucket
    /// identified by `dk`. Precondition: `solution` is not null.
    void insert(const cache_device_key& dk, const value& key, const value& solution);

    /// Insert a pending sentinel (null value) for `key`, recording that a
    /// solution for this problem is being tuned. A later lookup that sees the
    /// sentinel treats the problem as already in progress rather than as a
    /// real cached solution.
    void mark(const cache_device_key& dk, const value& key);

    /// Return the stored solution for `key` in the bucket `dk`, or nullopt
    /// if no entry exists. A present optional with a null value indicates
    /// a sentinel (set via mark()).
    optional<value> get(const cache_device_key& dk, const value& key) const;

    /// True iff any entry (real solution or sentinel) exists for `key` in
    /// the bucket `dk`.
    bool has(const cache_device_key& dk, const value& key) const;
};

#else

#ifdef TYPE_ERASED_DECLARATION

// Type-erased interface for:
struct MIGRAPHX_EXPORT problem_cache_backend
{
    //
    void load(const std::string& path);
    //
    void save(const std::string& path) const;
    //
    void insert(const cache_device_key& dk, const value& key, const value& solution);
    //
    void mark(const cache_device_key& dk, const value& key);
    //
    optional<value> get(const cache_device_key& dk, const value& key) const;
    //
    bool has(const cache_device_key& dk, const value& key) const;
};

#else
// NOLINTBEGIN(performance-unnecessary-value-param)
struct problem_cache_backend
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
        decltype(std::declval<PrivateDetailTypeErasedT>().load(std::declval<const std::string&>()),
                 std::declval<PrivateDetailTypeErasedT>().save(std::declval<const std::string&>()),
                 std::declval<PrivateDetailTypeErasedT>().insert(
                     std::declval<const cache_device_key&>(),
                     std::declval<const value&>(),
                     std::declval<const value&>()),
                 std::declval<PrivateDetailTypeErasedT>().mark(
                     std::declval<const cache_device_key&>(), std::declval<const value&>()),
                 std::declval<PrivateDetailTypeErasedT>().get(
                     std::declval<const cache_device_key&>(), std::declval<const value&>()),
                 std::declval<PrivateDetailTypeErasedT>().has(
                     std::declval<const cache_device_key&>(), std::declval<const value&>()),
                 void());

    template <class PrivateDetailTypeErasedT>
    using private_te_constraints = private_te_constraints_impl<
        typename private_te_unwrap_reference<private_te_pure<PrivateDetailTypeErasedT>>::type>;

    public:
    // Constructors
    problem_cache_backend() = default;

    template <typename PrivateDetailTypeErasedT,
              typename = private_te_constraints<PrivateDetailTypeErasedT>,
              typename = typename std::enable_if<
                  not std::is_same<private_te_pure<PrivateDetailTypeErasedT>,
                                   problem_cache_backend>{}>::type>
    problem_cache_backend(PrivateDetailTypeErasedT&& value)
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
                                   problem_cache_backend>{}>::type>
    problem_cache_backend& operator=(PrivateDetailTypeErasedT && value)
    {
        using std::swap;
        auto* derived = this->any_cast<private_te_pure<PrivateDetailTypeErasedT>>();
        if(derived and private_detail_te_handle_mem_var.use_count() == 1)
        {
            *derived = std::forward<PrivateDetailTypeErasedT>(value);
        }
        else
        {
            problem_cache_backend rhs(value);
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

    void load(const std::string& path)
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().load(path);
    }

    void save(const std::string& path) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().save(path);
    }

    void insert(const cache_device_key& dk, const value& key, const value& solution)
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().insert(dk, key, solution);
    }

    void mark(const cache_device_key& dk, const value& key)
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().mark(dk, key);
    }

    optional<value> get(const cache_device_key& dk, const value& key) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().get(dk, key);
    }

    bool has(const cache_device_key& dk, const value& key) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().has(dk, key);
    }

    friend bool is_shared(const problem_cache_backend& private_detail_x,
                          const problem_cache_backend& private_detail_y)
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

        virtual void load(const std::string& path)       = 0;
        virtual void save(const std::string& path) const = 0;
        virtual void
        insert(const cache_device_key& dk, const value& key, const value& solution)     = 0;
        virtual void mark(const cache_device_key& dk, const value& key)                 = 0;
        virtual optional<value> get(const cache_device_key& dk, const value& key) const = 0;
        virtual bool has(const cache_device_key& dk, const value& key) const            = 0;
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

        void load(const std::string& path) override { private_detail_te_value.load(path); }

        void save(const std::string& path) const override { private_detail_te_value.save(path); }

        void insert(const cache_device_key& dk, const value& key, const value& solution) override
        {

            private_detail_te_value.insert(dk, key, solution);
        }

        void mark(const cache_device_key& dk, const value& key) override
        {

            private_detail_te_value.mark(dk, key);
        }

        optional<value> get(const cache_device_key& dk, const value& key) const override
        {

            return private_detail_te_value.get(dk, key);
        }

        bool has(const cache_device_key& dk, const value& key) const override
        {

            return private_detail_te_value.has(dk, key);
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
inline const ValueType* any_cast(const problem_cache_backend* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType* any_cast(problem_cache_backend* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType& any_cast(problem_cache_backend& x)
{
    auto* y = x.any_cast<typename std::remove_reference<ValueType>::type>();
    if(y == nullptr)
        throw std::bad_cast();
    return *y;
}

template <typename ValueType>
inline const ValueType& any_cast(const problem_cache_backend& x)
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

#endif // MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_BACKEND_HPP
