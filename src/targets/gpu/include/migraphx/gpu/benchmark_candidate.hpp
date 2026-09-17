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
// te.py DSL for migraphx::gpu::benchmark_candidate.
//
// The generated header lives at
// src/targets/gpu/include/migraphx/gpu/benchmark_candidate.hpp; regenerate it
// with `cd tools && python generate.py` (generate_all routes include/gpu/ inputs
// into the gpu target tree). Do not edit the generated header by hand.
//
#ifndef MIGRAPHX_GUARD_GPU_BENCHMARK_CANDIDATE_HPP
#define MIGRAPHX_GUARD_GPU_BENCHMARK_CANDIDATE_HPP

#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/program.hpp>
#include <migraphx/tracer.hpp>
#include <migraphx/value.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/export.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

#ifdef DOXYGEN

/// Type-erased interface for a tuning candidate that can be timed by a
/// benchmarker (see simple_benchmark and adaptive_topk_benchmark in
/// <migraphx/gpu/time_op.hpp>). A candidate knows how to build a runnable
/// program for itself and how to generate the input data used to run it.
struct benchmark_candidate
{
    /// Generate one input argument per parameter of the program returned by
    /// make_program(), ordered to match its parameter order.
    std::vector<argument> generate_arguments(const context& ictx) const;

    /// Build a runnable program for this candidate.
    program make_program() const;

    /// Tracer used to report benchmark progress; return a disabled tracer to
    /// silence the output.
    tracer trace() const;

    /// The tuning solution this candidate was compiled with.
    value solution() const;

    /// Called with the program before it is timed; used to print the program
    /// and other info on higher trace levels.
    void before_run(const program& p) const;
};

#else

#ifdef TYPE_ERASED_DECLARATION

// Type-erased interface for:
struct MIGRAPHX_EXPORT benchmark_candidate
{
    //
    std::vector<argument> generate_arguments(const context& ictx) const;
    //
    program make_program() const;
    //
    tracer trace() const;
    //
    value solution() const;
    //
    void before_run(const program& p) const;
};

#else
// NOLINTBEGIN(performance-unnecessary-value-param)
struct benchmark_candidate
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
        decltype(std::declval<PrivateDetailTypeErasedT>().generate_arguments(
                     std::declval<const context&>()),
                 std::declval<PrivateDetailTypeErasedT>().make_program(),
                 std::declval<PrivateDetailTypeErasedT>().trace(),
                 std::declval<PrivateDetailTypeErasedT>().solution(),
                 std::declval<PrivateDetailTypeErasedT>().before_run(
                     std::declval<const program&>()),
                 void());

    template <class PrivateDetailTypeErasedT>
    using private_te_constraints = private_te_constraints_impl<
        typename private_te_unwrap_reference<private_te_pure<PrivateDetailTypeErasedT>>::type>;

    public:
    // Constructors
    benchmark_candidate() = default;

    template <typename PrivateDetailTypeErasedT,
              typename = private_te_constraints<PrivateDetailTypeErasedT>,
              typename = typename std::enable_if<
                  not std::is_same<private_te_pure<PrivateDetailTypeErasedT>,
                                   benchmark_candidate>{}>::type>
    benchmark_candidate(PrivateDetailTypeErasedT&& value)
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
                                   benchmark_candidate>{}>::type>
    benchmark_candidate& operator=(PrivateDetailTypeErasedT && value)
    {
        using std::swap;
        auto* derived = this->any_cast<private_te_pure<PrivateDetailTypeErasedT>>();
        if(derived and private_detail_te_handle_mem_var.use_count() == 1)
        {
            *derived = std::forward<PrivateDetailTypeErasedT>(value);
        }
        else
        {
            benchmark_candidate rhs(value);
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

    std::vector<argument> generate_arguments(const context& ictx) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().generate_arguments(ictx);
    }

    program make_program() const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().make_program();
    }

    tracer trace() const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().trace();
    }

    value solution() const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().solution();
    }

    void before_run(const program& p) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().before_run(p);
    }

    friend bool is_shared(const benchmark_candidate& private_detail_x,
                          const benchmark_candidate& private_detail_y)
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

        virtual std::vector<argument> generate_arguments(const context& ictx) const = 0;
        virtual program make_program() const                                        = 0;
        virtual tracer trace() const                                                = 0;
        virtual value solution() const                                              = 0;
        virtual void before_run(const program& p) const                             = 0;
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

        std::vector<argument> generate_arguments(const context& ictx) const override
        {

            return private_detail_te_value.generate_arguments(ictx);
        }

        program make_program() const override { return private_detail_te_value.make_program(); }

        tracer trace() const override { return private_detail_te_value.trace(); }

        value solution() const override { return private_detail_te_value.solution(); }

        void before_run(const program& p) const override { private_detail_te_value.before_run(p); }

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
inline const ValueType* any_cast(const benchmark_candidate* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType* any_cast(benchmark_candidate* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType& any_cast(benchmark_candidate& x)
{
    auto* y = x.any_cast<typename std::remove_reference<ValueType>::type>();
    if(y == nullptr)
        throw std::bad_cast();
    return *y;
}

template <typename ValueType>
inline const ValueType& any_cast(const benchmark_candidate& x)
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

#endif // MIGRAPHX_GUARD_GPU_BENCHMARK_CANDIDATE_HPP
