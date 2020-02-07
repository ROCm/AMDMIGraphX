#ifndef MIGRAPHX_GUARD_SCHEDULE_MODEL_HPP
#define MIGRAPHX_GUARD_SCHEDULE_MODEL_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <migraphx/config.hpp>
#include <migraphx/instruction_ref.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;
struct operation;

#ifdef DOXYGEN

/// An interface for target-dependent model for the scheduler
struct schedule_model
{
    /// Get the number of concurrent instruction allowed
    std::size_t concurrency() const;
    /// Schedule a concurrent instruction
    void sched(program& p, instruction_ref ins, std::size_t n) const;
    // Insert necessary waits before an instruction
    void wait(program& p, instruction_ref ins, std::size_t wait_id) const;
    // Insert necessary records after an instruction
    void record(program& p, instruction_ref ins, std::size_t wait_id) const;
    /// Compute weights for an operation
    std::size_t weight(const operation& op) const;
};

#else

/*
 * Type-erased interface for:
 *
 * struct schedule_model
 * {
 *      std::size_t concurrency() const;
 *      void sched(program& p,instruction_ref ins,std::size_t n) const;
 *      void wait(program& p,instruction_ref ins,std::size_t wait_id) const;
 *      void record(program& p,instruction_ref ins,std::size_t wait_id) const;
 *      std::size_t weight(const operation& op) const;
 * };
 *
 */

struct schedule_model
{
    // Constructors
    schedule_model() = default;

    template <typename PrivateDetailTypeErasedT>
    schedule_model(PrivateDetailTypeErasedT value)
        : private_detail_te_handle_mem_var(
              std::make_shared<private_detail_te_handle_type<
                  typename std::remove_reference<PrivateDetailTypeErasedT>::type>>(
                  std::forward<PrivateDetailTypeErasedT>(value)))
    {
    }

    // Assignment
    template <typename PrivateDetailTypeErasedT>
    schedule_model& operator=(PrivateDetailTypeErasedT value)
    {
        using std::swap;
        auto* derived = this->any_cast<PrivateDetailTypeErasedT>();
        if(derived and private_detail_te_handle_mem_var.unique())
        {
            *derived = std::forward<PrivateDetailTypeErasedT>(value);
        }
        else
        {
            schedule_model rhs(value);
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

    std::size_t concurrency() const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().concurrency();
    }

    void sched(program& p, instruction_ref ins, std::size_t n) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().sched(p, ins, n);
    }

    void wait(program& p, instruction_ref ins, std::size_t wait_id) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().wait(p, ins, wait_id);
    }

    void record(program& p, instruction_ref ins, std::size_t wait_id) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().record(p, ins, wait_id);
    }

    std::size_t weight(const operation& op) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().weight(op);
    }

    friend bool is_shared(const schedule_model& private_detail_x,
                          const schedule_model& private_detail_y)
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

        virtual std::size_t concurrency() const                                         = 0;
        virtual void sched(program& p, instruction_ref ins, std::size_t n) const        = 0;
        virtual void wait(program& p, instruction_ref ins, std::size_t wait_id) const   = 0;
        virtual void record(program& p, instruction_ref ins, std::size_t wait_id) const = 0;
        virtual std::size_t weight(const operation& op) const                           = 0;
    };

    template <typename PrivateDetailTypeErasedT>
    struct private_detail_te_handle_type : private_detail_te_handle_base_type
    {
        template <typename PrivateDetailTypeErasedU = PrivateDetailTypeErasedT>
        private_detail_te_handle_type(
            PrivateDetailTypeErasedT value,
            typename std::enable_if<std::is_reference<PrivateDetailTypeErasedU>::value>::type* =
                nullptr)
            : private_detail_te_value(value)
        {
        }

        template <typename PrivateDetailTypeErasedU = PrivateDetailTypeErasedT>
        private_detail_te_handle_type(
            PrivateDetailTypeErasedT value,
            typename std::enable_if<!std::is_reference<PrivateDetailTypeErasedU>::value,
                                    int>::type* = nullptr) noexcept
            : private_detail_te_value(std::move(value))
        {
        }

        std::shared_ptr<private_detail_te_handle_base_type> clone() const override
        {
            return std::make_shared<private_detail_te_handle_type>(private_detail_te_value);
        }

        const std::type_info& type() const override { return typeid(private_detail_te_value); }

        std::size_t concurrency() const override { return private_detail_te_value.concurrency(); }

        void sched(program& p, instruction_ref ins, std::size_t n) const override
        {

            private_detail_te_value.sched(p, ins, n);
        }

        void wait(program& p, instruction_ref ins, std::size_t wait_id) const override
        {

            private_detail_te_value.wait(p, ins, wait_id);
        }

        void record(program& p, instruction_ref ins, std::size_t wait_id) const override
        {

            private_detail_te_value.record(p, ins, wait_id);
        }

        std::size_t weight(const operation& op) const override
        {

            return private_detail_te_value.weight(op);
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
        if(!private_detail_te_handle_mem_var.unique())
            private_detail_te_handle_mem_var = private_detail_te_handle_mem_var->clone();
        return *private_detail_te_handle_mem_var;
    }

    std::shared_ptr<private_detail_te_handle_base_type> private_detail_te_handle_mem_var;
};

template <typename ValueType>
inline const ValueType* any_cast(const schedule_model* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType* any_cast(schedule_model* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType& any_cast(schedule_model& x)
{
    auto* y = x.any_cast<typename std::remove_reference<ValueType>::type>();
    if(y == nullptr)
        throw std::bad_cast();
    return *y;
}

template <typename ValueType>
inline const ValueType& any_cast(const schedule_model& x)
{
    const auto* y = x.any_cast<typename std::remove_reference<ValueType>::type>();
    if(y == nullptr)
        throw std::bad_cast();
    return *y;
}

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
