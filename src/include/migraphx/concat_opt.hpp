#ifndef MIGRAPHX_GUARD_CONCAT_OPT_HPP
#define MIGRAPHX_GUARD_CONCAT_OPT_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <migraphx/operation.hpp>
#include <migraphx/op/concat.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

#ifdef DOXYGEN

/// An interface for target-dependent optimization for the concat instruction
struct concat_optimization
{
    /// The name of the target-dependent concat operator
    std::string name() const;
    /// A name of the target-dependent allocate operator
    std::string allocate() const;
    /// Return the target-independent concat operator
    op::concat get_concat(const operation& op) const;
};

#else

/*
 * Type-erased interface for:
 *
 * struct concat_optimization
 * {
 *      std::string name() const;
 *      std::string allocate() const;
 *      op::concat get_concat(const operation& op) const;
 * };
 *
 */

struct concat_optimization
{
    // Constructors
    concat_optimization() = default;

    template <typename PrivateDetailTypeErasedT>
    concat_optimization(PrivateDetailTypeErasedT value)
        : private_detail_te_handle_mem_var(
              std::make_shared<private_detail_te_handle_type<
                  typename std::remove_reference<PrivateDetailTypeErasedT>::type>>(
                  std::forward<PrivateDetailTypeErasedT>(value)))
    {
    }

    // Assignment
    template <typename PrivateDetailTypeErasedT>
    concat_optimization& operator=(PrivateDetailTypeErasedT value)
    {
        using std::swap;
        auto* derived = this->any_cast<PrivateDetailTypeErasedT>();
        if(derived and private_detail_te_handle_mem_var.unique())
        {
            *derived = std::forward<PrivateDetailTypeErasedT>(value);
        }
        else
        {
            concat_optimization rhs(value);
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

    std::string name() const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().name();
    }

    std::string allocate() const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().allocate();
    }

    op::concat get_concat(const operation& op) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().get_concat(op);
    }

    friend bool is_shared(const concat_optimization& private_detail_x,
                          const concat_optimization& private_detail_y)
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

        virtual std::string name() const                         = 0;
        virtual std::string allocate() const                     = 0;
        virtual op::concat get_concat(const operation& op) const = 0;
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

        std::string name() const override { return private_detail_te_value.name(); }

        std::string allocate() const override { return private_detail_te_value.allocate(); }

        op::concat get_concat(const operation& op) const override
        {

            return private_detail_te_value.get_concat(op);
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
inline const ValueType* any_cast(const concat_optimization* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType* any_cast(concat_optimization* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType& any_cast(concat_optimization& x)
{
    auto* y = x.any_cast<typename std::remove_reference<ValueType>::type>();
    if(y == nullptr)
        throw std::bad_cast();
    return *y;
}

template <typename ValueType>
inline const ValueType& any_cast(const concat_optimization& x)
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
