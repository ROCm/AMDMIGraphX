#ifndef RTG_GUARD_RTGLIB_OPERAND_HPP
#define RTG_GUARD_RTGLIB_OPERAND_HPP

#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <rtg/shape.hpp>
#include <rtg/argument.hpp>
#include <rtg/context.hpp>

namespace rtg {

namespace operation_stream {

template <class T>
auto operator<<(std::ostream& os, const T& x) -> decltype(os << x.name())
{
    return os << x.name();
}

} // namespace operation_stream

/*
 * Type-erased interface for:
 *
 * struct operation
 * {
 *      std::string name() const;
 *      shape compute_shape(std::vector<shape> input) const;
 *      argument compute(context& ctx,shape output,std::vector<argument> input) const;
 *     friend std::ostream & operator<<(std::ostream & os,const operation & op) ;
 * };
 *
 */

struct operation
{
    // Constructors
    operation() = default;

    template <typename PrivateDetailTypeErasedT>
    operation(PrivateDetailTypeErasedT value)
        : private_detail_te_handle_mem_var(
              std::make_shared<private_detail_te_handle_type<
                  typename std::remove_reference<PrivateDetailTypeErasedT>::type>>(
                  std::forward<PrivateDetailTypeErasedT>(value)))
    {
    }

    // Assignment
    template <typename PrivateDetailTypeErasedT>
    operation& operator=(PrivateDetailTypeErasedT value)
    {
        if(private_detail_te_handle_mem_var.unique())
            *private_detail_te_handle_mem_var = std::forward<PrivateDetailTypeErasedT>(value);
        else if(!private_detail_te_handle_mem_var)
            private_detail_te_handle_mem_var = std::make_shared<PrivateDetailTypeErasedT>(
                std::forward<PrivateDetailTypeErasedT>(value));
        return *this;
    }

    // Cast
    template <typename PrivateDetailTypeErasedT>
    PrivateDetailTypeErasedT* any_cast()
    {
        return private_detail_te_get_handle().type() == typeid(PrivateDetailTypeErasedT)
                   ? std::addressof(static_cast<private_detail_te_handle_type<
                                        typename std::remove_cv<PrivateDetailTypeErasedT>::type>&>(
                                        private_detail_te_get_handle())
                                        .private_detail_te_value)
                   : nullptr;
    }

    template <typename PrivateDetailTypeErasedT>
    const typename std::remove_cv<PrivateDetailTypeErasedT>::type* any_cast() const
    {
        return private_detail_te_get_handle().type() == typeid(PrivateDetailTypeErasedT)
                   ? std::addressof(static_cast<const private_detail_te_handle_type<
                                        typename std::remove_cv<PrivateDetailTypeErasedT>::type>&>(
                                        private_detail_te_get_handle())
                                        .private_detail_te_value)
                   : nullptr;
    }

    std::string name() const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().name();
    }

    shape compute_shape(std::vector<shape> input) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().compute_shape(std::move(input));
    }

    argument compute(context& ctx, shape output, std::vector<argument> input) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().compute(
            ctx, std::move(output), std::move(input));
    }

    friend std::ostream& operator<<(std::ostream& os, const operation& op)
    {
        assert(op.private_detail_te_handle_mem_var);
        return op.private_detail_te_get_handle().operator_shift_left(os);
    }

    private:
    struct private_detail_te_handle_base_type
    {
        virtual ~private_detail_te_handle_base_type() {}
        virtual std::shared_ptr<private_detail_te_handle_base_type> clone() const = 0;
        virtual const std::type_info& type() const                                = 0;

        virtual std::string name() const                                                        = 0;
        virtual shape compute_shape(std::vector<shape> input) const                             = 0;
        virtual argument compute(context& ctx, shape output, std::vector<argument> input) const = 0;
        virtual std::ostream& operator_shift_left(std::ostream& os) const                       = 0;
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

        shape compute_shape(std::vector<shape> input) const override
        {

            return private_detail_te_value.compute_shape(std::move(input));
        }

        argument compute(context& ctx, shape output, std::vector<argument> input) const override
        {

            return private_detail_te_value.compute(ctx, std::move(output), std::move(input));
        }

        std::ostream& operator_shift_left(std::ostream& os) const override
        {
            using rtg::operation_stream::operator<<;
            return os << private_detail_te_value;
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

    const private_detail_te_handle_base_type& private_detail_te_get_handle() const
    {
        return *private_detail_te_handle_mem_var;
    }

    private_detail_te_handle_base_type& private_detail_te_get_handle()
    {
        if(!private_detail_te_handle_mem_var.unique())
            private_detail_te_handle_mem_var = private_detail_te_handle_mem_var->clone();
        return *private_detail_te_handle_mem_var;
    }

    std::shared_ptr<private_detail_te_handle_base_type> private_detail_te_handle_mem_var;
};

template <typename ValueType>
inline const ValueType* any_cast(const operation* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType* any_cast(operation* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType& any_cast(operation& x)
{
    auto* y = x.any_cast<typename std::remove_reference<ValueType>::type>();
    if(y == nullptr)
        throw std::bad_cast();
    return *y;
}

template <typename ValueType>
inline const ValueType& any_cast(const operation& x)
{
    const auto* y = x.any_cast<typename std::remove_reference<ValueType>::type>();
    if(y == nullptr)
        throw std::bad_cast();
    return *y;
}

} // namespace rtg

#endif
