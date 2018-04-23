#ifndef GUARD_RTGLIB_OPERAND_HPP
#define RTG_GUARD_RTGLIB_OPERAND_HPP

#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <rtg/shape.hpp>
#include <rtg/argument.hpp>

namespace rtg {

/*
 * Type-erased interface for:
 *
 * struct operand
 * {
 *     std::string name() const;
 *     shape compute_shape(std::vector<shape> input) const;
 *     argument compute(std::vector<argument> input) const;
 * };
 *
 */

struct operand
{
    // Constructors
    operand() = default;

    template <typename PrivateDetailTypeErasedT>
    operand(PrivateDetailTypeErasedT value)
        : private_detail_te_handle_mem_var(
              std::make_shared<private_detail_te_handle_type<
                  typename std::remove_reference<PrivateDetailTypeErasedT>::type>>(
                  std::forward<PrivateDetailTypeErasedT>(value)))
    {
    }

    // Assignment
    template <typename PrivateDetailTypeErasedT>
    operand& operator=(PrivateDetailTypeErasedT value)
    {
        if(private_detail_te_handle_mem_var.unique())
            *private_detail_te_handle_mem_var = std::forward<PrivateDetailTypeErasedT>(value);
        else if(!private_detail_te_handle_mem_var)
            private_detail_te_handle_mem_var = std::make_shared<PrivateDetailTypeErasedT>(
                std::forward<PrivateDetailTypeErasedT>(value));
        return *this;
    }

    std::string name() const
    {
        assert(private_detail_te_handle_mem_var);
        return private_detail_te_get_handle().name();
    }

    shape compute_shape(std::vector<shape> input) const
    {
        assert(private_detail_te_handle_mem_var);
        return private_detail_te_get_handle().compute_shape(std::move(input));
    }

    argument compute(std::vector<argument> input) const
    {
        assert(private_detail_te_handle_mem_var);
        return private_detail_te_get_handle().compute(std::move(input));
    }

    private:
    struct private_detail_te_handle_base_type
    {
        virtual ~private_detail_te_handle_base_type() {}
        virtual std::shared_ptr<private_detail_te_handle_base_type> clone() const = 0;

        virtual std::string name() const                            = 0;
        virtual shape compute_shape(std::vector<shape> input) const = 0;
        virtual argument compute(std::vector<argument> input) const = 0;
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

        std::string name() const override { return private_detail_te_value.name(); }

        shape compute_shape(std::vector<shape> input) const override
        {
            return private_detail_te_value.compute_shape(std::move(input));
        }

        argument compute(std::vector<argument> input) const override
        {
            return private_detail_te_value.compute(std::move(input));
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

} // namespace rtg

#endif
