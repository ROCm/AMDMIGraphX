#ifndef GUARD_RTGLIB_OPERAND_HPP
#define GUARD_RTGLIB_OPERAND_HPP

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

    template <typename TypeErased_T_>
    operand(TypeErased_T_ value)
        : handle_mem_var_(
              std::make_shared<handle_type_<typename std::remove_reference<TypeErased_T_>::type>>(
                  std::forward<TypeErased_T_>(value)))
    {
    }

    // Assignment
    template <typename TypeErased_T_>
    operand& operator=(TypeErased_T_ value)
    {
        if(handle_mem_var_.unique())
            *handle_mem_var_ = std::forward<TypeErased_T_>(value);
        else if(!handle_mem_var_)
            handle_mem_var_ = std::make_shared<TypeErased_T_>(std::forward<TypeErased_T_>(value));
        return *this;
    }

    std::string name() const
    {
        assert(handle_mem_var_);
        return get_handle_().name();
    }

    shape compute_shape(std::vector<shape> input) const
    {
        assert(handle_mem_var_);
        return get_handle_().compute_shape(std::move(input));
    }

    argument compute(std::vector<argument> input) const
    {
        assert(handle_mem_var_);
        return get_handle_().compute(std::move(input));
    }

    private:
    struct handle_base_type_
    {
        virtual ~handle_base_type_() {}
        virtual std::shared_ptr<handle_base_type_> clone() const = 0;

        virtual std::string name() const                            = 0;
        virtual shape compute_shape(std::vector<shape> input) const = 0;
        virtual argument compute(std::vector<argument> input) const = 0;
    };

    template <typename TypeErased_T_>
    struct handle_type_ : handle_base_type_
    {
        template <typename TypeErased_U_ = TypeErased_T_>
        handle_type_(TypeErased_T_ value,
                     typename std::enable_if<std::is_reference<TypeErased_U_>::value>::type* = 0)
            : value_(value)
        {
        }

        template <typename TypeErased_U_ = TypeErased_T_>
        handle_type_(TypeErased_T_ value,
                     typename std::enable_if<!std::is_reference<TypeErased_U_>::value, int>::type* =
                         0) noexcept : value_(std::move(value))
        {
        }

        virtual std::shared_ptr<handle_base_type_> clone() const
        {
            return std::make_shared<handle_type_>(value_);
        }

        virtual std::string name() const { return value_.name(); }

        virtual shape compute_shape(std::vector<shape> input) const
        {
            return value_.compute_shape(std::move(input));
        }

        virtual argument compute(std::vector<argument> input) const
        {
            return value_.compute(std::move(input));
        }

        TypeErased_T_ value_;
    };

    template <typename TypeErased_T_>
    struct handle_type_<std::reference_wrapper<TypeErased_T_>> : handle_type_<TypeErased_T_&>
    {
        handle_type_(std::reference_wrapper<TypeErased_T_> ref)
            : handle_type_<TypeErased_T_&>(ref.get())
        {
        }
    };

    const handle_base_type_& get_handle_() const { return *handle_mem_var_; }

    handle_base_type_& get_handle_()
    {
        if(!handle_mem_var_.unique())
            handle_mem_var_ = handle_mem_var_->clone();
        return *handle_mem_var_;
    }

    std::shared_ptr<handle_base_type_> handle_mem_var_;
};

}

#endif
