#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_TARGET_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_TARGET_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include <migraphx/context.hpp>
#include <migraphx/pass.hpp>
#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/rank.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#ifdef DOXYGEN

/// An interface for a compilation target
struct target
{
    /// A unique name used to identify the target
    std::string name() const;
    /**
     * @brief The transformation pass to be run during compilation.
     *
     * @param ctx This is the target-dependent context that is created by `get_context`
     * @return The passes to be ran
     */
    std::vector<pass> get_passes(context& ctx) const;
    /**
     * @brief Construct a context for the target.
     * @return The context to be used during compilation and execution.
     */
    context get_context() const;
    /**
     * @brief copy an argument to the current target.
     *
     * @param arg Input argument to be copied to the target
     * @return Argument in the target.
     */
    argument copy_to(const argument& arg) const;
    /**
     * @brief copy an argument from the current target.
     *
     * @param arg Input argument to be copied from the target
     * @return Argument in the host.
     */
    argument copy_from(const argument& arg) const;
    /**
     * @brief Allocate an argument based on the input shape
     *
     * @param s Shape of the argument to be allocated in the target
     * @return Allocated argument in the target.
     */
    argument allocate(const shape& s) const;
};

#else

template <class T>
auto target_allocate(rank<1>, T& x, const shape& s) -> decltype(x.allocate(s))
{
    return x.allocate(s);
}

template <class T>
argument target_allocate(rank<0>, T& x, const shape&)
{
    std::string name = x.name();
    MIGRAPHX_THROW("Not computable: " + name);
}

template <class T>
argument target_allocate(T& x, const shape& s)
{
    return target_allocate(rank<1>{}, x, s);
}

template <class T>
auto copy_to_target(rank<1>, T& x, const argument& arg) -> decltype(x.copy_to(arg))
{
    return x.copy_to(arg);
}

template <class T>
argument copy_to_target(rank<0>, T&, const argument& arg)
{
    return arg;
}

template <class T>
argument copy_to_target(T& x, const argument& arg)
{
    return copy_to_target(rank<1>{}, x, arg);
}

template <class T>
auto copy_from_target(rank<1>, T& x, const argument& arg) -> decltype(x.copy_from(arg))
{
    return x.copy_from(arg);
}

template <class T>
argument copy_from_target(rank<0>, T&, const argument& arg)
{
    return arg;
}

template <class T>
argument copy_from_target(T& x, const argument& arg)
{
    return copy_from_target(rank<1>{}, x, arg);
}

/*
 * Type-erased interface for:
 *
 * struct target
 * {
 *      std::string name() const;
 *      std::vector<pass> get_passes(context& ctx) const;
 *      context get_context() const;
 *      argument copy_to(const argument& input) const;
 *      argument copy_from(const argument& input) const;
 *      argument allocate(const shape& s) const;
 * };
 *
 */

struct target
{
    // Constructors
    target() = default;

    template <typename PrivateDetailTypeErasedT>
    target(PrivateDetailTypeErasedT value)
        : private_detail_te_handle_mem_var(
              std::make_shared<private_detail_te_handle_type<
                  typename std::remove_reference<PrivateDetailTypeErasedT>::type>>(
                  std::forward<PrivateDetailTypeErasedT>(value)))
    {
    }

    // Assignment
    template <typename PrivateDetailTypeErasedT>
    target& operator=(PrivateDetailTypeErasedT value)
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

    std::vector<pass> get_passes(context& ctx) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().get_passes(ctx);
    }

    context get_context() const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().get_context();
    }

    argument copy_to(const argument& input) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().copy_to(input);
    }

    argument copy_from(const argument& input) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().copy_from(input);
    }

    argument allocate(const shape& s) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().allocate(s);
    }

    friend bool is_shared(const target& private_detail_x, const target& private_detail_y)
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
        virtual std::vector<pass> get_passes(context& ctx) const = 0;
        virtual context get_context() const                      = 0;
        virtual argument copy_to(const argument& input) const    = 0;
        virtual argument copy_from(const argument& input) const  = 0;
        virtual argument allocate(const shape& s) const          = 0;
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

        std::vector<pass> get_passes(context& ctx) const override
        {

            return private_detail_te_value.get_passes(ctx);
        }

        context get_context() const override { return private_detail_te_value.get_context(); }

        argument copy_to(const argument& input) const override
        {

            return copy_to_target(private_detail_te_value, input);
        }

        argument copy_from(const argument& input) const override
        {

            return copy_from_target(private_detail_te_value, input);
        }

        argument allocate(const shape& s) const override
        {

            return target_allocate(private_detail_te_value, s);
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
inline const ValueType* any_cast(const target* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType* any_cast(target* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType& any_cast(target& x)
{
    auto* y = x.any_cast<typename std::remove_reference<ValueType>::type>();
    if(y == nullptr)
        throw std::bad_cast();
    return *y;
}

template <typename ValueType>
inline const ValueType& any_cast(const target& x)
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
