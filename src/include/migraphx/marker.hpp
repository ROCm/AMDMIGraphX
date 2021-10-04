#ifndef MIGRAPHX_GUARD_MARKER_HPP
#define MIGRAPHX_GUARD_MARKER_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#ifdef DOXYGEN

/// Marker is an interface to general marking functions, such as rocTX markers.

#else

/*
 * Type-erased interface for:
 *
 * struct marker
 * {
 *      std::size_t mark_range_start(std::size_t range_id) ;
 *      void mark_ins_start(std::string log) ;
 *      void mark_program_start() ;
 *      void mark_range_finish(std::size_t range_id) ;
 *      void mark_ins_finish() ;
 *      void mark_program_finish() ;
 * };
 *
 */

struct marker
{
    // Constructors
    marker() = default;

    template <typename PrivateDetailTypeErasedT>
    marker(PrivateDetailTypeErasedT value)
        : private_detail_te_handle_mem_var(
              std::make_shared<private_detail_te_handle_type<
                  typename std::remove_reference<PrivateDetailTypeErasedT>::type>>(
                  std::forward<PrivateDetailTypeErasedT>(value)))
    {
    }

    // Assignment
    template <typename PrivateDetailTypeErasedT>
    marker& operator=(PrivateDetailTypeErasedT value)
    {
        using std::swap;
        auto* derived = this->any_cast<PrivateDetailTypeErasedT>();
        if(derived and private_detail_te_handle_mem_var.unique())
        {
            *derived = std::forward<PrivateDetailTypeErasedT>(value);
        }
        else
        {
            marker rhs(value);
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

    std::size_t mark_range_start(std::size_t range_id)
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().mark_range_start(range_id);
    }

    void mark_ins_start(std::string log)
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().mark_ins_start(std::move(log));
    }

    void mark_program_start()
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().mark_program_start();
    }

    void mark_range_finish(std::size_t range_id)
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().mark_range_finish(range_id);
    }

    void mark_ins_finish()
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().mark_ins_finish();
    }

    void mark_program_finish()
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().mark_program_finish();
    }

    friend bool is_shared(const marker& private_detail_x, const marker& private_detail_y)
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

        virtual std::size_t mark_range_start(std::size_t range_id) = 0;
        virtual void mark_ins_start(std::string log)               = 0;
        virtual void mark_program_start()                          = 0;
        virtual void mark_range_finish(std::size_t range_id)       = 0;
        virtual void mark_ins_finish()                             = 0;
        virtual void mark_program_finish()                         = 0;
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

        std::size_t mark_range_start(std::size_t range_id) override
        {

            return private_detail_te_value.mark_range_start(range_id);
        }

        void mark_ins_start(std::string log) override
        {

            private_detail_te_value.mark_ins_start(std::move(log));
        }

        void mark_program_start() override { private_detail_te_value.mark_program_start(); }

        void mark_range_finish(std::size_t range_id) override
        {

            private_detail_te_value.mark_range_finish(range_id);
        }

        void mark_ins_finish() override { private_detail_te_value.mark_ins_finish(); }

        void mark_program_finish() override { private_detail_te_value.mark_program_finish(); }

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
inline const ValueType* any_cast(const marker* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType* any_cast(marker* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType& any_cast(marker& x)
{
    auto* y = x.any_cast<typename std::remove_reference<ValueType>::type>();
    if(y == nullptr)
        throw std::bad_cast();
    return *y;
}

template <typename ValueType>
inline const ValueType& any_cast(const marker& x)
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
