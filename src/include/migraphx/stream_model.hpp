#ifndef MIGRAPHX_GUARD_STREAM_MODEL_HPP
#define MIGRAPHX_GUARD_STREAM_MODEL_HPP

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

#ifdef DOXYGEN

/// An interface for target-dependent model for the scheduler
struct stream_model
{
    /// Get the number of streams used in the program
    std::size_t get_nstream() const;
    /// Get stream for instruction
    std::size_t get_stream(instruction_ref ins) const;
    /// Get unique event id for instruction
    std::size_t get_event_id(instruction_ref ins) const;
    /// Returns true if instruction has a stream assignment
    bool has_stream(instruction_ref ins) const;
    /// Returns true if the instruction records the event
    bool is_record(instruction_ref ins) const;
    /// Returns true if the instruction wait on the event
    bool is_wait(instruction_ref ins) const;
};

#else

#ifdef TYPE_ERASED_DECLARATION

// Type-erased interface for:
struct stream_model
{
    //
    std::size_t get_nstream() const;
    //
    std::size_t get_stream(instruction_ref ins) const;
    //
    std::size_t get_event_id(instruction_ref ins) const;
    //
    bool has_stream(instruction_ref ins) const;
    //
    bool is_record(instruction_ref ins) const;
    //
    bool is_wait(instruction_ref ins) const;
};

#else

struct stream_model
{
    // Constructors
    stream_model() = default;

    template <typename PrivateDetailTypeErasedT>
    stream_model(PrivateDetailTypeErasedT value)
        : private_detail_te_handle_mem_var(
              std::make_shared<private_detail_te_handle_type<
                  typename std::remove_reference<PrivateDetailTypeErasedT>::type>>(
                  std::forward<PrivateDetailTypeErasedT>(value)))
    {
    }

    // Assignment
    template <typename PrivateDetailTypeErasedT>
    stream_model& operator=(PrivateDetailTypeErasedT value)
    {
        using std::swap;
        auto* derived = this->any_cast<PrivateDetailTypeErasedT>();
        if(derived and private_detail_te_handle_mem_var.unique())
        {
            *derived = std::forward<PrivateDetailTypeErasedT>(value);
        }
        else
        {
            stream_model rhs(value);
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

    std::size_t get_nstream() const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().get_nstream();
    }

    std::size_t get_stream(instruction_ref ins) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().get_stream(ins);
    }

    std::size_t get_event_id(instruction_ref ins) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().get_event_id(ins);
    }

    bool has_stream(instruction_ref ins) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().has_stream(ins);
    }

    bool is_record(instruction_ref ins) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().is_record(ins);
    }

    bool is_wait(instruction_ref ins) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().is_wait(ins);
    }

    friend bool is_shared(const stream_model& private_detail_x,
                          const stream_model& private_detail_y)
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

        virtual std::size_t get_nstream() const                     = 0;
        virtual std::size_t get_stream(instruction_ref ins) const   = 0;
        virtual std::size_t get_event_id(instruction_ref ins) const = 0;
        virtual bool has_stream(instruction_ref ins) const          = 0;
        virtual bool is_record(instruction_ref ins) const           = 0;
        virtual bool is_wait(instruction_ref ins) const             = 0;
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

        std::size_t get_nstream() const override { return private_detail_te_value.get_nstream(); }

        std::size_t get_stream(instruction_ref ins) const override
        {

            return private_detail_te_value.get_stream(ins);
        }

        std::size_t get_event_id(instruction_ref ins) const override
        {

            return private_detail_te_value.get_event_id(ins);
        }

        bool has_stream(instruction_ref ins) const override
        {

            return private_detail_te_value.has_stream(ins);
        }

        bool is_record(instruction_ref ins) const override
        {

            return private_detail_te_value.is_record(ins);
        }

        bool is_wait(instruction_ref ins) const override
        {

            return private_detail_te_value.is_wait(ins);
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
inline const ValueType* any_cast(const stream_model* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType* any_cast(stream_model* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType& any_cast(stream_model& x)
{
    auto* y = x.any_cast<typename std::remove_reference<ValueType>::type>();
    if(y == nullptr)
        throw std::bad_cast();
    return *y;
}

template <typename ValueType>
inline const ValueType& any_cast(const stream_model& x)
{
    const auto* y = x.any_cast<typename std::remove_reference<ValueType>::type>();
    if(y == nullptr)
        throw std::bad_cast();
    return *y;
}
#endif

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
