#ifndef MIGRAPHX_GUARD_FIND_CONCUR_HPP
#define MIGRAPHX_GUARD_FIND_CONCUR_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <unordered_map>
#include <vector>

#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

#ifdef DOXYGEN

/// An interface for target-dependent analysis to find concurrent instructions
/// executing in different streams.
struct find_concur
{
    void get_concur(program* p,
                    int num_of_streams,
                    std::unordered_map<const instruction*,
                                       std::vector<std::vector<const instruction*>>>& concur_instrs,
                    std::unordered_map<const instruction*, int>& instr2_points);
};

#else

/*
 * Type-erased interface for:
 *
 * struct find_concur
 * {
 *      void get_concur(program* p,int num_of_stream,std::unordered_map<const instruction*,
 * std::vector<std::vector<const instruction*>>>& concur_instrs,std::unordered_map<const
 * instruction*, int>& input) ;
 * };
 *
 */

struct find_concur
{
    // Constructors
    find_concur() = default;

    template <typename PrivateDetailTypeErasedT>
    find_concur(PrivateDetailTypeErasedT value)
        : private_detail_te_handle_mem_var(
              std::make_shared<private_detail_te_handle_type<
                  typename std::remove_reference<PrivateDetailTypeErasedT>::type>>(
                  std::forward<PrivateDetailTypeErasedT>(value)))
    {
    }

    // Assignment
    template <typename PrivateDetailTypeErasedT>
    find_concur& operator=(PrivateDetailTypeErasedT value)
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

    void get_concur(program* p,
                    int num_of_stream,
                    std::unordered_map<const instruction*,
                                       std::vector<std::vector<const instruction*>>>& concur_instrs,
                    std::unordered_map<const instruction*, int>& input)
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().get_concur(p, num_of_stream, concur_instrs, input);
    }

    friend bool is_shared(const find_concur& private_detail_x, const find_concur& private_detail_y)
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

        virtual void
        get_concur(program* p,
                   int num_of_stream,
                   std::unordered_map<const instruction*,
                                      std::vector<std::vector<const instruction*>>>& concur_instrs,
                   std::unordered_map<const instruction*, int>& input) = 0;
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

        void
        get_concur(program* p,
                   int num_of_stream,
                   std::unordered_map<const instruction*,
                                      std::vector<std::vector<const instruction*>>>& concur_instrs,
                   std::unordered_map<const instruction*, int>& input) override
        {

            private_detail_te_value.get_concur(p, num_of_stream, concur_instrs, input);
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
inline const ValueType* any_cast(const find_concur* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType* any_cast(find_concur* x)
{
    return x->any_cast<ValueType>();
}

template <typename ValueType>
inline ValueType& any_cast(find_concur& x)
{
    auto* y = x.any_cast<typename std::remove_reference<ValueType>::type>();
    if(y == nullptr)
        throw std::bad_cast();
    return *y;
}

template <typename ValueType>
inline const ValueType& any_cast(const find_concur& x)
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
