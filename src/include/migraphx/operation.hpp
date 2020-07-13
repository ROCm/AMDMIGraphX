#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_OPERAND_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_OPERAND_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <migraphx/reflect.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/auto_any_cast.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct context;

#ifdef DOXYGEN

/// The operation interface represents an action an instruction will perform. All
/// operation classes must be CopyConstructible.
struct operation
{
    /// A unique name identifying the operation
    std::string name() const;
    /// An optional method that can be used to finalize the operator before running
    void finalize(context& ctx);
    /// This is used to compute the resulting shape from an operation. If an
    /// operation cannot be run with input shapes, then it should throw an
    /// exception.
    shape compute_shape(const std::vector<shape>& input) const;
    /**
     * @brief This performs the operation's computation.
     *
     * This method can be optional when the operation is only used as a placeholder to be lowered
     * later on.
     *
     * @param ctx This is the context created by the `target` during compilation. Implementations
     * can use the target's `context` class rather than the `context` interface class.
     * @param output This is the output shape. It is equivalent to running `compute_shape` with each
     * `shape` of the `argument`.
     * @param input This is the `argument` result from the previous instruction's computation.
     * @return Return an `argument` of the result computation. The `shape` of `argument` should be
     * the same the `output` shape.
     */
    argument compute(context& ctx, const shape& output, const std::vector<argument>& input) const;
    /// An optional method to return which argument the output will alias. If
    /// there is no aliased output then -1 can be returned.
    std::ptrdiff_t output_alias(const std::vector<shape>& input) const;
    /// An optional stream operator to print the operation. When this is not
    /// implemented, it will just print the operation's name.
    friend std::ostream& operator<<(std::ostream& os, const operation& op);
};

/// Returns true if operation does not require a context to run compute
bool is_context_free(const operation& x);
/// Returns true if the operation has a finalize method
bool has_finalize(const operation& x);

#else

namespace detail {

namespace operation_operators {

template <class T>
auto operator<<(std::ostream& os, const T& x) -> decltype(os << x.name())
{
    os << x.name();
    char delim = '[';
    reflect_each(x, [&](auto&& y, auto name) {
        os << delim;
        os << name << "=";
        stream_write_value(os, y);
        delim = ',';
    });
    if(delim == ',')
        os << "]";
    return os;
}

template <class T, class U>
auto operator==(const T& x, const U& y) -> decltype(x.name() == y.name())
{
    static_assert(is_reflectable<T>{} or sizeof(T) <= 1,
                  "Missing equality operator or reflect method.");
    if(x.name() != y.name())
        return false;
    const auto& yy = any_cast<T>(y);
    return reflect_tie(x) == reflect_tie(yy);
}

} // namespace operation_operators

template <class T>
auto compute_op(rank<2>,
                const T& x,
                context& ctx,
                const shape& output_shape,
                const std::vector<argument>& input)
    -> decltype(x.compute(auto_any_cast(ctx), output_shape, input))
{
    return x.compute(auto_any_cast(ctx), output_shape, input);
}

template <class T>
auto compute_op(
    rank<1>, const T& x, context&, const shape& output_shape, const std::vector<argument>& input)
    -> decltype(x.compute(output_shape, input))
{
    return x.compute(output_shape, input);
}

template <class T>
argument compute_op(rank<0>, const T& x, context&, const shape&, const std::vector<argument>&)
{
    std::string name = x.name();
    MIGRAPHX_THROW("Not computable: " + name);
}

template <class T>
argument
compute_op(const T& x, context& ctx, const shape& output_shape, const std::vector<argument>& input)
{
    return compute_op(rank<2>{}, x, ctx, output_shape, input);
}

template <class T>
auto compute_op(rank<2>, const T& x, const shape& output_shape, const std::vector<argument>& input)
    -> decltype(x.compute(output_shape, input))
{
    return x.compute(output_shape, input);
}

template <class T>
auto compute_op(rank<1>, const T& x, const shape& output_shape, const std::vector<argument>& input)
    -> decltype(x.compute(auto_any_cast(std::declval<context&>()), output_shape, input))
{
    std::string name = x.name();
    MIGRAPHX_THROW("Not computable without a context: " + name);
}

template <class T>
argument compute_op(rank<0>, const T& x, const shape&, const std::vector<argument>&)
{
    std::string name = x.name();
    MIGRAPHX_THROW("Not computable: " + name);
}

template <class T>
argument compute_op(const T& x, const shape& output_shape, const std::vector<argument>& input)
{
    return compute_op(rank<2>{}, x, output_shape, input);
}

template <class T>
auto is_context_free_op(rank<1>,
                        const T& x,
                        const shape& output_shape,
                        const std::vector<argument>& input)
    -> decltype(x.compute(output_shape, input), std::true_type{});

template <class T>
auto is_context_free_op(rank<0>, const T&, const shape&, const std::vector<argument>&)
    -> std::false_type;

template <class T>
auto is_context_free_op(const T& x) -> decltype(is_context_free_op(
    rank<1>{}, x, std::declval<const shape&>(), std::declval<std::vector<argument>>()))
{
    return {};
}

template <class T>
std::ptrdiff_t output_alias_op(const T&, const std::vector<shape>&)
{
    return -1;
}

template <class T>
auto finalize_op(
    rank<1>, T& x, context& ctx, const shape& output_shape, const std::vector<shape>& input)
    -> decltype(x.finalize(auto_any_cast(ctx), output_shape, input), void())
{
    x.finalize(auto_any_cast(ctx), output_shape, input);
}

template <class T>
void finalize_op(rank<0>, T&, context&, const shape&, const std::vector<shape>&)
{
}

template <class T>
void finalize_op(T& x, context& ctx, const shape& output_shape, const std::vector<shape>& input)
{
    finalize_op(rank<1>{}, x, ctx, output_shape, input);
}

template <class T>
auto has_finalize_op(
    rank<1>, T& x, context& ctx, const shape& output_shape, const std::vector<shape>& input)
    -> decltype(x.finalize(auto_any_cast(ctx), output_shape, input), std::true_type{});

template <class T>
auto has_finalize_op(rank<0>, T&, context&, const shape&, const std::vector<shape>&)
    -> std::false_type;

template <class T>
auto has_finalize_op(const T&) -> decltype(has_finalize_op(rank<1>{},
                                                           std::declval<T&>(),
                                                           std::declval<context&>(),
                                                           std::declval<const shape&>(),
                                                           std::declval<std::vector<shape>>()))
{
    return {};
}

template <class T>
value to_value_op(const T& x)
{
    return migraphx::to_value(x);
}

template <class T>
void from_value_op(T& x, const value& v)
{
    return migraphx::from_value(v, x);
}

} // namespace detail

/*
 * Type-erased interface for:
 *
 * struct operation
 * {
 *      std::string name() const;
 *      bool is_context_free() const;
 *      bool has_finalize() const;
 *      std::ptrdiff_t output_alias(const std::vector<shape>& input) const;
 *      void finalize(context& ctx,const shape& output,const std::vector<shape>& input) ;
 *      shape compute_shape(const std::vector<shape>& input) const;
 *      argument compute(context& ctx,const shape& output,const std::vector<argument>& input) const;
 *      argument compute(const shape& output,const std::vector<argument>& input) const;
 *      value to_value() const;
 *      void from_value(const value v) ;
 *     friend std::ostream & operator<<(std::ostream & os,const operation & op) ;
 *     friend bool operator==(const operation & x,const operation & y) ;
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
        using std::swap;
        auto* derived = this->any_cast<PrivateDetailTypeErasedT>();
        if(derived and private_detail_te_handle_mem_var.unique())
        {
            *derived = std::forward<PrivateDetailTypeErasedT>(value);
        }
        else
        {
            operation rhs(value);
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

    bool is_context_free() const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().is_context_free();
    }

    bool has_finalize() const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().has_finalize();
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& input) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().output_alias(input);
    }

    void finalize(context& ctx, const shape& output, const std::vector<shape>& input)
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().finalize(ctx, output, input);
    }

    shape compute_shape(const std::vector<shape>& input) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().compute_shape(input);
    }

    argument compute(context& ctx, const shape& output, const std::vector<argument>& input) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().compute(ctx, output, input);
    }

    argument compute(const shape& output, const std::vector<argument>& input) const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().compute(output, input);
    }

    value to_value() const
    {
        assert((*this).private_detail_te_handle_mem_var);
        return (*this).private_detail_te_get_handle().to_value();
    }

    void from_value(const value v)
    {
        assert((*this).private_detail_te_handle_mem_var);
        (*this).private_detail_te_get_handle().from_value(std::move(v));
    }

    friend std::ostream& operator<<(std::ostream& os, const operation& op)
    {
        assert(op.private_detail_te_handle_mem_var);
        return op.private_detail_te_get_handle().operator_shift_left(os);
    }

    friend bool operator==(const operation& x, const operation& y)
    {
        assert(x.private_detail_te_handle_mem_var);
        return x.private_detail_te_get_handle().operator==(y);
    }

    friend bool is_shared(const operation& private_detail_x, const operation& private_detail_y)
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

        virtual std::string name() const                                           = 0;
        virtual bool is_context_free() const                                       = 0;
        virtual bool has_finalize() const                                          = 0;
        virtual std::ptrdiff_t output_alias(const std::vector<shape>& input) const = 0;
        virtual void
        finalize(context& ctx, const shape& output, const std::vector<shape>& input) = 0;
        virtual shape compute_shape(const std::vector<shape>& input) const           = 0;
        virtual argument
        compute(context& ctx, const shape& output, const std::vector<argument>& input) const    = 0;
        virtual argument compute(const shape& output, const std::vector<argument>& input) const = 0;
        virtual value to_value() const                                                          = 0;
        virtual void from_value(const value v)                                                  = 0;
        virtual std::ostream& operator_shift_left(std::ostream& os) const                       = 0;
        virtual bool operator==(const operation& y) const                                       = 0;
    };

    template <class T>
    static auto private_detail_te_default_is_context_free(char, T&& private_detail_te_self)
        -> decltype(private_detail_te_self.is_context_free())
    {
        return private_detail_te_self.is_context_free();
    }

    template <class T>
    static bool private_detail_te_default_is_context_free(float, T&& private_detail_te_self)
    {
        return detail::is_context_free_op(private_detail_te_self);
    }

    template <class T>
    static auto private_detail_te_default_has_finalize(char, T&& private_detail_te_self)
        -> decltype(private_detail_te_self.has_finalize())
    {
        return private_detail_te_self.has_finalize();
    }

    template <class T>
    static bool private_detail_te_default_has_finalize(float, T&& private_detail_te_self)
    {
        return detail::has_finalize_op(private_detail_te_self);
    }

    template <class T>
    static auto private_detail_te_default_output_alias(char,
                                                       T&& private_detail_te_self,
                                                       const std::vector<shape>& input)
        -> decltype(private_detail_te_self.output_alias(input))
    {
        return private_detail_te_self.output_alias(input);
    }

    template <class T>
    static std::ptrdiff_t private_detail_te_default_output_alias(float,
                                                                 T&& private_detail_te_self,
                                                                 const std::vector<shape>& input)
    {
        return detail::output_alias_op(private_detail_te_self, input);
    }

    template <class T>
    static auto private_detail_te_default_finalize(char,
                                                   T&& private_detail_te_self,
                                                   context& ctx,
                                                   const shape& output,
                                                   const std::vector<shape>& input)
        -> decltype(private_detail_te_self.finalize(ctx, output, input))
    {
        private_detail_te_self.finalize(ctx, output, input);
    }

    template <class T>
    static void private_detail_te_default_finalize(float,
                                                   T&& private_detail_te_self,
                                                   context& ctx,
                                                   const shape& output,
                                                   const std::vector<shape>& input)
    {
        detail::finalize_op(private_detail_te_self, ctx, output, input);
    }

    template <class T>
    static auto private_detail_te_default_compute(char,
                                                  T&& private_detail_te_self,
                                                  context& ctx,
                                                  const shape& output,
                                                  const std::vector<argument>& input)
        -> decltype(private_detail_te_self.compute(ctx, output, input))
    {
        return private_detail_te_self.compute(ctx, output, input);
    }

    template <class T>
    static argument private_detail_te_default_compute(float,
                                                      T&& private_detail_te_self,
                                                      context& ctx,
                                                      const shape& output,
                                                      const std::vector<argument>& input)
    {
        return detail::compute_op(private_detail_te_self, ctx, output, input);
    }

    template <class T>
    static auto private_detail_te_default_compute(char,
                                                  T&& private_detail_te_self,
                                                  const shape& output,
                                                  const std::vector<argument>& input)
        -> decltype(private_detail_te_self.compute(output, input))
    {
        return private_detail_te_self.compute(output, input);
    }

    template <class T>
    static argument private_detail_te_default_compute(float,
                                                      T&& private_detail_te_self,
                                                      const shape& output,
                                                      const std::vector<argument>& input)
    {
        return detail::compute_op(private_detail_te_self, output, input);
    }

    template <class T>
    static auto private_detail_te_default_to_value(char, T&& private_detail_te_self)
        -> decltype(private_detail_te_self.to_value())
    {
        return private_detail_te_self.to_value();
    }

    template <class T>
    static value private_detail_te_default_to_value(float, T&& private_detail_te_self)
    {
        return detail::to_value_op(private_detail_te_self);
    }

    template <class T>
    static auto
    private_detail_te_default_from_value(char, T&& private_detail_te_self, const value v)
        -> decltype(private_detail_te_self.from_value(std::move(v)))
    {
        private_detail_te_self.from_value(std::move(v));
    }

    template <class T>
    static void
    private_detail_te_default_from_value(float, T&& private_detail_te_self, const value v)
    {
        detail::from_value_op(private_detail_te_self, std::move(v));
    }

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

        bool is_context_free() const override
        {

            return private_detail_te_default_is_context_free(char(0), private_detail_te_value);
        }

        bool has_finalize() const override
        {

            return private_detail_te_default_has_finalize(char(0), private_detail_te_value);
        }

        std::ptrdiff_t output_alias(const std::vector<shape>& input) const override
        {

            return private_detail_te_default_output_alias(char(0), private_detail_te_value, input);
        }

        void finalize(context& ctx, const shape& output, const std::vector<shape>& input) override
        {

            private_detail_te_default_finalize(
                char(0), private_detail_te_value, ctx, output, input);
        }

        shape compute_shape(const std::vector<shape>& input) const override
        {

            return private_detail_te_value.compute_shape(input);
        }

        argument compute(context& ctx,
                         const shape& output,
                         const std::vector<argument>& input) const override
        {

            return private_detail_te_default_compute(
                char(0), private_detail_te_value, ctx, output, input);
        }

        argument compute(const shape& output, const std::vector<argument>& input) const override
        {

            return private_detail_te_default_compute(
                char(0), private_detail_te_value, output, input);
        }

        value to_value() const override
        {

            return private_detail_te_default_to_value(char(0), private_detail_te_value);
        }

        void from_value(const value v) override
        {

            private_detail_te_default_from_value(char(0), private_detail_te_value, std::move(v));
        }

        std::ostream& operator_shift_left(std::ostream& os) const override
        {
            using migraphx::detail::operation_operators::operator<<;
            return os << private_detail_te_value;
        }

        bool operator==(const operation& y) const override
        {
            using migraphx::detail::operation_operators::operator==;
            return private_detail_te_value == y;
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

inline bool operator!=(const operation& x, const operation& y) { return !(x == y); }

inline bool is_context_free(const operation& op) { return op.is_context_free(); }

template <class T>
bool is_context_free(const T& x)
{
    return detail::is_context_free_op(x);
}

inline bool has_finalize(const operation& op) { return op.has_finalize(); }

template <class T>
bool has_finalize(const T& x)
{
    return detail::has_finalize_op(x);
}

inline void migraphx_to_value(value& v, const operation& op) { v = op.to_value(); }
inline void migraphx_from_value(const value& v, operation& op) { op.from_value(v); }

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
