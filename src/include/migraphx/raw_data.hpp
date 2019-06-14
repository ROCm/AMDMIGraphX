
#ifndef MIGRAPHX_GUARD_RAW_DATA_HPP
#define MIGRAPHX_GUARD_RAW_DATA_HPP

#include <migraphx/tensor_view.hpp>
#include <migraphx/requires.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct raw_data_base
{
};

/**
 * @brief Provides a base class for common operations with raw buffer
 *
 * For classes that handle a raw buffer of data, this will provide common operations such as equals,
 * printing, and visitors. To use this class the derived class needs to provide a `data()` method to
 * retrieve a raw pointer to the data, and `get_shape` method that provides the shape of the data.
 *
 */
template <class Derived>
struct raw_data : raw_data_base
{
    template <class Stream>
    friend Stream& operator<<(Stream& os, const Derived& d)
    {
        if(not d.empty())
            d.visit([&](auto x) { os << x; });
        return os;
    }

    /**
     * @brief Visits a single data element at a certain index.
     *
     * @param v A function which will be called with the type of data
     * @param n The index to read from
     */
    template <class Visitor>
    void visit_at(Visitor v, std::size_t n = 0) const
    {
        auto&& derived = static_cast<const Derived&>(*this);
        if(derived.empty())
            MIGRAPHX_THROW("Visiting empty data!");
        auto&& s      = derived.get_shape();
        auto&& buffer = derived.data();
        s.visit_type([&](auto as) { v(*(as.from(buffer) + s.index(n))); });
    }

    /**
     * @brief Visits the data
     *
     *  This will call the visitor function with a `tensor_view<T>` based on the shape of the data.
     *
     * @param v A function to be called with `tensor_view<T>`
     */
    template <class Visitor>
    void visit(Visitor v) const
    {
        auto&& derived = static_cast<const Derived&>(*this);
        if(derived.empty())
            MIGRAPHX_THROW("Visiting empty data!");
        auto&& s      = derived.get_shape();
        auto&& buffer = derived.data();
        s.visit_type([&](auto as) { v(make_view(s, as.from(buffer))); });
    }

    /// Returns true if the raw data is only one element
    bool single() const
    {
        auto&& s = static_cast<const Derived&>(*this).get_shape();
        return s.elements() == 1;
    }

    /**
     * @brief Retrieves a single element of data
     *
     * @param n The index to retrieve the data from
     * @tparam T The type of data to be retrieved
     * @return The element as `T`
     */
    template <class T>
    T at(std::size_t n = 0) const
    {
        T result;
        this->visit_at([&](auto x) { result = x; }, n);
        return result;
    }

    struct auto_cast
    {
        const Derived* self;
        template <class T>
        operator T()
        {
            assert(self->single());
            return self->template at<T>();
        }

        template <class T>
        using is_data_ptr =
            bool_c<(std::is_void<T>{} or std::is_same<char, std::remove_cv_t<T>>{} or
                    std::is_same<unsigned char, std::remove_cv_t<T>>{})>;

        template <class T>
        using get_data_type = std::conditional_t<is_data_ptr<T>{}, float, T>;

        template <class T>
        bool matches() const
        {
            return is_data_ptr<T>{} ||
                   self->get_shape().type() == migraphx::shape::get_type<get_data_type<T>>{};
        }

        template <class T>
        operator T*()
        {
            using type = std::remove_cv_t<T>;
            assert(matches<T>());
            return reinterpret_cast<type*>(self->data());
        }
    };

    /// Implicit conversion of raw data pointer
    auto_cast implicit() const { return {static_cast<const Derived*>(this)}; }

    /// Get a tensor_view to the data
    template <class T>
    tensor_view<T> get() const
    {
        auto&& s      = static_cast<const Derived&>(*this).get_shape();
        auto&& buffer = static_cast<const Derived&>(*this).data();
        if(s.type() != migraphx::shape::get_type<T>{})
            MIGRAPHX_THROW("Incorrect data type for raw data");
        return make_view(s, reinterpret_cast<T*>(buffer));
    }

    /// Cast the data pointer
    template <class T>
    T* cast() const
    {
        auto&& s      = static_cast<const Derived&>(*this).get_shape();
        auto&& buffer = static_cast<const Derived&>(*this).data();
        assert(s.type() == migraphx::shape::get_type<T>{});
        return reinterpret_cast<T*>(buffer);
    }
};

template <class T,
          class U,
          MIGRAPHX_REQUIRES(std::is_base_of<raw_data_base, T>{} &&
                            std::is_base_of<raw_data_base, U>{})>
bool operator==(const T& x, const U& y)
{
    auto&& xshape = x.get_shape();
    auto&& yshape = y.get_shape();
    bool result   = x.empty() && y.empty();
    if(not result && xshape == yshape)
    {
        auto&& xbuffer = x.data();
        auto&& ybuffer = y.data();
        // TODO: Dont use tensor view for single values
        xshape.visit_type([&](auto as) {
            auto xview = make_view(xshape, as.from(xbuffer));
            auto yview = make_view(yshape, as.from(ybuffer));
            result     = xview == yview;
        });
    }
    return result;
}

template <class T,
          class U,
          MIGRAPHX_REQUIRES(std::is_base_of<raw_data_base, T>{} &&
                            std::is_base_of<raw_data_base, U>{})>
bool operator!=(const T& x, const U& y)
{
    return !(x == y);
}

namespace detail {
template <class V, class... Ts>
void visit_all_impl(const shape& s, V&& v, Ts&&... xs)
{
    s.visit_type([&](auto as) { v(make_view(xs.get_shape(), as.from(xs.data()))...); });
}
} // namespace detail

/**
 * @brief Visits every object together
 * @details This will visit every object, but assumes each object is the same type. This can reduce
 * the deeply nested visit calls. This will return a function that will take the visitor callback.
 * So it will be called with `visit_all(xs...)([](auto... ys) {})` where `xs...` and `ys...` are the
 * same number of parameters.
 *
 * @param x A raw data object
 * @param xs Many raw data objects
 * @return A function to be called with the visitor
 */
template <class T, class... Ts>
auto visit_all(T&& x, Ts&&... xs)
{
    auto&& s                                   = x.get_shape();
    std::initializer_list<shape::type_t> types = {xs.get_shape().type()...};
    if(!std::all_of(types.begin(), types.end(), [&](shape::type_t t) { return t == s.type(); }))
        MIGRAPHX_THROW("Types must be the same");
    return [&](auto v) {
        // Workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70100
        detail::visit_all_impl(s, v, x, xs...);
    };
}

template <class T>
auto visit_all(const std::vector<T>& x)
{
    auto&& s = x.front().get_shape();
    if(!std::all_of(
           x.begin(), x.end(), [&](const T& y) { return y.get_shape().type() == s.type(); }))
        MIGRAPHX_THROW("Types must be the same");
    return [&](auto v) {
        s.visit_type([&](auto as) {
            using type = typename decltype(as)::type;
            std::vector<tensor_view<type>> result;
            for(const auto& y : x)
                result.push_back(make_view(y.get_shape(), as.from(y.data())));
            v(result);
        });
    };
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
