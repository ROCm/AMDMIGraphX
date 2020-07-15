#ifndef MIGRAPHX_GUARD_RTGLIB_VALUE_HPP
#define MIGRAPHX_GUARD_RTGLIB_VALUE_HPP

#include <migraphx/config.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/requires.hpp>
#include <migraphx/type_name.hpp>
#include <migraphx/rank.hpp>
#include <algorithm>
#include <memory>
#include <sstream>
#include <type_traits>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct value_base_impl;

template <class To>
struct value_converter
{
    template <class T = To>
    static auto apply(const std::string& x)
        -> decltype((std::declval<std::stringstream&>() >> std::declval<T&>()), To{})
    {
        To result;
        std::stringstream ss;
        ss.str(x);
        ss >> result;
        if(ss.fail())
            throw std::runtime_error("Failed to parse: " + x);
        return result;
    }

    template <class From, MIGRAPHX_REQUIRES(std::is_convertible<From, To>{})>
    static To apply(const From& x)
    {
        return To(x);
    }
};

template <>
struct value_converter<std::string>
{
    static const std::string& apply(const std::string& x) { return x; }

    static std::string apply(const std::nullptr_t&) { return "null"; }

    template <class From>
    static auto apply(const From& x) -> decltype(std::declval<std::stringstream&>() << x, std::string())
    {
        std::stringstream ss;
        ss << x;
        if(ss.fail())
            throw std::runtime_error("Failed to parse");
        return ss.str();
    }
};

template <class T, class U>
struct value_converter<std::pair<T, U>>
{
    template <class Key, class From>
    static auto apply(const std::pair<Key, From>& x)
        -> decltype(std::pair<T, U>(x.first, value_converter<U>::apply(x.second)))
    {
        return std::pair<T, U>(x.first, value_converter<U>::apply(x.second));
    }
};

namespace detail {
template <class To, class From>
auto try_convert_value_impl(rank<1>, const From& x) -> decltype(value_converter<To>::apply(x))
{
    return value_converter<To>::apply(x);
}

template <class To, class From>
To try_convert_value_impl(rank<0>, const From& x)
{
    MIGRAPHX_THROW("Incompatible values: " + get_type_name(x) + " -> " + get_type_name<To>());
}
} // namespace detail

template <class To, class From>
To try_convert_value(const From& x)
{
    return detail::try_convert_value_impl<To>(rank<1>{}, x);
}

struct value
{
// clang-format off
#define MIGRAPHX_VISIT_VALUE_TYPES(m) \
    m(int64, std::int64_t) \
    m(uint64, std::uint64_t) \
    m(float, double) \
    m(string, std::string) \
    m(bool, bool)
    // clang-format on
    enum type_t
    {
#define MIGRAPHX_VALUE_GENERATE_ENUM_TYPE(vt, cpp_type) vt##_type,
        MIGRAPHX_VISIT_VALUE_TYPES(MIGRAPHX_VALUE_GENERATE_ENUM_TYPE) object_type,
        array_type,
        null_type
#undef MIGRAPHX_VALUE_GENERATE_ENUM_TYPE
    };
    using iterator        = value*;
    using const_iterator  = const value*;
    using value_type      = value;
    using key_type        = std::string;
    using mapped_type     = value;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using array           = std::vector<value>;
    using object          = std::unordered_map<std::string, value>;

    value() = default;

    value(const value& rhs);
    value& operator=(value rhs);
    value(const std::string& pkey, const value& rhs);

    value(const std::initializer_list<value>& i);
    value(const std::vector<value>& v, bool array_on_empty = true);
    value(const std::unordered_map<std::string, value>& m);
    value(const std::string& pkey, const std::vector<value>& v, bool array_on_empty = true);
    value(const std::string& pkey, const std::unordered_map<std::string, value>& m);
    value(const std::string& pkey, std::nullptr_t);

    value(const char* i);

#define MIGRAPHX_VALUE_GENERATE_DECL_METHODS(vt, cpp_type) \
    value(cpp_type i);                                     \
    value(const std::string& pkey, cpp_type i);            \
    value& operator=(cpp_type rhs);                        \
    bool is_##vt() const;                                  \
    const cpp_type& get_##vt() const;                      \
    const cpp_type* if_##vt() const;
    MIGRAPHX_VISIT_VALUE_TYPES(MIGRAPHX_VALUE_GENERATE_DECL_METHODS)

    template <class T>
    using pick = std::conditional_t<
        std::is_floating_point<T>{},
        double,
        std::conditional_t<std::is_signed<T>{},
                           std::int64_t,
                           std::conditional_t<std::is_unsigned<T>{}, std::uint64_t, T>>>;

    template <class T>
    using is_pickable =
        std::integral_constant<bool, (std::is_arithmetic<T>{} and not std::is_pointer<T>{})>;

    template <class T, MIGRAPHX_REQUIRES(is_pickable<T>{})>
    value(T i) : value(pick<T>{i})
    {
    }
    template <class T, MIGRAPHX_REQUIRES(is_pickable<T>{})>
    value(const std::string& pkey, T i) : value(pkey, pick<T>{i})
    {
    }
    template <class T, class U, class = decltype(value(T{}, U{}))>
    value(const std::pair<T, U>& p) : value(p.first, p.second)
    {
    }
    template <class T, MIGRAPHX_REQUIRES(is_pickable<T>{})>
    value& operator=(T rhs)
    {
        return *this = pick<T>{rhs}; // NOLINT
    }

    bool is_array() const;
    const std::vector<value>& get_array() const;
    const std::vector<value>* if_array() const;

    bool is_object() const;
    const std::vector<value>& get_object() const;
    const std::vector<value>* if_object() const;

    bool is_null() const;

    const std::string& get_key() const;
    value* find(const std::string& pkey);
    const value* find(const std::string& pkey) const;
    bool contains(const std::string& pkey) const;
    std::size_t size() const;
    bool empty() const;
    const value* data() const;
    value* data();
    value* begin();
    const value* begin() const;
    value* end();
    const value* end() const;

    value& front();
    const value& front() const;
    value& back();
    const value& back() const;
    value& at(std::size_t i);
    const value& at(std::size_t i) const;
    value& at(const std::string& pkey);
    const value& at(const std::string& pkey) const;
    value& operator[](std::size_t i);
    const value& operator[](std::size_t i) const;
    value& operator[](const std::string& pkey);

    std::pair<value*, bool> insert(const value& v);
    value* insert(const value* pos, const value& v);

    template <class... Ts>
    std::pair<value*, bool> emplace(Ts&&... xs)
    {
        return insert(value(std::forward<Ts>(xs)...));
    }

    template <class... Ts>
    value* emplace(const value* pos, Ts&&... xs)
    {
        return insert(pos, value(std::forward<Ts>(xs)...));
    }

    void push_back(const value& v) { insert(end(), v); }

    void push_front(const value& v) { insert(begin(), v); }

    value without_key() const;

    template <class Visitor>
    void visit(Visitor v) const
    {
        switch(this->get_type())
        {
        case null_type:
        {
            v(std::nullptr_t{});
            return;
        }
#define MIGRAPHX_VALUE_GENERATE_CASE(vt, cpp_type)                          \
    case vt##_type:                                                         \
    {                                                                       \
        if(this->key.empty())                                               \
            v(this->get_##vt());                                            \
        else                                                                \
            v(std::make_pair(this->get_key(), std::ref(this->get_##vt()))); \
        return;                                                             \
    }
            MIGRAPHX_VISIT_VALUE_TYPES(MIGRAPHX_VALUE_GENERATE_CASE)
            MIGRAPHX_VALUE_GENERATE_CASE(array, )
            MIGRAPHX_VALUE_GENERATE_CASE(object, )
        }
        MIGRAPHX_THROW("Unknown type");
    }

    template <class To>
    To to() const
    {
        To result;
        this->visit([&](auto y) { result = try_convert_value<To>(y); });
        return result;
    }

    template <class To>
    std::vector<To> to_vector() const
    {
        std::vector<To> result;
        const auto& values = is_object() ? get_object() : get_array();
        result.reserve(values.size());
        std::transform(values.begin(), values.end(), std::back_inserter(result), [&](auto v) {
            return v.template to<To>();
        });
        return result;
    }

    friend bool operator==(const value& x, const value& y);
    friend bool operator!=(const value& x, const value& y);
    friend bool operator<(const value& x, const value& y);
    friend bool operator<=(const value& x, const value& y);
    friend bool operator>(const value& x, const value& y);
    friend bool operator>=(const value& x, const value& y);

    friend std::ostream& operator<<(std::ostream& os, const value& d);

    void debug_print() const;

    private:
    type_t get_type() const;
    std::shared_ptr<value_base_impl> x;
    std::string key;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
