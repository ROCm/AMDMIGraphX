#ifndef MIGRAPHX_GUARD_RTGLIB_VALUE_HPP
#define MIGRAPHX_GUARD_RTGLIB_VALUE_HPP

#include <migraphx/config.hpp>
#include <memory>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct value_base_impl;

struct value
{
#define MIGRAPHX_VISIT_VALUE_TYPES(m)                                                       \
    m(int64, std::int64_t) m(uint64, std::uint64_t) m(float, double) m(string, std::string) \
        m(bool, bool)
    enum type_t
    {
#define MIGRAPHX_VALUE_ENUM_TYPE(vt, cpp_type) vt##_type,
        MIGRAPHX_VISIT_VALUE_TYPES(MIGRAPHX_VALUE_ENUM_TYPE) object_type,
        array_type,
        null_type
#undef MIGRAPHX_VALUE_ENUM_TYPE
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

    value() = default;

    value(const value& rhs);
    value& operator=(value rhs);
    value(const std::string& pkey, const value& rhs);

    value(const std::vector<value>& v);
    value(const std::string& pkey, const std::vector<value>& v);
    value(const std::string& pkey, std::nullptr_t);

#define MIGRAPHX_VALUE_DECL_METHODS(vt, cpp_type) \
    value(cpp_type i);                            \
    value(const std::string& pkey, cpp_type i);   \
    value& operator=(cpp_type rhs);               \
    bool is_##vt() const;                         \
    const cpp_type& get_##vt() const;             \
    const cpp_type* if_##vt() const;
    MIGRAPHX_VISIT_VALUE_TYPES(MIGRAPHX_VALUE_DECL_METHODS)

    bool is_array() const;
    const std::vector<value>& get_array() const;
    const std::vector<value>* if_array() const;

    bool is_object() const;
    const std::vector<value>& get_object() const;
    const std::vector<value>* if_object() const;

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

    template <class Visitor>
    void visit(Visitor v) const
    {
        if(!x)
            v(std::nullptr_t{}) switch(x->get_type())
            {
#define MIGRAPHX_VALUE_CASE(vt, cpp_type)                 \
    case vt##_type:                                       \
    {                                                     \
        if(x.key.empty())                                 \
            v(x.get_##vt());                              \
        else                                              \
            v(std::make_pair(x.get_key(), x.get_##vt())); \
        break;                                            \
    }
                MIGRAPHX_VISIT_VALUE_TYPES(MIGRAPHX_VALUE_CASE)
            case array_type:
            {
                if()
                    break;
            }
            }
    }

    private:
    std::shared_ptr<value_base_impl> x;
    std::string key;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
