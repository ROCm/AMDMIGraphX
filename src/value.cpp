#include <cassert>
#include <iostream>
#include <migraphx/cloneable.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/value.hpp>
#include <unordered_map>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct value_base_impl : cloneable<value_base_impl>
{
    virtual value::type_t get_type() { return value::null_type; }
#define MIGRAPHX_VALUE_GENERATE_BASE_FUNCTIONS(vt, cpp_type) \
    virtual const cpp_type* if_##vt() const { return nullptr; }
    MIGRAPHX_VISIT_VALUE_TYPES(MIGRAPHX_VALUE_GENERATE_BASE_FUNCTIONS)
    virtual std::vector<value>* if_array() { return nullptr; }
    virtual std::unordered_map<std::string, std::size_t>* if_object() { return nullptr; }
    virtual value_base_impl* if_value() const { return nullptr; }
    value_base_impl()                       = default;
    value_base_impl(const value_base_impl&) = default;
    value_base_impl& operator=(const value_base_impl&) = default;
    virtual ~value_base_impl() {}
};

#define MIGRAPHX_VALUE_GENERATE_BASE_TYPE(vt, cpp_type)               \
    struct vt##_value_holder : value_base_impl::share                 \
    {                                                                 \
        vt##_value_holder(cpp_type d) : data(std::move(d)) {}         \
        virtual value::type_t get_type() { return value::vt##_type; } \
        virtual const cpp_type* if_##vt() const { return &data; }     \
        cpp_type data;                                                \
    };
MIGRAPHX_VISIT_VALUE_TYPES(MIGRAPHX_VALUE_GENERATE_BASE_TYPE)

struct array_value_holder : value_base_impl::derive<array_value_holder>
{
    array_value_holder() {}
    array_value_holder(std::vector<value> d) : data(std::move(d)) {}
    virtual value::type_t get_type() { return value::array_type; }
    virtual std::vector<value>* if_array() { return &data; }
    std::vector<value> data;
};

struct object_value_holder : value_base_impl::derive<object_value_holder>
{
    object_value_holder() {}
    object_value_holder(std::vector<value> d, std::unordered_map<std::string, std::size_t> l)
        : data(std::move(d)), lookup(std::move(l))
    {
    }
    virtual value::type_t get_type() { return value::object_type; }
    virtual std::vector<value>* if_array() { return &data; }
    virtual std::unordered_map<std::string, std::size_t>* if_object() { return &lookup; }
    std::vector<value> data;
    std::unordered_map<std::string, std::size_t> lookup;
};

value::value(const value& rhs) : x(rhs.x ? rhs.x->clone() : nullptr), key(rhs.key) {}
value& value::operator=(value rhs)
{
    std::swap(rhs.x, x);
    if(not rhs.key.empty())
        std::swap(rhs.key, key);
    return *this;
}

void set_vector(std::shared_ptr<value_base_impl>& x,
                const std::vector<value>& v,
                bool array_on_empty = true)
{
    if(v.empty())
    {
        if(array_on_empty)
            x = std::make_shared<array_value_holder>();
        else
            x = std::make_shared<object_value_holder>();
        return;
    }
    if(v.front().get_key().empty())
    {
        x = std::make_shared<array_value_holder>(v);
    }
    else
    {
        std::unordered_map<std::string, std::size_t> lookup;
        std::size_t i = 0;
        for(auto&& e : v)
        {
            lookup[e.get_key()] = i;
            i++;
        }
        x = std::make_shared<object_value_holder>(v, lookup);
    }
}

value::value(const std::initializer_list<value>& i) : x(nullptr)
{
    if(i.size() == 2 and i.begin()->is_string())
    {
        key    = i.begin()->get_string();
        auto r = (i.begin() + 1)->x;
        x      = r ? r->clone() : nullptr;
        return;
    }
    set_vector(x, std::vector<value>(i.begin(), i.end()));
}

value::value(const std::vector<value>& v, bool array_on_empty) : x(nullptr)
{
    set_vector(x, v, array_on_empty);
}

value::value(const std::unordered_map<std::string, value>& m)
    : value(std::vector<value>(m.begin(), m.end()), false)
{
}

value::value(const std::string& pkey, const std::vector<value>& v, bool array_on_empty)
    : x(nullptr), key(pkey)
{
    set_vector(x, v, array_on_empty);
}

value::value(const std::string& pkey, const std::unordered_map<std::string, value>& m)
    : value(pkey, std::vector<value>(m.begin(), m.end()), false)
{
}

value::value(const std::string& pkey, std::nullptr_t) : x(nullptr), key(pkey) {}

value::value(const std::string& pkey, const value& rhs)
    : x(rhs.x ? rhs.x->clone() : nullptr), key(pkey)
{
}

value::value(const char* i) : value(std::string(i)) {}

#define MIGRAPHX_VALUE_GENERATE_DEFINE_METHODS(vt, cpp_type)                           \
    value::value(cpp_type i) : x(std::make_shared<vt##_value_holder>(std::move(i))) {} \
    value::value(const std::string& pkey, cpp_type i)                                  \
        : x(std::make_shared<vt##_value_holder>(std::move(i))), key(pkey)              \
    {                                                                                  \
    }                                                                                  \
    value& value::operator=(cpp_type rhs)                                              \
    {                                                                                  \
        x = std::make_shared<vt##_value_holder>(std::move(rhs));                       \
        return *this;                                                                  \
    }                                                                                  \
    bool value::is_##vt() const { return x ? x->get_type() == vt##_type : false; }     \
    const cpp_type& value::get_##vt() const                                            \
    {                                                                                  \
        auto* r = this->if_##vt();                                                     \
        assert(r);                                                                     \
        return *r;                                                                     \
    }                                                                                  \
    const cpp_type* value::if_##vt() const { return x ? x->if_##vt() : nullptr; }
MIGRAPHX_VISIT_VALUE_TYPES(MIGRAPHX_VALUE_GENERATE_DEFINE_METHODS)

bool value::is_array() const { return x ? x->get_type() == array_type : false; }
const std::vector<value>& value::value::get_array() const
{
    auto* r = this->if_array();
    assert(r);
    return *r;
}
const std::vector<value>* value::if_array() const { return x ? x->if_array() : nullptr; }

bool value::is_object() const { return x ? x->get_type() == object_type : false; }
const std::vector<value>& value::get_object() const
{
    auto* r = this->if_object();
    assert(r);
    return *r;
}
const std::vector<value>* value::if_object() const
{
    auto* r = x ? x->if_array() : nullptr;
    assert(r == nullptr or
           std::none_of(r->begin(), r->end(), [](auto&& v) { return v.get_key().empty(); }));
    return r;
}

bool value::is_null() const { return x == nullptr; }

const std::string& value::get_key() const { return key; }

std::vector<value>* if_array_impl(const std::shared_ptr<value_base_impl>& x)
{
    if(!x)
        return nullptr;
    return x->if_array();
}

std::vector<value>& get_array_impl(const std::shared_ptr<value_base_impl>& x)
{
    auto* a = if_array_impl(x);
    assert(a);
    return *a;
}

value* find_impl(const std::shared_ptr<value_base_impl>& x, const std::string& key)
{
    auto* a = if_array_impl(x);
    if(a == nullptr)
        return nullptr;
    auto* lookup = x->if_object();
    if(lookup == nullptr)
        return nullptr;
    auto it = lookup->find(key);
    if(it == lookup->end())
        return a->data() + a->size();
    return std::addressof((*a)[it->second]);
}

value* value::find(const std::string& pkey) { return find_impl(x, pkey); }

const value* value::find(const std::string& pkey) const { return find_impl(x, pkey); }
bool value::contains(const std::string& pkey) const
{
    auto it = find(pkey);
    if(it == nullptr)
        return false;
    if(it == end())
        return false;
    return true;
}
std::size_t value::size() const
{
    auto* a = if_array_impl(x);
    if(a == nullptr)
        return 0;
    return a->size();
}
bool value::empty() const { return size() == 0; }
const value* value::data() const
{
    auto* a = if_array_impl(x);
    if(a == nullptr)
        return nullptr;
    return a->data();
}
value* value::data()
{
    auto* a = if_array_impl(x);
    if(a == nullptr)
        return nullptr;
    return a->data();
}
value* value::begin()
{
    // cppcheck-suppress assertWithSideEffect
    assert(data() or empty());
    return data();
}
const value* value::begin() const
{
    assert(data() or empty());
    return data();
}
value* value::end() { return begin() + size(); }
const value* value::end() const { return begin() + size(); }

value& value::front() { return *begin(); }
const value& value::front() const { return *begin(); }
value& value::back() { return *std::prev(end()); }
const value& value::back() const { return *std::prev(end()); }
value& value::at(std::size_t i)
{
    auto* a = if_array_impl(x);
    if(a == nullptr)
        MIGRAPHX_THROW("Not an array");
    return a->at(i);
}
const value& value::at(std::size_t i) const
{
    auto* a = if_array_impl(x);
    if(a == nullptr)
        MIGRAPHX_THROW("Not an array");
    return a->at(i);
}
value& value::at(const std::string& pkey)
{
    auto* r = find(pkey);
    if(r == nullptr)
        MIGRAPHX_THROW("Not an object");
    if(r == end())
        MIGRAPHX_THROW("Key not found");
    return *r;
}
const value& value::at(const std::string& pkey) const
{
    auto* r = find(pkey);
    if(r == nullptr)
        MIGRAPHX_THROW("Not an object");
    if(r == end())
        MIGRAPHX_THROW("Key not found");
    return *r;
}
value& value::operator[](std::size_t i) { return *(begin() + i); }
const value& value::operator[](std::size_t i) const { return *(begin() + i); }
value& value::operator[](const std::string& pkey) { return *emplace(pkey, nullptr).first; }

std::pair<value*, bool> value::insert(const value& v)
{
    if(v.key.empty())
    {
        if(!x)
            x = std::make_shared<array_value_holder>();
        get_array_impl(x).push_back(v);
        assert(this->if_array());
        return std::make_pair(&back(), true);
    }
    else
    {
        if(!x)
            x = std::make_shared<object_value_holder>();
        auto p = x->if_object()->emplace(v.key, get_array_impl(x).size());
        if(p.second)
            get_array_impl(x).push_back(v);
        assert(this->if_object());
        return std::make_pair(&get_array_impl(x)[p.first->second], p.second);
    }
}
value* value::insert(const value* pos, const value& v)
{
    assert(v.key.empty());
    if(!x)
        x = std::make_shared<array_value_holder>();
    auto&& a = get_array_impl(x);
    auto it  = a.insert(a.begin() + (pos - begin()), v);
    return std::addressof(*it);
}

value value::without_key() const
{
    value result = *this;
    result.key   = "";
    return result;
}

value value::with_key(const std::string& pkey) const
{
    value result = *this;
    result.key   = pkey;
    return result;
}

template <class F, class T, class U, class Common = typename std::common_type<T, U>::type>
auto compare_common_impl(
    rank<1>, F f, const std::string& keyx, const T& x, const std::string& keyy, const U& y)
{
    return f(std::forward_as_tuple(keyx, Common(x)), std::forward_as_tuple(keyy, Common(y)));
}

template <class F>
auto compare_common_impl(
    rank<1>, F f, const std::string& keyx, std::nullptr_t, const std::string& keyy, std::nullptr_t)
{
    return f(std::forward_as_tuple(keyx, 0), std::forward_as_tuple(keyy, 0));
}

template <class F, class T, class U>
auto compare_common_impl(rank<0>, F, const std::string&, const T&, const std::string&, const U&)
{
    return false;
}

template <class F>
bool compare(const value& x, const value& y, F f)
{
    bool result = false;
    x.visit([&](auto&& a) {
        y.visit([&](auto&& b) {
            result = compare_common_impl(rank<1>{}, f, x.get_key(), a, y.get_key(), b);
        });
    });
    return result;
}

value::type_t value::get_type() const
{
    if(!x)
        return null_type;
    return x->get_type();
}

bool operator==(const value& x, const value& y)
{
    if(x.get_type() != y.get_type())
        return false;
    return compare(x, y, std::equal_to<>{});
}
bool operator!=(const value& x, const value& y) { return !(x == y); }
bool operator<(const value& x, const value& y) { return compare(x, y, std::less<>{}); }
bool operator<=(const value& x, const value& y) { return x == y or x < y; }
bool operator>(const value& x, const value& y) { return y < x; }
bool operator>=(const value& x, const value& y) { return x == y or x > y; }

template <class T>
void print_value(std::ostream& os, const T& x)
{
    os << x;
}

template <class T, class U>
void print_value(std::ostream& os, const std::pair<T, U>& x)
{
    os << x.first;
    os << ": ";
    print_value(os, x.second);
}

void print_value(std::ostream& os, const std::nullptr_t&) { os << "null"; }
void print_value(std::ostream& os, const std::vector<value>& x)
{
    os << "{";
    os << to_string_range(x);
    os << "}";
}

std::ostream& operator<<(std::ostream& os, const value& d)
{
    d.visit([&](auto&& y) { print_value(os, y); });
    return os;
}

void value::debug_print() const { std::cout << *this << std::endl; }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
