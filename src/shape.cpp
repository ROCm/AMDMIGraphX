
#include <migraphx/shape.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/serialize.hpp>
#include <numeric>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct shape_impl
{
    static std::shared_ptr<shape_impl> default_shape()
    {
        static const std::shared_ptr<shape_impl> result = std::make_shared<shape_impl>();
        return result;
    }

    shape_impl() : m_type(shape::float_type), m_standard(false) {}

    shape_impl(shape::type_t t) : m_type(t), m_lens({1}), m_strides({0}), m_standard(true) {}
    shape_impl(shape::type_t t, std::vector<std::size_t> l)
        : m_type(t), m_lens(std::move(l)), m_standard(true)
    {
        this->calculate_strides();
        assert(m_lens.size() == m_strides.size());
    }
    shape_impl(shape::type_t t, std::vector<std::size_t> l, std::vector<std::size_t> s)
        : m_type(t), m_lens(std::move(l)), m_strides(std::move(s))
    {
        assert(m_lens.size() == m_strides.size());
        // assert(std::any_of(m_strides.begin(), m_strides.end(), [](auto x) { return x > 0; }) and
        //        "At least one stride must be non-zero");
        m_standard = this->elements() == this->element_space() and
                     std::is_sorted(m_strides.rbegin(), m_strides.rend());
    }
    shape::type_t m_type;
    std::vector<std::size_t> m_lens;
    std::vector<std::size_t> m_strides;
    bool m_standard;

    void calculate_strides()
    {
        m_strides.clear();
        m_strides.resize(m_lens.size(), 0);
        if(m_strides.empty())
            return;
        m_strides.back() = 1;
        std::partial_sum(m_lens.rbegin(),
                         m_lens.rend() - 1,
                         m_strides.rbegin() + 1,
                         std::multiplies<std::size_t>());
    }

    std::size_t element_space() const
    {
        assert(m_lens.size() == m_strides.size());
        if(m_lens.empty())
            return 0;
        return std::inner_product(m_lens.begin(),
                                  m_lens.end(),
                                  m_strides.begin(),
                                  std::size_t{0},
                                  std::plus<std::size_t>{},
                                  [](std::size_t l, std::size_t s) { return (l - 1) * s; }) +
               1;
    }

    std::size_t elements() const
    {
        assert(m_lens.size() == m_strides.size());
        if(m_lens.empty())
            return 0;
        return std::accumulate(
            m_lens.begin(), m_lens.end(), std::size_t{1}, std::multiplies<std::size_t>());
    }
};

shape::shape() : impl(shape_impl::default_shape()) {}

shape::shape(type_t t) : impl(std::make_shared<shape_impl>(t)) {}
shape::shape(type_t t, std::vector<std::size_t> l)
    : impl(std::make_shared<shape_impl>(t, std::move(l)))
{
}
shape::shape(type_t t, std::vector<std::size_t> l, std::vector<std::size_t> s)
    : impl(std::make_shared<shape_impl>(t, std::move(l), std::move(s)))
{
}

shape::type_t shape::type() const { return impl->m_type; }
const std::vector<std::size_t>& shape::lens() const { return impl->m_lens; }
const std::vector<std::size_t>& shape::strides() const { return impl->m_strides; }
std::size_t shape::elements() const { return impl->elements(); }
std::size_t shape::bytes() const
{
    std::size_t n = 0;
    this->visit_type([&](auto as) { n = as.size(); });
    return n * this->element_space();
}
std::size_t shape::type_size() const
{
    std::size_t n = 0;
    this->visit_type([&](auto as) { n = as.size(); });
    return n;
}
std::size_t shape::index(std::initializer_list<std::size_t> l) const
{
    assert(l.size() <= this->lens().size());
    assert(this->lens().size() == this->strides().size());
    return std::inner_product(l.begin(), l.end(), this->strides().begin(), std::size_t{0});
}
std::size_t shape::index(const std::vector<std::size_t>& l) const
{
    assert(l.size() <= this->lens().size());
    assert(this->lens().size() == this->strides().size());
    return std::inner_product(l.begin(), l.end(), this->strides().begin(), std::size_t{0});
}
std::size_t shape::index(std::size_t i) const
{
    assert(this->lens().size() == this->strides().size());
    if(this->standard())
        return i;
    else
    {
        std::size_t s      = 1;
        std::size_t result = 0;
        for(std::size_t j = 0; j < this->lens().size(); j++)
        {
            const std::size_t k      = this->lens().size() - j - 1;
            const std::size_t stride = this->strides()[k];
            const std::size_t len    = this->lens()[k];
            const std::size_t idx    = (i % (s * len)) / s;
            result += stride * idx;
            s *= len;
        }
        return result;
    }
}

std::vector<std::size_t> shape::multi(std::size_t i) const
{
    assert(this->standard());

    std::vector<std::size_t> indices(lens().size());
    multi_copy(i, indices.data(), indices.data() + lens().size());

    return indices;
}

void shape::multi_copy(std::size_t i, std::size_t* start, std::size_t* end) const
{
    assert(this->standard());
    assert(lens().size() <= (end - start));
    std::transform(strides().begin(),
                   strides().end(),
                   lens().begin(),
                   start,
                   [&](std::size_t stride, std::size_t len) {
                       assert(len > 0 and stride > 0);
                       return (i / stride) % len;
                   });
}

bool shape::packed() const { return this->elements() == this->element_space(); }

bool shape::transposed() const
{
    if(this->broadcasted())
    {
        // TODO: Use a filter_iterator instead
        std::vector<std::size_t> s;
        s.reserve(this->strides().size());
        std::copy_if(this->strides().begin(),
                     this->strides().end(),
                     std::back_inserter(s),
                     [](std::size_t x) { return x != 0; });
        return not std::is_sorted(s.rbegin(), s.rend());
    }
    else
    {
        return not std::is_sorted(this->strides().rbegin(), this->strides().rend());
    }
}

bool shape::broadcasted() const
{
    assert(this->lens().size() == this->strides().size());
    return std::accumulate(this->strides().begin(),
                           this->strides().end(),
                           std::size_t{1},
                           std::multiplies<std::size_t>()) == 0;
}

bool shape::scalar() const
{
    assert(this->lens().size() == this->strides().size());
    // if any stride > 0, then accumulate will return false
    return std::accumulate(this->strides().begin(), this->strides().end(), std::size_t(0)) == 0;
}

bool shape::standard() const { return impl->m_standard; }

shape shape::normalize_standard() const
{
    if(this->standard())
        return {this->type(), this->lens()};
    else
        return *this;
}

std::size_t shape::element_space() const { return impl->element_space(); }

std::string shape::type_string() const
{
    switch(this->type())
    {
#define MIGRAPHX_SHAPE_GENERATE_TYPE_STRING_CASE(x, t) \
    case x: return #x;
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_GENERATE_TYPE_STRING_CASE)
#undef MIGRAPHX_SHAPE_GENERATE_TYPE_STRING_CASE
    }
    MIGRAPHX_THROW("Invalid type");
}

bool operator==(const shape& x, const shape& y)
{
    return x.type() == y.type() && x.lens() == y.lens() && x.strides() == y.strides();
}
bool operator!=(const shape& x, const shape& y) { return !(x == y); }

std::ostream& operator<<(std::ostream& os, const shape& x)
{
    os << x.type_string() << ", ";
    os << "{" << to_string_range(x.lens()) << "}, ";
    os << "{" << to_string_range(x.strides()) << "}";
    return os;
}

shape::type_t shape::parse_type(const std::string& s)
{
    static const std::unordered_map<std::string, shape::type_t> m = {
#define MIGRAPHX_SHAPE_GENERATE_TYPE_STRING_MAP(x, t) {#x, x}, {#t, x},
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_GENERATE_TYPE_STRING_MAP)};
    return m.at(s);
}

void migraphx_to_value(value& v, const shape& s)
{
    value result;
    result["type"]    = migraphx::to_value(s.type_string());
    result["lens"]    = migraphx::to_value(s.lens());
    result["strides"] = migraphx::to_value(s.strides());
    v                 = result;
}
void migraphx_from_value(const value& v, shape& s)
{
    s = shape{shape::parse_type(v.at("type").get_string()),
              v.at("lens").to_vector<std::size_t>(),
              v.at("strides").to_vector<std::size_t>()};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
