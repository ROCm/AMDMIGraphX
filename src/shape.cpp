
#include <rtg/shape.hpp>
#include <rtg/stringutils.hpp>
#include <numeric>
#include <algorithm>
#include <functional>
#include <iostream>

namespace rtg {

shape::shape() : m_type(float_type), m_packed(false) {}

shape::shape(type_t t) : m_type(t), m_lens({1}), m_strides({1}), m_packed(true) {}
shape::shape(type_t t, std::vector<std::size_t> l) : m_type(t), m_lens(std::move(l)), m_packed(true)
{
    this->calculate_strides();
    assert(m_lens.size() == m_strides.size());
}
shape::shape(type_t t, std::vector<std::size_t> l, std::vector<std::size_t> s)
    : m_type(t), m_lens(std::move(l)), m_strides(std::move(s))
{
    assert(m_lens.size() == m_strides.size());
    assert(std::any_of(m_strides.begin(), m_strides.end(), [](auto x) { return x > 0; }) and
           "At least one stride must be non-zero");
    m_packed = this->elements() == this->element_space();
}

void shape::calculate_strides()
{
    m_strides.clear();
    m_strides.resize(m_lens.size(), 0);
    if(m_strides.empty())
        return;
    m_strides.back() = 1;
    std::partial_sum(
        m_lens.rbegin(), m_lens.rend() - 1, m_strides.rbegin() + 1, std::multiplies<std::size_t>());
}

shape::type_t shape::type() const { return this->m_type; }
const std::vector<std::size_t>& shape::lens() const { return this->m_lens; }
const std::vector<std::size_t>& shape::strides() const { return this->m_strides; }
std::size_t shape::elements() const
{
    assert(this->lens().size() == this->strides().size());
    return std::accumulate(
        this->lens().begin(), this->lens().end(), std::size_t{1}, std::multiplies<std::size_t>());
}
std::size_t shape::bytes() const
{
    std::size_t n = 0;
    this->visit_type([&](auto as) { n = as.size(); });
    return n * this->element_space();
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
    if(this->packed())
        return i;
    else
        return std::inner_product(this->lens().begin(),
                                  this->lens().end(),
                                  this->strides().begin(),
                                  std::size_t{0},
                                  std::plus<std::size_t>{},
                                  [&](std::size_t len, std::size_t stride) {
                                      assert(stride > 0 and len > 0);
                                      return ((i / stride) % len) * stride;
                                  });
}
bool shape::packed() const { return this->m_packed; }

bool shape::broadcasted() const
{
    assert(this->lens().size() == this->strides().size());
    return std::accumulate(
        this->strides().begin(), this->strides().end(), std::size_t{1}, std::multiplies<std::size_t>()) == 0; 
}

std::size_t shape::element_space() const
{
    // TODO: Get rid of intermediate vector
    assert(this->lens().size() == this->strides().size());
    std::vector<std::size_t> max_indices(this->lens().size());
    std::transform(this->lens().begin(),
                   this->lens().end(),
                   std::vector<std::size_t>(this->lens().size(), 1).begin(),
                   max_indices.begin(),
                   std::minus<std::size_t>());
    return std::inner_product(
               max_indices.begin(), max_indices.end(), this->strides().begin(), std::size_t{0}) +
           1;
}

std::string shape::type_string() const
{
    switch(this->m_type)
    {
    case any_type: return "any";
#define RTG_SHAPE_TYPE_STRING_CASE(x, t) \
    case x: return #x;
        RTG_SHAPE_VISIT_TYPES(RTG_SHAPE_TYPE_STRING_CASE)
#undef RTG_SHAPE_TYPE_STRING_CASE
    }
    RTG_THROW("Invalid type");
}

bool operator==(const shape& x, const shape& y)
{
    return x.type() == y.type() && x.lens() == y.lens() && x.strides() == y.strides();
}
bool operator!=(const shape& x, const shape& y) { return !(x == y); }

std::ostream& operator<<(std::ostream& os, const shape& x)
{
    os << x.type_string() << ", ";
    os << "{" << to_string(x.lens()) << "}, ";
    os << "{" << to_string(x.strides()) << "}";
    return os;
}

} // namespace rtg
