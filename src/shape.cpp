
#include <rtg/shape.hpp>
#include <rtg/stringutils.hpp>
#include <numeric>
#include <algorithm>
#include <functional>

namespace rtg {

shape::shape() : type_(float_type), lens_(), strides_(), packed_(false) {}

shape::shape(type_t t) : type_(t), lens_({1}), strides_({1}), packed_(true) {}
shape::shape(type_t t, std::vector<std::size_t> l) : type_(t), lens_(std::move(l)), packed_(true)
{
    this->calculate_strides();
    assert(lens_.size() == strides_.size());
}
shape::shape(type_t t, std::vector<std::size_t> l, std::vector<std::size_t> s)
    : type_(t), lens_(std::move(l)), strides_(std::move(s))
{
    assert(lens_.size() == strides_.size());
    packed_ = this->elements() == this->element_space();
}

void shape::calculate_strides()
{
    strides_.clear();
    strides_.resize(lens_.size(), 0);
    if(strides_.empty())
        return;
    strides_.back() = 1;
    std::partial_sum(
        lens_.rbegin(), lens_.rend() - 1, strides_.rbegin() + 1, std::multiplies<std::size_t>());
}

shape::type_t shape::type() const { return this->type_; }
const std::vector<std::size_t>& shape::lens() const { return this->lens_; }
const std::vector<std::size_t>& shape::strides() const { return this->strides_; }
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
    return std::inner_product(
        this->lens().begin(),
        this->lens().end(),
        this->strides().begin(),
        std::size_t{0},
        std::plus<std::size_t>{},
        [&](std::size_t len, std::size_t stride) { return ((i / stride) % len) * stride; });
}
bool shape::packed() const { return this->packed_; }
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
    switch(this->type_)
    {
#define RTG_SHAPE_TYPE_STRING_CASE(x, t) \
    case x: return #x;
        RTG_SHAPE_VISIT_TYPES(RTG_SHAPE_TYPE_STRING_CASE)
#undef RTG_SHAPE_TYPE_STRING_CASE
    }
    throw "Invalid type";
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
