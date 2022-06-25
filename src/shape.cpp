/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <migraphx/shape.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/permutation.hpp>
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

    shape_impl() : m_type(shape::float_type) {}

    shape_impl(shape::type_t t) : m_type(t), m_lens({1}), m_strides({0}), m_standard(true)
    {
        assert(t != shape::tuple_type);
    }
    shape_impl(shape::type_t t, std::vector<std::size_t> l)
        : m_type(t), m_lens(std::move(l)), m_standard(true)
    {
        assert(t != shape::tuple_type);
        this->calculate_strides();
        assert(m_lens.size() == m_strides.size());
    }
    shape_impl(shape::type_t t, std::vector<std::size_t> l, std::vector<std::size_t> s)
        : m_type(t), m_lens(std::move(l)), m_strides(std::move(s))
    {
        assert(t != shape::tuple_type);
        assert(m_lens.size() == m_strides.size());
        m_standard = this->elements() == this->element_space() and not skips() and
                     std::is_sorted(m_strides.rbegin(), m_strides.rend());
    }

    shape_impl(const std::vector<shape>& subs) : m_type(shape::tuple_type), m_shapes(subs) {}
    shape::type_t m_type;
    std::vector<std::size_t> m_lens    = {};
    std::vector<std::size_t> m_strides = {};
    std::vector<shape> m_shapes        = {};
    bool m_standard                    = false;

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

    // Does the shape skip over elements?
    bool skips() const
    {
        assert(m_lens.size() == m_strides.size());
        if(elements() == 1)
            return false;
        return std::none_of(m_strides.begin(), m_strides.end(), [](auto x) { return x == 1; });
    }

    std::shared_ptr<shape_impl> copy() const { return std::make_shared<shape_impl>(*this); }
};

const std::vector<shape::type_t>& shape::types()
{
    static const std::vector<shape::type_t> result = {
#define MIGRAPHX_GENERATE_TYPE_VECTOR(x, t) x,
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_GENERATE_TYPE_VECTOR) tuple_type};
    return result;
}

std::string shape::name(shape::type_t t)
{
    switch(t)
    {
    case tuple_type: return "tuple_type";
#define MIGRAPHX_SHAPE_GENERATE_TYPE_NAME_CASE(x, t) \
    case x: return #x;
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_GENERATE_TYPE_NAME_CASE)
#undef MIGRAPHX_SHAPE_GENERATE_TYPE_NAME_CASE
    }
    MIGRAPHX_THROW("Invalid type");
}
std::string shape::cpp_type(shape::type_t t)
{
    switch(t)
    {
    case tuple_type: MIGRAPHX_THROW("No C++ type for tuple");
#define MIGRAPHX_SHAPE_GENERATE_CPP_TYPE_CASE(x, t) \
    case x: return #t;
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_GENERATE_CPP_TYPE_CASE)
#undef MIGRAPHX_SHAPE_GENERATE_CPP_TYPE_CASE
    }
    MIGRAPHX_THROW("Invalid type");
}

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

shape::shape(const std::vector<shape>& subs) : impl(std::make_shared<shape_impl>(subs)) {}

shape::shape(std::shared_ptr<shape_impl> pimpl) : impl(std::move(pimpl)) {}

shape shape::from_permutation(type_t t,
                              const std::vector<std::size_t>& l,
                              const std::vector<int64_t>& perm)
{
    auto new_lens = reorder_dims(l, perm);
    shape result  = reorder_shape({t, new_lens}, invert_permutation(perm));
    assert(result.lens() == l);
    return result;
}

shape::type_t shape::type() const { return impl->m_type; }
const std::vector<std::size_t>& shape::lens() const { return impl->m_lens; }
const std::vector<std::size_t>& shape::strides() const { return impl->m_strides; }
std::size_t shape::elements() const { return impl->elements(); }
std::size_t shape::bytes() const
{
    if(this->sub_shapes().empty())
    {
        std::size_t n = 0;
        this->visit_type([&](auto as) { n = as.size(); });
        return n * this->element_space();
    }
    else
    {
        return std::accumulate(this->sub_shapes().begin(),
                               this->sub_shapes().end(),
                               std::size_t{0},
                               [&](auto x, auto y) { return x + y.bytes(); });
    }
}
std::size_t shape::type_size() const
{
    std::size_t n = 0;
    if(this->sub_shapes().empty())
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

void shape::multi_copy(std::size_t i, std::size_t* start, const std::size_t* end) const
{
    assert(this->standard());
    (void)end;
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

bool shape::packed() const
{
    return this->sub_shapes().empty() and not impl->skips() and
           this->elements() == this->element_space();
}

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
    return std::any_of(
        this->strides().begin(), this->strides().end(), [](auto x) { return x == 0; });
}

bool shape::scalar() const
{
    assert(this->lens().size() == this->strides().size());
    // if any stride > 0, then accumulate will return false
    return this->sub_shapes().empty() and
           std::accumulate(this->strides().begin(), this->strides().end(), std::size_t(0)) == 0;
}

bool shape::standard() const { return impl->m_standard; }

shape shape::normalize_standard() const
{
    if(this->standard())
        return {this->type(), this->lens()};
    else
        return *this;
}

shape shape::with_lens(type_t t, const std::vector<std::size_t>& l) const
{
    assert(l.size() == this->lens().size());
    auto perm = find_permutation(*this);
    return shape::from_permutation(t, l, perm);
}

shape shape::with_lens(const std::vector<std::size_t>& l) const
{
    return this->with_lens(this->type(), l);
}

shape shape::with_type(type_t t) const
{
    auto c    = impl->copy();
    c->m_type = t;
    return {c};
}

std::size_t shape::element_space() const { return impl->element_space(); }

std::string shape::type_string() const { return name(this->type()); }

bool operator==(const shape& x, const shape& y)
{
    return x.impl == y.impl or (x.type() == y.type() and x.lens() == y.lens() and
                                x.strides() == y.strides() and x.sub_shapes() == y.sub_shapes());
}
bool operator!=(const shape& x, const shape& y) { return !(x == y); }

std::ostream& operator<<(std::ostream& os, const shape& x)
{
    if(x.sub_shapes().empty())
    {
        os << x.type_string() << ", ";
        os << "{" << to_string_range(x.lens()) << "}, ";
        os << "{" << to_string_range(x.strides()) << "}";
    }
    else
    {
        os << "[" << to_string_range(x.sub_shapes()) << "]";
    }
    return os;
}

shape::type_t shape::parse_type(const std::string& s)
{
    static const std::unordered_map<std::string, shape::type_t> m = {
#define MIGRAPHX_SHAPE_GENERATE_TYPE_STRING_MAP(x, t) {#x, x}, {#t, x},
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_GENERATE_TYPE_STRING_MAP){"tuple_type",
                                                                            tuple_type},
        {"tuple", tuple_type}};
    return m.at(s);
}

const std::vector<shape>& shape::sub_shapes() const { return impl->m_shapes; }

void migraphx_to_value(value& v, const shape& s)
{
    value result;
    result["type"]       = migraphx::to_value(s.type_string());
    result["lens"]       = migraphx::to_value(s.lens());
    result["strides"]    = migraphx::to_value(s.strides());
    result["sub_shapes"] = migraphx::to_value(s.sub_shapes());
    v                    = result;
}
void migraphx_from_value(const value& v, shape& s)
{
    auto t = v.at("type").get_string();
    if(t == "tuple_type")
    {
        s = shape{migraphx::from_value<std::vector<migraphx::shape>>(v.at("sub_shapes"))};
    }
    else
    {
        s = shape{shape::parse_type(t),
                  v.at("lens").to_vector<std::size_t>(),
                  v.at("strides").to_vector<std::size_t>()};
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
