/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/sym.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/ranges.hpp>
#include <numeric>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>
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
    }

    shape_impl(shape::type_t t, std::vector<std::size_t> l, std::vector<std::size_t> s)
        : m_type(t), m_lens(std::move(l)), m_strides(std::move(s))
    {
        assert(t != shape::tuple_type);
        assert(m_lens.size() == m_strides.size());

        // Calculate standard shape flag for these lens/strides.  Strides on size-1
        // axes are ignored to support an MLIR rule.
        std::vector<size_t> filtered_strides;
        for(size_t ind = 0; ind < m_strides.size(); ind++)
            if(m_lens[ind] != 1)
                filtered_strides.push_back(m_strides[ind]);
        m_standard = this->elements() == this->element_space() and not skips() and
                     std::is_sorted(filtered_strides.rbegin(), filtered_strides.rend());
    }

    shape_impl(shape::type_t t, std::vector<shape::dynamic_dimension> dims)
        : m_type(t), m_dyn_dims(std::move(dims))
    {
        if(not m_dyn_dims.empty() and std::all_of(m_dyn_dims.begin(),
                                                  m_dyn_dims.end(),
                                                  [](const auto& d) { return d.is_symbolic(); }))
        {
            calculate_dyn_strides();
            m_standard = true;
        }
    }

    shape_impl(shape::type_t t,
               std::vector<shape::dynamic_dimension> dims,
               std::vector<sym::expr> dstrides)
        : m_type(t), m_dyn_dims(std::move(dims)), m_dyn_strides(std::move(dstrides))
    {
    }

    shape_impl(shape::type_t t,
               std::vector<std::size_t> mins,
               std::vector<std::size_t> maxes,
               std::vector<std::set<std::size_t>> optimals_list)
        : m_type(t)
    {
        if(optimals_list.empty())
        {
            for(size_t i = 0; i < mins.size(); ++i)
            {
                m_dyn_dims.push_back(shape::dynamic_dimension{mins[i], maxes[i]});
            }
        }
        else
        {
            assert(mins.size() == maxes.size() and maxes.size() == optimals_list.size());
            for(size_t i = 0; i < mins.size(); ++i)
            {
                m_dyn_dims.push_back(shape::dynamic_dimension{mins[i], maxes[i], optimals_list[i]});
            }
        }
        if(not m_dyn_dims.empty() and std::all_of(m_dyn_dims.begin(),
                                                  m_dyn_dims.end(),
                                                  [](const auto& d) { return d.is_symbolic(); }))
        {
            calculate_dyn_strides();
            m_standard = true;
        }
    }

    shape_impl(const std::vector<shape>& subs) : m_type(shape::tuple_type), m_shapes(subs) {}

    shape::type_t m_type;
    std::vector<std::size_t> m_lens    = {};
    std::vector<std::size_t> m_strides = {};
    std::vector<shape> m_shapes        = {};
    bool m_standard                    = false;

    std::vector<shape::dynamic_dimension> m_dyn_dims = {};
    std::vector<sym::expr> m_dyn_strides             = {};

    void calculate_dyn_strides()
    {
        m_dyn_strides.clear();
        if(m_dyn_dims.empty())
            return;
        m_dyn_strides.resize(m_dyn_dims.size());
        m_dyn_strides.back() = sym::lit(1);
        std::transform(m_dyn_dims.rbegin(),
                       m_dyn_dims.rend() - 1,
                       m_dyn_strides.rbegin(),
                       m_dyn_strides.rbegin() + 1,
                       [](const auto& dd, const auto& stride) {
                           return dd.sym_expr.value_or(sym::lit(dd.min)) * stride;
                       });
    }

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
        if(not m_dyn_dims.empty())
        {
            auto maxes          = max_lens();
            std::size_t max_val = std::numeric_limits<std::size_t>::max();

            return std::accumulate(
                maxes.begin(), maxes.end(), std::size_t{1}, [&](std::size_t x, std::size_t y) {
                    // overflow check and clip
                    if(x != 0 and y > max_val / x)
                    {
                        return max_val;
                    }
                    return x * y;
                });
        }

        assert(m_lens.size() == m_strides.size());
        return compute_element_space<std::size_t>(m_lens, m_strides);
    }

    std::size_t elements() const
    {
        if(not m_dyn_dims.empty())
        {
            MIGRAPHX_THROW("SHAPE: elements() called on dynamic shape");
        }

        assert(m_lens.size() == m_strides.size());
        return compute_elements<std::size_t>(m_lens);
    }

    std::size_t get_index(size_t i) const
    {
        std::size_t result = 0;
        std::size_t s      = 1;

        for(auto k : migraphx::reverse(migraphx::range(m_lens.size())))
        {
            std::size_t stride = m_strides[k];
            std::size_t len    = m_lens[k];
            std::size_t idx    = (i % (s * len)) / s;
            result += stride * idx;
            s *= len;
        }
        return result;
    }

    std::vector<std::size_t> min_lens() const
    {
        std::vector<std::size_t> ret(m_dyn_dims.size());
        std::transform(m_dyn_dims.cbegin(),
                       m_dyn_dims.cend(),
                       ret.begin(),
                       [](const shape::dynamic_dimension& x) { return x.min; });
        return ret;
    }

    std::vector<std::size_t> max_lens() const
    {
        std::vector<std::size_t> ret(m_dyn_dims.size());
        std::transform(m_dyn_dims.cbegin(),
                       m_dyn_dims.cend(),
                       ret.begin(),
                       [](const shape::dynamic_dimension& x) { return x.max; });
        return ret;
    }

    std::vector<std::set<std::size_t>> opt_lens() const
    {
        std::vector<std::set<std::size_t>> ret(m_dyn_dims.size());
        std::transform(m_dyn_dims.cbegin(),
                       m_dyn_dims.cend(),
                       ret.begin(),
                       [](const shape::dynamic_dimension& x) { return x.optimals; });
        return ret;
    }

    // Does the shape skip over elements?
    bool skips() const
    {
        assert(m_lens.size() == m_strides.size());
        return compute_skips<std::size_t>(m_lens, m_strides);
    }

    template <class T>
    static T make_identity(int64_t n)
    {
        if constexpr(std::is_same_v<T, sym::expr>)
            return sym::lit(n);
        else
            return T(n);
    }

    std::vector<sym::expr> sym_dim_exprs() const
    {
        std::vector<sym::expr> result(m_dyn_dims.size());
        std::transform(m_dyn_dims.cbegin(), m_dyn_dims.cend(), result.begin(), [](const auto& dd) {
            return dd.sym_expr.value_or(sym::lit(dd.min));
        });
        return result;
    }

    template <class T>
    static T compute_elements(const std::vector<T>& dims)
    {
        if(dims.empty())
            return make_identity<T>(0);
        return std::accumulate(dims.begin(), dims.end(), make_identity<T>(1), std::multiplies<>{});
    }

    template <class T>
    static T compute_element_space(const std::vector<T>& dims, const std::vector<T>& strides)
    {
        if(dims.empty())
            return make_identity<T>(0);
        auto one = make_identity<T>(1);
        return std::inner_product(dims.begin(),
                                  dims.end(),
                                  strides.begin(),
                                  make_identity<T>(0),
                                  std::plus<>{},
                                  [&](const T& l, const T& s) { return (l - one) * s; }) +
               one;
    }

    template <class T>
    static bool compute_skips(const std::vector<T>& dims, const std::vector<T>& strides)
    {
        if(compute_elements<T>(dims) == make_identity<T>(1))
            return false;
        auto one = make_identity<T>(1);
        return std::none_of(
            strides.begin(), strides.end(), [&](const auto& x) { return x == one; });
    }

    template <class T>
    static bool compute_packed(const std::vector<T>& dims, const std::vector<T>& strides)
    {
        return not compute_skips<T>(dims, strides) and
               compute_elements<T>(dims) == compute_element_space<T>(dims, strides);
    }

    template <class T>
    static bool compute_broadcasted(const std::vector<T>& strides)
    {
        auto zero = make_identity<T>(0);
        return std::any_of(
            strides.begin(), strides.end(), [&](const auto& x) { return x == zero; });
    }

    template <class T>
    static bool compute_scalar(const std::vector<T>& strides)
    {
        auto zero = make_identity<T>(0);
        return std::accumulate(strides.begin(), strides.end(), zero) == zero;
    }

    template <class T>
    static bool compute_transposed(const std::vector<T>& strides)
    {
        if(compute_broadcasted<T>(strides))
        {
            std::vector<T> s;
            s.reserve(strides.size());
            auto zero = make_identity<T>(0);
            std::copy_if(strides.begin(), strides.end(), std::back_inserter(s), [&](const auto& x) {
                return x != zero;
            });
            return not std::is_sorted(s.rbegin(), s.rend());
        }
        return not std::is_sorted(strides.rbegin(), strides.rend());
    }

    bool is_packed() const
    {
        if(not m_dyn_dims.empty())
        {
            if(m_dyn_strides.empty())
                return false;
            return compute_packed<sym::expr>(sym_dim_exprs(), m_dyn_strides);
        }
        return compute_packed<std::size_t>(m_lens, m_strides);
    }

    bool is_broadcasted() const
    {
        if(not m_dyn_dims.empty())
        {
            if(m_dyn_strides.empty())
                return false;
            return compute_broadcasted<sym::expr>(m_dyn_strides);
        }
        return compute_broadcasted<std::size_t>(m_strides);
    }

    bool is_transposed() const
    {
        if(not m_dyn_dims.empty())
        {
            if(m_dyn_strides.empty())
                return false;
            return compute_transposed<sym::expr>(m_dyn_strides);
        }
        return compute_transposed<std::size_t>(m_strides);
    }

    bool is_scalar() const
    {
        if(not m_dyn_dims.empty())
        {
            if(m_dyn_strides.empty())
                return false;
            return compute_scalar<sym::expr>(m_dyn_strides);
        }
        return compute_scalar<std::size_t>(m_strides);
    }

    std::shared_ptr<shape_impl> copy() const { return std::make_shared<shape_impl>(*this); }
};

std::string shape::to_sizes_string(const std::vector<shape>& shapes)
{
    std::vector<std::string> sizes;
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(sizes), [&](const shape& s) {
        std::string r = to_string_range(s.lens(), "x");
        if(not s.standard())
            r += ":" + to_string_range(s.strides(), "x");
        return r;
    });
    return join_strings(sizes, ", ");
}

const std::vector<shape::type_t>& shape::types()
{
    static const std::vector<shape::type_t> result = {
    // clang-format off
#define MIGRAPHX_GENERATE_TYPE_VECTOR(x, t) x,
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_GENERATE_TYPE_VECTOR)
        tuple_type};
    // clang-format on
    return result;
}

std::string shape::name(shape::type_t t)
{
    switch(t)
    {
    case tuple_type: return "tuple_type";
    case fp4x2_type: return "fp4x2_type";
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
    case fp4x2_type: return "uint8_t";
#define MIGRAPHX_SHAPE_GENERATE_CPP_TYPE_CASE(x, t) \
    case x: return #t;
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_GENERATE_CPP_TYPE_CASE)
#undef MIGRAPHX_SHAPE_GENERATE_CPP_TYPE_CASE
    }
    MIGRAPHX_THROW("Invalid type");
}

bool shape::is_integral(shape::type_t t)
{
    bool result = false;
    visit(t, [&](auto as) { result = as.is_integral(); });
    return result;
}

bool shape::is_compatible(const shape& actual, const shape& expected)
{
    // Check subshapes
    if(expected.type() == shape::tuple_type)
        return migraphx::equal(actual.sub_shapes(), expected.sub_shapes(), &is_compatible);
    if(actual == expected)
        return true;
    if(actual.type() != expected.type())
        return false;
    // Only the expected can be dynamic
    if(expected.dynamic())
        return actual.ndim() == expected.ndim();
    if(actual.dynamic())
        return false;
    if(actual.lens() != expected.lens())
        return false;
    // Check strides from dimensions that are not 1
    return all_of(range(actual.lens().size()), [&](auto i) {
        if(actual.lens()[i] == 1)
            return true;
        return actual.strides()[i] == expected.strides()[i];
    });
}

bool shape::is_compatible_lens(const shape& actual, const shape& expected)
{
    if(actual.dynamic())
        return expected.dynamic() and actual.dyn_dims() == expected.dyn_dims();
    if(expected.dynamic())
    {
        if(actual.ndim() != expected.ndim())
            return false;
        return std::equal(actual.lens().begin(),
                          actual.lens().end(),
                          expected.dyn_dims().begin(),
                          [&](auto a, const auto& e) { return a >= e.min and a <= e.max; });
    }
    return actual.lens() == expected.lens();
}

bool shape::is_unsigned(shape::type_t t)
{
    bool result = false;
    visit(t, [&](auto as) { result = as.is_unsigned(); });
    return result;
}

bool shape::is_computable(shape::type_t t) { return t != shape::fp4x2_type; }

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

shape::shape(type_t t, std::initializer_list<std::size_t> d)
    : shape::shape(t, std::vector<std::size_t>{d.begin(), d.end()})
{
}

shape::shape(type_t t, std::initializer_list<std::size_t> l, std::initializer_list<std::size_t> s)
    : shape::shape(t,
                   std::vector<std::size_t>{l.begin(), l.end()},
                   std::vector<std::size_t>{s.begin(), s.end()})
{
}

shape::shape(type_t t, std::vector<shape::dynamic_dimension> dims)
    : impl(std::make_shared<shape_impl>(t, std::move(dims)))
{
}

shape::shape(type_t t, std::vector<dynamic_dimension> dims, std::vector<sym::expr> dstrides)
    : impl(std::make_shared<shape_impl>(t, std::move(dims), std::move(dstrides)))
{
}

shape::shape(type_t t,
             std::vector<std::size_t> mins,
             std::vector<std::size_t> maxes,
             std::vector<std::set<std::size_t>> optimals_list)
    : impl(std::make_shared<shape_impl>(
          t, std::move(mins), std::move(maxes), std::move(optimals_list)))
{
}

shape::shape(const std::vector<shape>& subs) : impl(std::make_shared<shape_impl>(subs)) {}

shape::shape(std::shared_ptr<shape_impl> pimpl) : impl(std::move(pimpl)) {}

template <class Dims>
static shape
from_permutation_impl(shape::type_t t, const Dims& dims, const std::vector<int64_t>& perm)
{
    auto reordered = reorder_dims(dims, perm);
    return reorder_shape({t, reordered}, invert_permutation(perm));
}

shape shape::from_permutation(type_t t,
                              const std::vector<std::size_t>& l,
                              const std::vector<int64_t>& perm)
{
    shape result = from_permutation_impl(t, l, perm);
    assert(result.lens() == l);
    return result;
}

shape shape::from_permutation(type_t t,
                              const std::vector<dynamic_dimension>& dds,
                              const std::vector<int64_t>& perm)
{
    shape result = from_permutation_impl(t, dds, perm);
    assert(result.dyn_dims() == dds);
    return result;
}

shape::type_t shape::type() const { return impl->m_type; }

const std::vector<std::size_t>& shape::lens() const
{
    if(this->dynamic())
    {
        MIGRAPHX_THROW("SHAPE: lens() called on a dynamic shape");
    }
    return impl->m_lens;
}

const std::vector<std::size_t>& shape::strides() const
{
    if(this->dynamic())
    {
        MIGRAPHX_THROW("SHAPE: strides() called on a dynamic shape");
    }
    return impl->m_strides;
}

std::size_t shape::ndim() const
{
    if(this->dynamic())
    {
        return impl->m_dyn_dims.size();
    }
    return impl->m_lens.size();
}

std::size_t shape::elements() const { return impl->elements(); }

std::size_t shape::bytes() const
{
    if(this->sub_shapes().empty())
    {
        std::size_t n = 0;
        if(type() == fp4x2_type)
        {
            n = sizeof(uint8_t);
        }
        else
        {
            this->visit_type([&](auto as) { n = as.size(); });
        }
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
    {
        if(this->computable())
        {
            this->visit_type([&](auto as) { n = as.size(); });
        }
        else
        {
            n = sizeof(uint8_t);
        }
    }
    return n;
}

std::size_t shape::index(std::initializer_list<std::size_t> l) const
{
    if(this->dynamic())
    {
        MIGRAPHX_THROW("SHAPE: index() called on dynamic shape");
    }
    assert(l.size() <= this->lens().size());
    assert(this->lens().size() == this->strides().size());
    return std::inner_product(l.begin(), l.end(), this->strides().begin(), std::size_t{0});
}

std::size_t shape::index(const std::vector<std::size_t>& l) const
{
    if(this->dynamic())
    {
        MIGRAPHX_THROW("SHAPE: index() called on dynamic shape");
    }
    assert(l.size() <= this->lens().size());
    assert(this->lens().size() == this->strides().size());
    return std::inner_product(l.begin(), l.end(), this->strides().begin(), std::size_t{0});
}

std::size_t shape::index(std::size_t i) const
{
    if(this->dynamic())
    {
        MIGRAPHX_THROW("SHAPE: index() called on dynamic shape");
    }
    assert(this->lens().size() == this->strides().size());
    if(this->standard())
        return i;

    return impl->get_index(i);
}

std::vector<std::size_t> shape::multi(std::size_t idx) const
{
    assert(idx < elements());
    std::vector<std::size_t> indices(lens().size());
    multi_copy(idx, indices.data(), indices.data() + lens().size());
    return indices;
}

void shape::multi_copy(std::size_t idx, std::size_t* start, const std::size_t* end) const
{
    size_t tidx = idx;
    (void)end;
    assert(idx < elements());
    assert(lens().size() <= (end - start));
    for(size_t ii = lens().size() - 1; ii > 0; ii--)
    {
        *(start + ii) = tidx % lens()[ii];
        tidx          = tidx / lens()[ii];
    }
    *start = tidx;
}

bool shape::multi_within_bounds(std::vector<std::size_t> multi) const
{
    assert(this->lens().size() == multi.size());
    return std::equal(multi.begin(), multi.end(), this->lens().begin(), std::less<>{});
}

std::size_t shape::single(const std::vector<std::size_t>& idx) const
{
    return this->single(idx.begin(), idx.end());
}

bool shape::packed() const
{
    if(this->dynamic() and not this->symbolic())
        return false;
    return this->sub_shapes().empty() and impl->is_packed();
}

bool shape::transposed() const
{
    if(this->dynamic() and not this->symbolic())
        return false;
    return impl->is_transposed();
}

bool shape::broadcasted() const
{
    if(this->dynamic() and not this->symbolic())
        return false;
    return impl->is_broadcasted();
}

bool shape::scalar() const
{
    if(this->dynamic() and not this->symbolic())
        return false;
    return this->sub_shapes().empty() and impl->is_scalar();
}

bool shape::standard() const { return impl->m_standard; }

shape shape::normalize_standard() const
{
    if(this->standard())
        return {this->type(), this->lens()};
    else
        return *this;
}

shape shape::as_standard() const
{
    if(not this->dynamic())
        return {this->type(), this->lens()};
    else
        return *this;
}

shape shape::with_lens(type_t t, const std::vector<std::size_t>& l) const
{
    if(this->dynamic())
    {
        MIGRAPHX_THROW("SHAPE: with_lens() called on dynamic shape");
    }
    assert(l.size() == this->lens().size());
    auto perm = find_permutation(*this);
    return shape::from_permutation(t, l, perm);
}

shape shape::with_lens(const std::vector<std::size_t>& l) const
{
    if(this->dynamic())
    {
        MIGRAPHX_THROW("SHAPE: with_lens() called on dynamic shape");
    }
    return this->with_lens(this->type(), l);
}

shape shape::with_lens(type_t t, const std::vector<dynamic_dimension>& dds) const
{
    auto perm = find_permutation(*this);
    return shape::from_permutation(t, dds, perm);
}

shape shape::with_lens(const std::vector<dynamic_dimension>& dds) const
{
    return this->with_lens(this->type(), dds);
}

shape shape::with_type(type_t t) const
{
    auto c    = impl->copy();
    c->m_type = t;
    return {c};
}

shape shape::to_dynamic() const
{
    if(not sub_shapes().empty())
    {
        std::vector<shape> subs;
        std::transform(sub_shapes().cbegin(),
                       sub_shapes().cend(),
                       std::back_inserter(subs),
                       [](auto s) { return s.to_dynamic(); });
        return shape(subs);
    }
    if(this->dynamic())
    {
        return *this;
    }
    std::vector<dynamic_dimension> dims;
    dims.reserve(ndim());
    std::transform(lens().begin(), lens().end(), std::back_inserter(dims), [](auto len) {
        return dynamic_dimension{len, len};
    });
    std::vector<sym::expr> dstrides;
    dstrides.reserve(ndim());
    std::transform(strides().begin(), strides().end(), std::back_inserter(dstrides), [](auto s) {
        return sym::lit(s);
    });
    return {type(), std::move(dims), std::move(dstrides)};
}

shape shape::to_static(std::size_t x) const
{
    if(not sub_shapes().empty())
    {
        std::vector<shape> subs;
        std::transform(sub_shapes().cbegin(),
                       sub_shapes().cend(),
                       std::back_inserter(subs),
                       [&](auto s) { return s.to_static(x); });
        return shape(subs);
    }
    if(not this->dynamic())
    {
        return *this;
    }
    auto static_lens = this->max_lens();
    std::transform(static_lens.begin(),
                   static_lens.end(),
                   this->dyn_dims().cbegin(),
                   static_lens.begin(),
                   [&](auto sl, const auto& dd) { return dd.is_fixed() ? sl : x; });
    return {type(), static_lens};
}

shape shape::to_static(const std::unordered_map<sym::expr, std::size_t>& symbol_map) const
{
    if(not sub_shapes().empty())
    {
        std::vector<shape> subs;
        std::transform(sub_shapes().cbegin(),
                       sub_shapes().cend(),
                       std::back_inserter(subs),
                       [&](auto s) { return s.to_static(symbol_map); });
        return shape(subs);
    }
    if(not this->dynamic())
        return *this;
    std::vector<std::size_t> static_lens(this->ndim());
    std::transform(this->dyn_dims().cbegin(),
                   this->dyn_dims().cend(),
                   static_lens.begin(),
                   [&](const auto& dd) -> std::size_t {
                       if(dd.is_fixed())
                           return dd.min;
                       if(dd.sym_expr)
                           return dd.sym_expr->eval_uint(symbol_map);
                       MIGRAPHX_THROW("to_static: non-fixed dimension has no symbolic expression");
                   });
    const auto& ds = this->dyn_strides();
    if(ds.empty())
        return {type(), static_lens};
    std::vector<std::size_t> static_strides(ds.size());
    std::transform(ds.cbegin(), ds.cend(), static_strides.begin(), [&](const auto& s) {
        return s.eval_uint(symbol_map);
    });
    return {type(), static_lens, static_strides};
}

std::size_t shape::element_space() const { return impl->element_space(); }

std::string shape::type_string() const { return name(this->type()); }

bool shape::dynamic() const { return not impl->m_dyn_dims.empty(); }

bool shape::any_of_dynamic() const
{
    if(this->dynamic())
    {
        return true;
    }
    return std::any_of(this->sub_shapes().cbegin(), this->sub_shapes().cend(), [](auto s) {
        return s.any_of_dynamic();
    });
}

bool shape::computable() const { return is_computable(this->type()); }

const std::vector<shape::dynamic_dimension>& shape::dyn_dims() const
{
    if(ndim() > 0 and not this->dynamic())
    {
        MIGRAPHX_THROW("SHAPE: dyn_dims() called on a static shape");
    }
    return impl->m_dyn_dims;
}

bool shape::symbolic() const { return not impl->m_dyn_strides.empty(); }

const std::vector<sym::expr>& shape::dyn_strides() const { return impl->m_dyn_strides; }

std::vector<std::size_t> shape::min_lens() const
{
    return this->dynamic() ? impl->min_lens() : this->lens();
}

std::vector<std::size_t> shape::max_lens() const
{
    return this->dynamic() ? impl->max_lens() : this->lens();
}

std::vector<std::set<std::size_t>> shape::opt_lens() const { return impl->opt_lens(); }

bool shape::dynamic_dimension::is_fixed() const { return this->min == this->max; }

bool shape::dynamic_dimension::has_optimal() const { return not optimals.empty(); }

shape::dynamic_dimension& shape::dynamic_dimension::operator+=(const std::size_t& x)
{
    return *this += dynamic_dimension{x, x};
}

shape::dynamic_dimension& shape::dynamic_dimension::operator-=(const std::size_t& x)
{
    return *this -= dynamic_dimension{x, x};
}

shape::dynamic_dimension& shape::dynamic_dimension::operator*=(const std::size_t& x)
{
    return *this *= dynamic_dimension{x, x};
}

shape::dynamic_dimension& shape::dynamic_dimension::operator/=(const std::size_t& x)
{
    return *this /= dynamic_dimension{x, x};
}

bool operator==(const shape::dynamic_dimension& x, const shape::dynamic_dimension& y)
{
    if(not(x.sym_expr == y.sym_expr))
        return false;
    return (x.min == y.min and x.max == y.max and
            ((x.is_fixed() and y.is_fixed()) or (x.optimals == y.optimals)));
}

bool operator!=(const shape::dynamic_dimension& x, const shape::dynamic_dimension& y)
{
    return not(x == y);
}
std::ostream& operator<<(std::ostream& os, const shape::dynamic_dimension& x)
{
    if(x.is_symbolic())
        os << x.sym_expr->to_string();
    if(x.is_fixed())
    {
        if(not x.is_symbolic())
            os << x.min;
        return os;
    }
    os << "[" << x.min << ".." << x.max << "]";
    return os;
}

bool operator==(const shape::dynamic_dimension& x, const std::size_t& y)
{
    return x.min == y and x.max == y;
}
bool operator==(const std::size_t& x, const shape::dynamic_dimension& y) { return y == x; }
bool operator!=(const shape::dynamic_dimension& x, const std::size_t& y) { return not(x == y); }
bool operator!=(const std::size_t& x, const shape::dynamic_dimension& y) { return not(x == y); }

shape::dynamic_dimension operator+(const shape::dynamic_dimension& x, const std::size_t& y)
{
    auto dd = x;
    return dd += y;
}

shape::dynamic_dimension operator+(const std::size_t& x, const shape::dynamic_dimension& y)
{
    return y + x;
}

shape::dynamic_dimension operator-(const shape::dynamic_dimension& x, const std::size_t& y)
{
    auto dd = x;
    return dd -= y;
}

shape::dynamic_dimension operator*(const shape::dynamic_dimension& x, const std::size_t& y)
{
    auto dd = x;
    return dd *= y;
}

shape::dynamic_dimension operator*(const std::size_t& x, const shape::dynamic_dimension& y)
{
    return y * x;
}

shape::dynamic_dimension operator/(const shape::dynamic_dimension& x, const std::size_t& y)
{
    auto dd = x;
    return dd /= y;
}

shape::dynamic_dimension& shape::dynamic_dimension::operator+=(const shape::dynamic_dimension& x)
{
    auto lhs_sym = sym_expr;
    auto rhs_sym = x.sym_expr;
    min          = min + x.min;
    max          = (max > std::numeric_limits<std::size_t>::max() - x.max)
                       ? std::numeric_limits<std::size_t>::max()
                       : max + x.max;
    if(x.is_fixed())
    {
        std::set<std::size_t> new_optimals;
        std::transform(optimals.begin(),
                       optimals.end(),
                       std::inserter(new_optimals, new_optimals.begin()),
                       [&](auto o) { return o + x.min; });
        optimals = new_optimals;
    }
    else
    {
        optimals.clear();
    }
    sym_expr = (lhs_sym and rhs_sym) ? optional<sym::expr>(*lhs_sym + *rhs_sym) : nullopt;
    normalize_sym();
    return *this;
}

shape::dynamic_dimension& shape::dynamic_dimension::operator-=(const shape::dynamic_dimension& x)
{
    auto lhs_sym = sym_expr;
    auto rhs_sym = x.sym_expr;
    min          = (min > x.max) ? min - x.max : 0;
    max          = (max > x.min) ? max - x.min : 0;
    if(x.is_fixed())
    {
        std::set<std::size_t> new_optimals;
        std::transform(optimals.begin(),
                       optimals.end(),
                       std::inserter(new_optimals, new_optimals.begin()),
                       [&](auto o) { return (o > x.min) ? o - x.min : 0; });
        optimals = new_optimals;
    }
    else
    {
        optimals.clear();
    }
    sym_expr = (lhs_sym and rhs_sym) ? optional<sym::expr>(*lhs_sym - *rhs_sym) : nullopt;
    normalize_sym();
    return *this;
}

shape::dynamic_dimension& shape::dynamic_dimension::operator*=(const shape::dynamic_dimension& x)
{
    auto lhs_sym = sym_expr;
    auto rhs_sym = x.sym_expr;
    min          = min * x.min;
    max          = (max > std::numeric_limits<std::size_t>::max() / (x.max == 0 ? 1 : x.max))
                       ? std::numeric_limits<std::size_t>::max()
                       : max * x.max;
    if(x.is_fixed())
    {
        std::set<std::size_t> new_optimals;
        std::transform(optimals.begin(),
                       optimals.end(),
                       std::inserter(new_optimals, new_optimals.begin()),
                       [&](auto o) { return o * x.min; });
        optimals = new_optimals;
    }
    else
    {
        optimals.clear();
    }
    sym_expr = (lhs_sym and rhs_sym) ? optional<sym::expr>(*lhs_sym * *rhs_sym) : nullopt;
    normalize_sym();
    return *this;
}

shape::dynamic_dimension& shape::dynamic_dimension::operator/=(const shape::dynamic_dimension& x)
{
    auto lhs_sym = sym_expr;
    auto rhs_sym = x.sym_expr;
    min          = (x.max == 0) ? 0 : min / x.max;
    max          = (x.min == 0) ? std::numeric_limits<std::size_t>::max() : max / x.min;
    if(x.is_fixed())
    {
        std::set<std::size_t> new_optimals;
        std::transform(optimals.begin(),
                       optimals.end(),
                       std::inserter(new_optimals, new_optimals.begin()),
                       [&](auto o) { return (x.min == 0) ? std::size_t{0} : o / x.min; });
        optimals = new_optimals;
    }
    else
    {
        optimals.clear();
    }
    sym_expr = (lhs_sym and rhs_sym) ? optional<sym::expr>(*lhs_sym / *rhs_sym) : nullopt;
    normalize_sym();
    return *this;
}

shape::dynamic_dimension operator+(const shape::dynamic_dimension& x,
                                   const shape::dynamic_dimension& y)
{
    auto result = x;
    result += y;
    return result;
}

shape::dynamic_dimension operator-(const shape::dynamic_dimension& x,
                                   const shape::dynamic_dimension& y)
{
    auto result = x;
    result -= y;
    return result;
}

shape::dynamic_dimension operator*(const shape::dynamic_dimension& x,
                                   const shape::dynamic_dimension& y)
{
    auto result = x;
    result *= y;
    return result;
}

shape::dynamic_dimension operator/(const shape::dynamic_dimension& x,
                                   const shape::dynamic_dimension& y)
{
    auto result = x;
    result /= y;
    return result;
}

bool operator==(const shape& x, const shape& y)
{
    if(x.dynamic() and y.dynamic())
    {
        return x.impl == y.impl or
               (x.type() == y.type() and x.dyn_dims() == y.dyn_dims() and
                x.dyn_strides() == y.dyn_strides() and x.sub_shapes() == y.sub_shapes());
    }
    return x.impl == y.impl or
           (x.dynamic() == y.dynamic() and x.type() == y.type() and x.lens() == y.lens() and
            x.strides() == y.strides() and x.sub_shapes() == y.sub_shapes());
}

bool operator!=(const shape& x, const shape& y) { return not(x == y); }

std::ostream& operator<<(std::ostream& os, const shape& x)
{
    if(x.sub_shapes().empty())
    {
        if(x.symbolic())
        {
            os << x.type_string() << ", {";
            const auto& dd = x.dyn_dims();
            for(std::size_t i = 0; i < dd.size(); ++i)
            {
                if(i > 0)
                    os << ", ";
                if(dd[i].is_symbolic())
                    os << dd[i];
                else
                    os << dd[i].min;
            }
            os << "}, ";
            os << "{" << to_string_range(x.dyn_strides()) << "}";
        }
        else if(x.dynamic())
        {
            os << "dynamic, ";
            os << x.type_string() << ", ";
            os << "{" << to_string_range(x.dyn_dims()) << "}";
        }
        else
        {
            os << x.type_string() << ", ";
            os << "{" << to_string_range(x.lens()) << "}, ";
            os << "{" << to_string_range(x.strides()) << "}";
        }
    }
    else
    {
        os << "[" << to_string_range(x.sub_shapes()) << "]";
    }
    return os;
}

bool shape::same_lens(const shape& x, const shape& y)
{
    return x.to_dynamic().dyn_dims() == y.to_dynamic().dyn_dims();
}

shape::type_t shape::parse_type(const std::string& s)
{
    static const std::unordered_map<std::string, shape::type_t> m = {
    // clang-format off
#define MIGRAPHX_SHAPE_GENERATE_TYPE_STRING_MAP(x, t) {#x, x}, {#t, x},
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_GENERATE_TYPE_STRING_MAP)
        {"fp4x2_type", fp4x2_type},
        {"tuple_type", tuple_type},
        {"tuple", tuple_type}};
    // clang-format on
    return m.at(s);
}

const std::vector<shape>& shape::sub_shapes() const { return impl->m_shapes; }

void shape::debug_print() const { std::cout << *this << std::endl; }

std::vector<shape> flatten(const std::vector<shape>& shapes)
{
    std::vector<shape> result;
    for(const auto& s : shapes)
    {
        if(s.type() == shape::tuple_type)
        {
            auto subs = flatten(s.sub_shapes());
            result.insert(result.end(), subs.begin(), subs.end());
        }
        else
        {
            result.push_back(s);
        }
    }
    return result;
}

std::size_t shape::tuple_size() const { return impl->m_shapes.size(); }

void migraphx_to_value(value& v, const shape& s)
{
    value result;
    result["type"]       = migraphx::to_value(s.type_string());
    result["sub_shapes"] = migraphx::to_value(s.sub_shapes());
    if(s.dynamic())
    {
        result["lens"]               = {};
        result["strides"]            = {};
        result["dynamic_dimensions"] = migraphx::to_value(s.dyn_dims());
    }
    else
    {
        result["lens"]               = migraphx::to_value(s.lens());
        result["strides"]            = migraphx::to_value(s.strides());
        result["dynamic_dimensions"] = {};
    }
    if(s.symbolic())
    {
        result["dyn_strides"] = migraphx::to_value(s.dyn_strides());
    }
    else
    {
        result["dyn_strides"] = {};
    }
    v = result;
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
        if(v.at("dynamic_dimensions").empty())
        {
            s = shape{shape::parse_type(t),
                      v.at("lens").to_vector<std::size_t>(),
                      v.at("strides").to_vector<std::size_t>()};
        }
        else
        {
            auto v_dd = v.at("dynamic_dimensions");
            std::vector<shape::dynamic_dimension> dyn_dims(v_dd.size());
            std::transform(
                v_dd.begin(), v_dd.end(), dyn_dims.begin(), [](const migraphx::value& x) {
                    return from_value<shape::dynamic_dimension>(x);
                });

            if(v.contains("dyn_strides") and not v.at("dyn_strides").empty())
            {
                auto v_ds = v.at("dyn_strides");
                std::vector<sym::expr> dstrides;
                dstrides.reserve(v_ds.size());
                std::transform(v_ds.begin(),
                               v_ds.end(),
                               std::back_inserter(dstrides),
                               [](const auto& x) { return from_value<sym::expr>(x); });
                s = shape(shape::parse_type(t), std::move(dyn_dims), std::move(dstrides));
            }
            else
            {
                s = shape{shape::parse_type(t), dyn_dims};
            }
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
