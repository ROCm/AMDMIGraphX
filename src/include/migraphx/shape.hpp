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
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_SHAPE_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_SHAPE_HPP

#include <vector>
#include <cassert>
#include <ostream>
#include <numeric>
#include <memory>

#include <migraphx/errors.hpp>
#include <migraphx/half.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct value;
struct shape_impl;

struct shape
{

// Add new types here
// clang-format off
#define MIGRAPHX_SHAPE_VISIT_TYPES(m) \
    m(bool_type, bool) \
    m(half_type, half) \
    m(float_type, float) \
    m(double_type, double) \
    m(uint8_type, uint8_t) \
    m(int8_type, int8_t) \
    m(uint16_type, uint16_t) \
    m(int16_type, int16_t) \
    m(int32_type, int32_t) \
    m(int64_type, int64_t) \
    m(uint32_type, uint32_t) \
    m(uint64_type, uint64_t)
    // clang-format on

#define MIGRAPHX_SHAPE_GENERATE_ENUM_TYPES(x, t) x,
    enum type_t
    {
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_GENERATE_ENUM_TYPES) tuple_type
    };
#undef MIGRAPHX_SHAPE_GENERATE_ENUM_TYPES

    template <class T, class = void>
    struct get_type;
#define MIGRAPHX_SHAPE_GENERATE_GET_TYPE(x, t)                \
    template <class T>                                        \
    struct get_type<t, T> : std::integral_constant<type_t, x> \
    {                                                         \
    };
    MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_GENERATE_GET_TYPE)
#undef MIGRAPHX_SHAPE_GENERATE_GET_TYPE

    template <class T>
    struct get_type<const T> : get_type<T>
    {
    };

    static const std::vector<type_t>& types();

    static std::string name(type_t t);
    static std::string cpp_type(type_t t);

    shape();
    shape(type_t t);
    shape(type_t t, std::vector<std::size_t> l);
    shape(type_t t, std::vector<std::size_t> l, std::vector<std::size_t> s);

    template <class Range>
    shape(type_t t, const Range& l) : shape(t, std::vector<std::size_t>(l.begin(), l.end()))
    {
    }

    template <class Range1, class Range2>
    shape(type_t t, const Range1& l, const Range2& s)
        : shape(t,
                std::vector<std::size_t>(l.begin(), l.end()),
                std::vector<std::size_t>(s.begin(), s.end()))
    {
    }

    shape(const std::vector<shape>& subs);

    static shape
    from_permutation(type_t t, const std::vector<std::size_t>& l, const std::vector<int64_t>& perm);
    type_t type() const;
    const std::vector<std::size_t>& lens() const;
    const std::vector<std::size_t>& strides() const;
    std::size_t elements() const;
    std::size_t bytes() const;
    std::size_t type_size() const;

    /// Map multiple indices to space index
    std::size_t index(std::initializer_list<std::size_t> l) const;
    /// Map multiple indices to space index
    std::size_t index(const std::vector<std::size_t>& l) const;

    /// Map multiple indices from a range of iterator to a space index
    template <class Iterator>
    std::size_t index(Iterator start, Iterator last) const
    {
        assert(std::distance(start, last) <= this->lens().size());
        assert(this->lens().size() == this->strides().size());
        return std::inner_product(start, last, this->strides().begin(), std::size_t{0}); // NOLINT
    }

    /// Map element index to space index
    std::size_t index(std::size_t i) const;

    std::vector<std::size_t> multi(std::size_t i) const;
    void multi_copy(std::size_t i, std::size_t* start, const std::size_t* end) const;

    /// Returns true if the shape is packed with no padding
    bool packed() const;
    /// Returns true is the shape has been transposed. That is the strides are not in descending
    /// order
    bool transposed() const;
    /// Returns true if the shape is broadcasting a dimension. That is, one of the strides are zero
    bool broadcasted() const;
    /// Returns true if the shape is in its standard format. That is, the shape is both packed and
    /// not transposed.
    bool standard() const;
    /// Returns true if all strides are equal to 0 (scalar tensor)
    bool scalar() const;

    shape normalize_standard() const;

    shape with_lens(type_t t, const std::vector<std::size_t>& l) const;
    shape with_lens(const std::vector<std::size_t>& l) const;

    shape with_type(type_t t) const;

    friend bool operator==(const shape& x, const shape& y);
    friend bool operator!=(const shape& x, const shape& y);
    friend std::ostream& operator<<(std::ostream& os, const shape& x);

    template <class T>
    struct as
    {
        using type = std::conditional_t<std::is_same<T, bool>{}, int8_t, T>;

        type max() const { return std::numeric_limits<type>::max(); }

        type min() const { return std::numeric_limits<type>::lowest(); }

        template <class U>
        type operator()(U u) const
        {
            return type(u);
        }

        template <class U>
        type* operator()(U* u) const
        {
            return static_cast<type*>(u);
        }

        template <class U>
        const type* operator()(const U* u) const
        {
            return static_cast<type*>(u);
        }

        type operator()() const { return {}; }

        std::size_t size(std::size_t n = 1) const { return sizeof(type) * n; }

        auto is_integral() const { return std::is_integral<type>{}; }
        auto is_signed() const { return std::is_signed<type>{}; }
        auto is_unsigned() const { return std::is_unsigned<type>{}; }

        template <class U>
        type* from(U* buffer, std::size_t n = 0) const
        {
            return reinterpret_cast<type*>(buffer) + n;
        }

        template <class U>
        const type* from(const U* buffer, std::size_t n = 0) const
        {
            return reinterpret_cast<const type*>(buffer) + n;
        }

        type_t type_enum() const { return get_type<type>{}; }
    };

    template <class Visitor, class TupleVisitor>
    static void visit(type_t t, Visitor v, TupleVisitor tv)
    {
        switch(t)
        {
        case tuple_type: {
            tv();
            return;
        }
#define MIGRAPHX_SHAPE_GENERATE_VISITOR_CASE(x, t) \
    case x: v(as<t>()); return;
            MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_GENERATE_VISITOR_CASE)
#undef MIGRAPHX_SHAPE_GENERATE_VISITOR_CASE
        }
        MIGRAPHX_THROW("Unknown type");
    }

    template <class Visitor>
    static void visit(type_t t, Visitor v)
    {
        return visit(t, v, [] { MIGRAPHX_THROW("Tuple cannot be visited."); });
    }

    template <class... Visitors>
    void visit_type(Visitors... vs) const
    {
        visit(this->type(), vs...);
    }

    template <class Visitor>
    static void visit_types(Visitor v)
    {
#define MIGRAPHX_SHAPE_GENERATE_VISITOR_ALL(x, t) v(as<t>());
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_GENERATE_VISITOR_ALL)
#undef MIGRAPHX_SHAPE_GENERATE_VISITOR_ALL
    }

    std::string type_string() const;
    static type_t parse_type(const std::string& s);

    const std::vector<shape>& sub_shapes() const;

    std::size_t element_space() const;

    private:
    shape(std::shared_ptr<shape_impl> pimpl);
    std::shared_ptr<const shape_impl> impl;
};

void migraphx_to_value(value& v, const shape& s);
void migraphx_from_value(const value& v, shape& s);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
