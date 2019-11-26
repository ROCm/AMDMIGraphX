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

struct shape_impl;

struct shape
{

// Add new types here
// clang-format off
#define MIGRAPHX_SHAPE_VISIT_TYPES(m) \
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
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_GENERATE_ENUM_TYPES)
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
        return std::inner_product(start, last, this->strides().begin(), std::size_t{0});
    }

    /// Map element index to space index
    std::size_t index(std::size_t i) const;

    std::vector<std::size_t> multi(std::size_t i) const;

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

    friend bool operator==(const shape& x, const shape& y);
    friend bool operator!=(const shape& x, const shape& y);
    friend std::ostream& operator<<(std::ostream& os, const shape& x);

    template <class T>
    struct as
    {
        using type = T;

        template <class U>
        T operator()(U u) const
        {
            return T(u);
        }

        template <class U>
        T* operator()(U* u) const
        {
            return static_cast<T*>(u);
        }

        template <class U>
        const T* operator()(const U* u) const
        {
            return static_cast<T*>(u);
        }

        T operator()() const { return {}; }

        std::size_t size(std::size_t n = 1) const { return sizeof(T) * n; }

        template <class U>
        T* from(U* buffer, std::size_t n = 0) const
        {
            return reinterpret_cast<T*>(buffer) + n;
        }

        template <class U>
        const T* from(const U* buffer, std::size_t n = 0) const
        {
            return reinterpret_cast<const T*>(buffer) + n;
        }

        type_t type_enum() const { return get_type<T>{}; }
    };

    template <class Visitor>
    void visit_type(Visitor v) const
    {
        switch(this->type())
        {
#define MIGRAPHX_SHAPE_GENERATE_VISITOR_CASE(x, t) \
    case x: v(as<t>()); return;
            MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_GENERATE_VISITOR_CASE)
#undef MIGRAPHX_SHAPE_GENERATE_VISITOR_CASE
        }
        MIGRAPHX_THROW("Unknown type");
    }

    template <class Visitor>
    static void visit_types(Visitor v)
    {
#define MIGRAPHX_SHAPE_GENERATE_VISITOR_ALL(x, t) v(as<t>());
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_GENERATE_VISITOR_ALL)
#undef MIGRAPHX_SHAPE_GENERATE_VISITOR_ALL
    }

    std::string type_string() const;
    private:
    std::shared_ptr<const shape_impl> impl;

    std::size_t element_space() const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
