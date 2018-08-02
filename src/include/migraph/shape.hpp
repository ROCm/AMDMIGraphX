#ifndef MIGRAPH_GUARD_MIGRAPHLIB_SHAPE_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_SHAPE_HPP

#include <vector>
#include <cassert>
#include <ostream>
#include <numeric>

#include <migraph/errors.hpp>

namespace migraph {

struct shape
{

// Add new types here
// clang-format off
#define MIGRAPH_SHAPE_VISIT_TYPES(m) \
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

#define MIGRAPH_SHAPE_ENUM_TYPES(x, t) x,
    enum type_t
    {
        MIGRAPH_SHAPE_VISIT_TYPES(MIGRAPH_SHAPE_ENUM_TYPES)
    };
#undef MIGRAPH_SHAPE_ENUM_TYPES

    template <class T, class = void>
    struct get_type;
#define MIGRAPH_SHAPE_GET_TYPE(x, t)                          \
    template <class T>                                        \
    struct get_type<t, T> : std::integral_constant<type_t, x> \
    {                                                         \
    };
    MIGRAPH_SHAPE_VISIT_TYPES(MIGRAPH_SHAPE_GET_TYPE)
#undef MIGRAPH_SHAPE_GET_TYPE

    template <class T>
    struct get_type<const T> : get_type<T>
    {
    };

    shape();
    shape(type_t t);
    shape(type_t t, std::vector<std::size_t> l);
    shape(type_t t, std::vector<std::size_t> l, std::vector<std::size_t> s);

    type_t type() const;
    const std::vector<std::size_t>& lens() const;
    const std::vector<std::size_t>& strides() const;
    std::size_t elements() const;
    std::size_t bytes() const;

    std::size_t index(std::initializer_list<std::size_t> l) const;
    std::size_t index(const std::vector<std::size_t>& l) const;

    template <class Iterator>
    std::size_t index(Iterator start, Iterator last) const
    {
        assert(std::distance(start, last) <= this->lens().size());
        assert(this->lens().size() == this->strides().size());
        return std::inner_product(start, last, this->strides().begin(), std::size_t{0});
    }

    // Map element index to space index
    std::size_t index(std::size_t i) const;

    bool packed() const;
    bool transposed() const;
    bool broadcasted() const;
    bool standard() const;

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
    };

    template <class Visitor>
    void visit_type(Visitor v) const
    {
        switch(this->m_type)
        {
#define MIGRAPH_SHAPE_VISITOR_CASE(x, t) \
    case x: v(as<t>()); return;
            MIGRAPH_SHAPE_VISIT_TYPES(MIGRAPH_SHAPE_VISITOR_CASE)
#undef MIGRAPH_SHAPE_VISITOR_CASE
        }
        MIGRAPH_THROW("Unknown type");
    }

    private:
    type_t m_type;
    std::vector<std::size_t> m_lens;
    std::vector<std::size_t> m_strides;
    bool m_standard;

    void calculate_strides();
    std::size_t element_space() const;
    std::string type_string() const;
};

} // namespace migraph

#endif
