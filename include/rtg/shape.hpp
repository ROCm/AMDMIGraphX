#ifndef GUARD_RTGLIB_SHAPE_HPP
#define GUARD_RTGLIB_SHAPE_HPP

#include <vector>
#include <cassert>


namespace rtg {

struct shape
{

// Add new types here
#define RTG_SHAPE_VISIT_TYPES(m) \
    m(float_type, float) \
    m(int_type, int) \

#define RTG_SHAPE_ENUM_TYPES(x, t) x,
    enum type_t
    {
        RTG_SHAPE_VISIT_TYPES(RTG_SHAPE_ENUM_TYPES)
    };
#undef RTG_SHAPE_ENUM_TYPES

    template<class T, class=void>
    struct get_type;
#define RTG_SHAPE_GET_TYPE(x, t) \
    template<class T> \
    struct get_type<t, T> : std::integral_constant<type_t, x> \
    {};
    RTG_SHAPE_VISIT_TYPES(RTG_SHAPE_GET_TYPE)
#undef RTG_SHAPE_GET_TYPE

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

    friend bool operator==(const shape& x, const shape& y);
    friend bool operator!=(const shape& x, const shape& y);

    template<class T>
    struct as
    {
        using type = T;

        template<class U>
        T operator()(U u) const
        {
            return T(u);
        }

        template<class U>
        T* operator()(U* u) const
        {
            return static_cast<T*>(u);
        }

        template<class U>
        const T* operator()(const U* u) const
        {
            return static_cast<T*>(u);
        }

        T operator()() const
        {
            return {};
        }

        std::size_t size(std::size_t n=1) const
        {
            return sizeof(T)*n;
        }

        template<class U>
        T& from(U* buffer, std::size_t n=0) const
        {
            return *(reinterpret_cast<T*>(buffer)+n);
        }

        template<class U>
        const T& from(const U* buffer, std::size_t n=0) const
        {
            return *(reinterpret_cast<const T*>(buffer)+n);
        }
    };

    template<class Visitor>
    void visit_type(Visitor v) const
    {
        switch(this->type_) 
        {
#define RTG_SHAPE_VISITOR_CASE(x, t) \
            case x: \
                v(as<t>()); \
                return;
            RTG_SHAPE_VISIT_TYPES(RTG_SHAPE_VISITOR_CASE)
#undef RTG_SHAPE_VISITOR_CASE
        }
        assert(true);
    }
private:
    type_t type_;
    std::vector<std::size_t> lens_;
    std::vector<std::size_t> strides_;

    void calculate_strides();
    std::size_t element_space() const;
};

}

#endif
