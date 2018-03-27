#ifndef GUARD_RTGLIB_SHAPE_HPP
#define GUARD_RTGLIB_SHAPE_HPP

#include <vector>
#include <cassert>

namespace rtg {

struct shape
{
    enum type_t
    {
        float_type,
        int_type
    };

    shape();
    shape(type_t t);
    shape(type_t t, std::vector<std::size_t> l);
    shape(type_t t, std::vector<std::size_t> l, std::vector<std::size_t> s);


    type_t type() const;
    const std::vector<std::size_t> lens() const;
    const std::vector<std::size_t> strides() const;
    std::size_t elements() const;
    std::size_t bytes() const;

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

        T operator()() const
        {
            return {};
        }

        std::size_t size(std::size_t n=0) const
        {
            return sizeof(T)*n;
        }

        template<class U>
        T& from(U* buffer, std::size_t n=0) const
        {
            return *(reinterpret_cast<T*>(buffer)+n);
        }
    };

    template<class Visitor>
    void visit_type(Visitor v) const
    {
        switch(this->type_) 
        {
            case float_type:
                v(as<float>());
                return;
            case int_type:
                v(as<int>());
                return;
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
