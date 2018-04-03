#ifndef GUARD_RTGLIB_LITERAL_HPP
#define GUARD_RTGLIB_LITERAL_HPP

#include <rtg/shape.hpp>
#include <rtg/argument.hpp>
#include <rtg/tensor_view.hpp>
#include <rtg/raw_data.hpp>

namespace rtg {

struct literal : raw_data<literal>
{
    literal()
    : buffer(), shape_()
    {}

    template<class T>
    literal(T x) 
    : buffer(sizeof(T), 0), shape_(shape::get_type<T>{})
    {
        static_assert(std::is_trivial<T>{}, "Literals can only be trivial types");
        *(reinterpret_cast<T*>(buffer.data())) = x;
    }

    template<class T>
    literal(shape s, const std::vector<T>& x) 
    : buffer(s.bytes(), 0), shape_(s)
    {
        assert(s.packed());
        static_assert(std::is_trivial<T>{}, "Literals can only be trivial types");
        std::copy(x.begin(), x.end(), reinterpret_cast<T*>(buffer.data()));
    }

    template<class T>
    literal(shape s, const std::initializer_list<T>& x) 
    : buffer(s.bytes(), 0), shape_(s)
    {
        assert(s.packed());
        static_assert(std::is_trivial<T>{}, "Literals can only be trivial types");
        std::copy(x.begin(), x.end(), reinterpret_cast<T*>(buffer.data()));
    }
    
    literal(shape s, const char* x)
    : buffer(x, x+s.bytes()), shape_(s)
    {}

    bool empty() const
    {
        return this->buffer.empty();
    }

    const char* data() const
    {
        return this->buffer.data();
    }

    const shape& get_shape() const
    {
        return this->shape_;
    }

    argument get_argument() const
    {
        auto b = buffer;
        return {shape_, [b]() mutable { return b.data(); }};
    }

private:
    std::vector<char> buffer;
    shape shape_;
};

}

#endif
