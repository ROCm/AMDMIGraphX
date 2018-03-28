#ifndef GUARD_RTGLIB_LITERAL_HPP
#define GUARD_RTGLIB_LITERAL_HPP

#include <rtg/shape.hpp>
#include <rtg/argument.hpp>

namespace rtg {

struct literal 
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
        static_assert(std::is_trivial<T>{}, "Literals can only be trivial types");
        std::copy(x.begin(), x.end(), reinterpret_cast<T*>(buffer.data()));
    }
    
    literal(shape s, const char* x)
    : buffer(x, x+s.bytes()), shape_(s)
    {}

    friend bool operator==(const literal& x, const literal& y)
    {
        bool result = x.buffer.empty() && y.buffer.empty();
        if(not result && x.shape_ == y.shape_ and x.buffer.size() == y.buffer.size())
        {
            x.shape_.visit_type([&](auto as) {
                auto space = x.shape_.bytes() / sizeof(as());
                auto * xstart = &as.from(x.buffer.data());
                auto * ystart = &as.from(y.buffer.data());
                result = std::equal(xstart, xstart+space, ystart, ystart+space);

            });
        }
        return result;
    }

    friend bool operator!=(const literal& x, const literal& y)
    {
        return !(x == y);
    }

    template<class Visitor>
    void visit(Visitor v, std::size_t n=0) const
    {
        shape_.visit_type([&](auto as) {
            v(as.from(this->buffer.data(), n));
        });
    }

    bool empty() const
    {
        return this->buffer.empty();
    }

    template<class T>
    T at(std::size_t n=0) const
    {
        T result;
        this->visit([&](auto x) {
            result = x;
        });
        return result;
    }

    const shape& get_shape() const
    {
        return this->shape_;
    }

    argument get_argument() const
    {
        argument arg;
        auto b = buffer;
        arg.s = shape_;
        arg.data = [b]() mutable { return b.data(); };
        return arg;
    }

private:
    std::vector<char> buffer;
    shape shape_;
};

}

#endif
