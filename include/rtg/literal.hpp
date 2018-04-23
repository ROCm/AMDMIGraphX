#ifndef GUARD_RTGLIB_LITERAL_HPP
#define GUARD_RTGLIB_LITERAL_HPP

#include <rtg/shape.hpp>
#include <rtg/argument.hpp>
#include <rtg/tensor_view.hpp>
#include <rtg/raw_data.hpp>

namespace rtg {

struct literal : raw_data<literal>
{
    literal() {}

    template <class T>
    literal(T x) : buffer(sizeof(T), 0), m_shape(shape::get_type<T>{})
    {
        static_assert(std::is_trivial<T>{}, "Literals can only be trivial types");
        *(reinterpret_cast<T*>(buffer.data())) = x;
    }

    template <class T>
    literal(shape s, const std::vector<T>& x) : buffer(s.bytes(), 0), m_shape(s)
    {
        assert(s.packed());
        static_assert(std::is_trivial<T>{}, "Literals can only be trivial types");
        s.visit_type([&](auto as) { std::copy(x.begin(), x.end(), as.from(buffer.data())); });
    }

    template <class T>
    literal(shape s, const std::initializer_list<T>& x) : buffer(s.bytes(), 0), m_shape(s)
    {
        assert(s.packed());
        static_assert(std::is_trivial<T>{}, "Literals can only be trivial types");
        s.visit_type([&](auto as) { std::copy(x.begin(), x.end(), as.from(buffer.data())); });
    }

    template <class Iterator>
    literal(shape s, Iterator start, Iterator end) : buffer(s.bytes(), 0), m_shape(s)
    {
        assert(s.packed());
        s.visit_type([&](auto as) { std::copy(start, end, as.from(buffer.data())); });
    }

    literal(shape s, const char* x) : buffer(x, x + s.bytes()), m_shape(s) {}

    bool empty() const { return this->buffer.empty(); }

    const char* data() const { return this->buffer.data(); }

    const shape& get_shape() const { return this->m_shape; }

    argument get_argument() const
    {
        auto b = buffer;
        return {m_shape, [b]() mutable { return b.data(); }};
    }

    private:
    std::vector<char> buffer;
    shape m_shape;
};

} // namespace rtg

#endif
