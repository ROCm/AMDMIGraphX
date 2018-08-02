#ifndef MIGRAPH_GUARD_MIGRAPHLIB_LITERAL_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_LITERAL_HPP

#include <migraph/shape.hpp>
#include <migraph/shape_for_each.hpp>
#include <migraph/argument.hpp>
#include <migraph/tensor_view.hpp>
#include <migraph/raw_data.hpp>

namespace migraph {

/**
 * @brief Represents a raw literal
 * @details This stores the literal has a raw buffer that is owned by this class
 */
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
        static_assert(std::is_trivial<T>{}, "Literals can only be trivial types");
        fill(x.begin(), x.end());
    }

    template <class T>
    literal(shape s, const std::initializer_list<T>& x) : buffer(s.bytes(), 0), m_shape(s)
    {
        static_assert(std::is_trivial<T>{}, "Literals can only be trivial types");
        fill(x.begin(), x.end());
    }

    template <class Iterator>
    literal(shape s, Iterator start, Iterator end) : buffer(s.bytes(), 0), m_shape(s)
    {
        fill(start, end);
    }

    literal(shape s, const char* x) : buffer(x, x + s.bytes()), m_shape(s) {}

    /// Whether data is available
    bool empty() const { return this->buffer.empty(); }

    /// Provides a raw pointer to the data
    const char* data() const { return this->buffer.data(); }

    const shape& get_shape() const { return this->m_shape; }

    /// Convert the data to an argument
    argument get_argument() const
    {
        auto b = buffer;
        return {m_shape, [b]() mutable { return b.data(); }};
    }

    private:
    std::vector<char> buffer;
    shape m_shape;

    template <class Iterator>
    void fill(Iterator start, Iterator end)
    {
        if(m_shape.standard())
        {
            m_shape.visit_type([&](auto as) { std::copy(start, end, as.from(buffer.data())); });
        }
        else
        {
            auto it = start;
            m_shape.visit_type([&](auto as) {
                auto output = make_view(m_shape, as.from(buffer.data()));
                shape_for_each(output.get_shape(), [&](const auto& idx) {
                    it++;
                    output(idx.begin(), idx.end()) = *it;
                });
            });
        }
    }
};

} // namespace migraph

#endif
