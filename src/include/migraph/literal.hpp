#ifndef MIGRAPH_GUARD_MIGRAPHLIB_LITERAL_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_LITERAL_HPP

#include <migraph/shape.hpp>
#include <migraph/shape_for_each.hpp>
#include <migraph/argument.hpp>
#include <migraph/tensor_view.hpp>
#include <migraph/raw_data.hpp>

#include <memory>

namespace migraph {

/**
 * @brief Represents a raw literal
 * @details This stores the literal has a raw buffer that is owned by this class
 */
struct literal : raw_data<literal>
{
    literal() {}

    template <class T>
    literal(T x) : buffer(std::make_unique<char[]>(sizeof(T))), m_shape(shape::get_type<T>{})
    {
        static_assert(std::is_trivial<T>{}, "Literals can only be trivial types");
        *(reinterpret_cast<T*>(buffer.get())) = x;
    }

    template <class T>
    literal(shape s, const std::vector<T>& x) : buffer(std::make_unique<char[]>(s.bytes())), m_shape(s)
    {
        static_assert(std::is_trivial<T>{}, "Literals can only be trivial types");
        fill(x.begin(), x.end());
    }

    template <class T>
    literal(shape s, const std::initializer_list<T>& x) : buffer(std::make_unique<char[]>(s.bytes())), m_shape(s)
    {
        static_assert(std::is_trivial<T>{}, "Literals can only be trivial types");
        fill(x.begin(), x.end());
    }

    template <class Iterator>
    literal(shape s, Iterator start, Iterator end) : buffer(std::make_unique<char[]>(s.bytes())), m_shape(s)
    {
        fill(start, end);
    }

    literal(shape s, const char* x) : buffer(std::make_unique<char[]>(s.bytes())), m_shape(s) 
    {
        std::copy(x, x + s.bytes(), buffer.get());
    }

    /// Whether data is available
    bool empty() const { return this->buffer == nullptr; }

    /// Provides a raw pointer to the data
    const char* data() const { return this->buffer.get(); }

    const shape& get_shape() const { return this->m_shape; }

    /// Convert the data to an argument
    argument get_argument() const
    {
        std::vector<char> b(buffer.get(), buffer.get() + m_shape.bytes());
        return {m_shape, [b]() mutable { return b.data(); }};
    }

    private:
    std::shared_ptr<char> buffer;
    shape m_shape;

    template <class Iterator>
    void fill(Iterator start, Iterator end)
    {
        if(m_shape.standard())
        {
            m_shape.visit_type([&](auto as) { std::copy(start, end, as.from(buffer.get())); });
        }
        else
        {
            auto it = start;
            m_shape.visit_type([&](auto as) {
                auto output = make_view(m_shape, as.from(buffer.get()));
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
