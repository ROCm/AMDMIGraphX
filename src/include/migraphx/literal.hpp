#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_LITERAL_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_LITERAL_HPP

#include <migraphx/shape.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/tensor_view.hpp>
#include <migraphx/raw_data.hpp>
#include <migraphx/make_shared_array.hpp>
#include <migraphx/config.hpp>

#include <memory>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * @brief Represents a raw literal
 * @details This stores the literal has a raw buffer that is owned by this class
 */
struct literal : raw_data<literal>
{
    literal() {}

    template <class U, class T = deduce<U>, shape::type_t ShapeType = shape::get_type<T>{}>
    literal(U x) : buffer(make_shared_array<char>(sizeof(T))), m_shape(ShapeType)
    {
        static_assert(std::is_trivially_copyable<T>{}, "Literals can only be trivial types");
        *(reinterpret_cast<T*>(buffer.get())) = x;
    }

    template <class T>
    literal(const shape& s, const std::vector<T>& x)
        : buffer(make_shared_array<char>(s.bytes())), m_shape(s)
    {
        static_assert(std::is_trivially_copyable<T>{}, "Literals can only be trivial types");
        fill(x.begin(), x.end());
    }

    template <class T>
    literal(const shape& s, const std::initializer_list<T>& x)
        : buffer(make_shared_array<char>(s.bytes())), m_shape(s)
    {
        static_assert(std::is_trivially_copyable<T>{}, "Literals can only be trivial types");
        fill(x.begin(), x.end());
    }

    template <class Iterator>
    literal(const shape& s, Iterator start, Iterator end)
        : buffer(make_shared_array<char>(s.bytes())), m_shape(s)
    {
        fill(start, end);
    }

    literal(const shape& s, const char* x) : buffer(make_shared_array<char>(s.bytes())), m_shape(s)
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
        assert(std::distance(start, end) == m_shape.elements());
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
                    output(idx.begin(), idx.end()) = *it;
                    it++;
                });
            });
        }
    }
};

template <class F>
literal transform(literal l, F f)
{
    literal result;
    l.visit([&](auto x) {
        using type = std::remove_cv_t<typename decltype(x)::value_type>;
        std::vector<type> output(x.size(), type(0));
        std::transform(x.begin(), x.end(), output.begin(), f);
        result = literal{l.get_shape(), output};
    });
    return result;
}

template <class F>
literal transform(literal l1, literal l2, F f)
{
    assert(l1.get_shape() == l2.get_shape());
    literal result;
    visit_all(l1, l2)([&](auto x, auto y) {
        using type = std::remove_cv_t<typename decltype(x)::value_type>;
        std::vector<type> output(x.size(), type(0));
        std::transform(x.begin(), x.end(), y.begin(), output.begin(), f);
        result = literal{l1.get_shape(), output};
    });
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
