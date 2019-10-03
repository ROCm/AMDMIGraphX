#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_ARGUMENT_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_ARGUMENT_HPP

#include <migraphx/shape.hpp>
#include <migraphx/raw_data.hpp>
#include <migraphx/config.hpp>
#include <functional>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * @brief Arguments passed to instructions
 *
 * An `argument` can represent a raw buffer of data that either be referenced from another element
 * or it can be owned by the argument.
 *
 */
struct argument : raw_data<argument>
{
    argument() {}

    argument(const shape& s) : m_shape(s)
    {
        std::vector<char> buffer(s.bytes());
        // TODO: Move vector
        data = [=]() mutable { return buffer.data(); };
    }

    argument(shape s, std::function<char*()> d) : data(std::move(d)), m_shape(std::move(s)) {}
    template <class T>
    argument(shape s, T* d)
        : data([d] { return reinterpret_cast<char*>(d); }), m_shape(std::move(s))
    {
    }

    /// Provides a raw pointer to the data
    std::function<char*()> data = nullptr;

    /// Whether data is available
    bool empty() const { return not data; }

    const shape& get_shape() const { return this->m_shape; }

    argument reshape(const shape& s) const
    {
        argument self = *this;
        return {s, [=]() mutable { return self.data(); }};
    }

    private:
    shape m_shape;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
