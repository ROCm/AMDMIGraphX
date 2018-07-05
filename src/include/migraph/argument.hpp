#ifndef MIGRAPH_GUARD_MIGRAPHLIB_ARGUMENT_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_ARGUMENT_HPP

#include <migraph/shape.hpp>
#include <migraph/raw_data.hpp>
#include <functional>

namespace migraph {

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

    argument(shape s) : m_shape(s)
    {
        std::vector<char> buffer(s.bytes());
        // TODO: Move vector
        data = [=]() mutable { return buffer.data(); };
    }

    argument(shape s, std::function<char*()> d) : data(d), m_shape(s) {}
    template <class T>
    argument(shape s, T* d) : data([d] { return reinterpret_cast<char*>(d); }), m_shape(s)
    {
    }

    /// Provides a raw pointer to the data
    std::function<char*()> data;

    /// Whether data is available
    bool empty() const { return not data; }

    const shape& get_shape() const { return this->m_shape; }

    private:
    shape m_shape;
};

} // namespace migraph

#endif
