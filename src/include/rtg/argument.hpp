#ifndef RTG_GUARD_RTGLIB_ARGUMENT_HPP
#define RTG_GUARD_RTGLIB_ARGUMENT_HPP

#include <rtg/shape.hpp>
#include <rtg/raw_data.hpp>
#include <functional>

namespace rtg {

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

    /// Provides a raw pointer to the data
    std::function<char*()> data;

    /// Whether data is available
    bool empty() const { return not data; }

    const shape& get_shape() const { return this->m_shape; }

    template <class T>
    T* cast() const
    {
        return reinterpret_cast<T*>(this->data());
    }

    private:
    shape m_shape;
};

} // namespace rtg

#endif
