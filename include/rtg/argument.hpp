#ifndef GUARD_RTGLIB_ARGUMENT_HPP
#define GUARD_RTGLIB_ARGUMENT_HPP

#include <rtg/shape.hpp>
#include <rtg/raw_data.hpp>
#include <functional>

namespace rtg {

struct argument : raw_data<argument>
{
    argument() {}

    argument(shape s, std::function<char*()> d) : data(d), m_shape(s) {}

    std::function<char*()> data;

    bool empty() const { return not data; }

    const shape& get_shape() const { return this->m_shape; }

    private:
    shape m_shape;
};

} // namespace rtg

#endif
