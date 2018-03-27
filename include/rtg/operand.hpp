#ifndef GUARD_RTGLIB_OPERAND_HPP
#define GUARD_RTGLIB_OPERAND_HPP

#include <functional>
#include <rtg/shape.hpp>

namespace rtg {

struct argument
{
    void* data;
    shape s;
};

struct operand 
{
    std::string name;
    std::function<shape(std::vector<shape>)> compute_shape;
    std::function<argument(std::vector<argument>)> compute;
};

}

#endif
