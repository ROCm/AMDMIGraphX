#ifndef GUARD_RTGLIB_OPERAND_HPP
#define GUARD_RTGLIB_OPERAND_HPP

#include <string>
#include <functional>
#include <rtg/shape.hpp>
#include <rtg/argument.hpp>

namespace rtg {

struct operand 
{
    std::string name;
    std::function<shape(std::vector<shape>)> compute_shape;
    std::function<argument(std::vector<argument>)> compute;
};

}

#endif
