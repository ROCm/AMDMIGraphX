#ifndef GUARD_RTGLIB_INSTRUCTION_HPP
#define GUARD_RTGLIB_INSTRUCTION_HPP

#include <rtg/operand.hpp>
#include <rtg/shape.hpp>

namespace rtg {

struct instruction
{
    unsigned int id;
    std::string name;
    shape result;
    std::vector<instruction*> arguments;
};

}

#endif
