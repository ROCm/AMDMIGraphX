#ifndef GUARD_RTGLIB_INSTRUCTION_HPP
#define GUARD_RTGLIB_INSTRUCTION_HPP

#include <rtg/literal.hpp>
#include <rtg/shape.hpp>
#include <string>

namespace rtg {

struct instruction
{
    instruction() {}

    instruction(std::string n, shape r, std::vector<instruction*> args)
    : name(std::move(n)), result(std::move(r)), arguments(std::move(args))
    {}

    instruction(literal l)
    : name("literal"), result(l.get_shape()), lit(std::move(l))
    {}

    std::string name;
    shape result;
    std::vector<instruction*> arguments;
    literal lit;
};

}

#endif
