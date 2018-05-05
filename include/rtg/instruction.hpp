#ifndef RTG_GUARD_RTGLIB_INSTRUCTION_HPP
#define RTG_GUARD_RTGLIB_INSTRUCTION_HPP

#include <rtg/literal.hpp>
#include <rtg/shape.hpp>
#include <rtg/builtin.hpp>
#include <rtg/instruction_ref.hpp>
#include <string>

namespace rtg {

struct instruction
{
    instruction() {}

    instruction(operation o, shape r, std::vector<instruction_ref> args)
        : op(std::move(o)), result(std::move(r)), arguments(std::move(args))
    {
    }

    instruction(literal l) : op(builtin::literal{}), result(l.get_shape()), lit(std::move(l)) {}

    operation op;
    shape result;
    std::vector<instruction_ref> output;
    std::vector<instruction_ref> arguments;
    literal lit;
};

} // namespace rtg

#endif
