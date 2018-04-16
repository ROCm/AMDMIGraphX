#ifndef GUARD_RTGLIB_PROGRAM_HPP
#define GUARD_RTGLIB_PROGRAM_HPP

#include <list>
#include <unordered_map>
#include <rtg/instruction.hpp>
#include <rtg/operand.hpp>
#include <rtg/builtin.hpp>

namespace rtg {

struct program
{
    template<class... Ts>
    instruction * add_instruction(operand op, Ts*... args)
    {
        shape r = op.compute_shape({args->result...});
        instructions.push_back({op, r, {args...}});
        return std::addressof(instructions.back());
    }
    template<class... Ts>
    instruction * add_literal(Ts&&... xs)
    {
        instructions.emplace_back(literal{std::forward<Ts>(xs)...});
        return std::addressof(instructions.back());
    }

    instruction * add_parameter(std::string name, shape s)
    {
        instructions.push_back({builtin::param{std::move(name)}, s, {}});
        return std::addressof(instructions.back());
    }

    literal eval(std::unordered_map<std::string, argument> params) const;

private:
    // A list is used to keep references to an instruction stable
    std::list<instruction> instructions;
};

}

#endif
