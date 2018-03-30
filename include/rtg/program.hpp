#ifndef GUARD_RTGLIB_PROGRAM_HPP
#define GUARD_RTGLIB_PROGRAM_HPP

#include <list>
#include <unordered_map>
#include <rtg/instruction.hpp>
#include <rtg/operand.hpp>

namespace rtg {

struct program
{
    template<class... Ts>
    instruction * add_instruction(std::string name, Ts*... args)
    {
        auto&& op = ops.at(name);
        shape r = op.compute_shape({args->result...});
        instructions.push_back({name, r, {args...}});
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
        instructions.push_back({"param:"+std::move(name), s, {}});
        return std::addressof(instructions.back());
    }

    template<class Op, class Shape>
    void add_operator(std::string name, Op op, Shape s)
    {
        operand result;
        result.name = name;
        result.compute = op;
        result.compute_shape = s;
        ops.emplace(name, result);
    }

    literal eval(std::unordered_map<std::string, argument> params) const;

private:
    // A list is used to keep references to an instruction stable
    std::list<instruction> instructions;

    std::unordered_map<std::string, operand> ops;

};

}

#endif
