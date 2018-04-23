#ifndef GUARD_RTGLIB_PROGRAM_HPP
#define GUARD_RTGLIB_PROGRAM_HPP

#include <list>
#include <unordered_map>
#include <rtg/instruction.hpp>
#include <rtg/operand.hpp>
#include <rtg/builtin.hpp>
#include <algorithm>

namespace rtg {

struct program
{
    // TODO: A program should be copyable
    program()               = default;
    program(const program&) = delete;
    program& operator=(const program&) = delete;

    template <class... Ts>
    instruction* add_instruction(operand op, Ts*... args)
    {
        shape r = op.compute_shape({args->result...});
        instructions.push_back({op, r, {args...}});
        return std::addressof(instructions.back());
    }
    instruction* add_instruction(operand op, std::vector<instruction*> args)
    {
        assert(std::all_of(
                   args.begin(), args.end(), [&](instruction* x) { return has_instruction(x); }) &&
               "Argument is not an exisiting instruction");
        std::vector<shape> shapes(args.size());
        std::transform(
            args.begin(), args.end(), shapes.begin(), [](instruction* ins) { return ins->result; });
        shape r = op.compute_shape(shapes);
        instructions.push_back({op, r, args});
        assert(instructions.back().arguments == args);
        return std::addressof(instructions.back());
    }
    template <class... Ts>
    instruction* add_literal(Ts&&... xs)
    {
        instructions.emplace_back(literal{std::forward<Ts>(xs)...});
        return std::addressof(instructions.back());
    }

    instruction* add_parameter(std::string name, shape s)
    {
        instructions.push_back({builtin::param{std::move(name)}, s, {}});
        return std::addressof(instructions.back());
    }

    literal eval(std::unordered_map<std::string, argument> params) const;

    // TODO: Change to stream operator
    void print() const;

    bool has_instruction(const instruction* ins) const
    {
        return std::find_if(instructions.begin(), instructions.end(), [&](const instruction& x) {
                   return ins == std::addressof(x);
               }) != instructions.end();
    }

    private:
    // A list is used to keep references to an instruction stable
    std::list<instruction> instructions;
};

} // namespace rtg

#endif
