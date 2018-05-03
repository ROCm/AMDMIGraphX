#ifndef RTG_GUARD_RTGLIB_PROGRAM_HPP
#define RTG_GUARD_RTGLIB_PROGRAM_HPP

#include <list>
#include <unordered_map>
#include <rtg/operation.hpp>
#include <rtg/literal.hpp>
#include <rtg/builtin.hpp>
#include <algorithm>

namespace rtg {

struct instruction;
struct program_impl;

struct program
{
    program();
    program(program&&) noexcept;
    program& operator=(program&&) noexcept;
    ~program() noexcept;

    template <class... Ts>
    instruction* add_instruction(operation op, Ts*... args)
    {
        return add_instruction(op, {args...});
    }
    instruction* add_instruction(operation op, std::vector<instruction*> args);
    template <class... Ts>
    instruction* add_literal(Ts&&... xs)
    {
        return add_literal(literal{std::forward<Ts>(xs)...});
    }

    instruction* add_literal(literal l);

    instruction* add_parameter(std::string name, shape s);

    literal eval(std::unordered_map<std::string, argument> params) const;

    // TODO: Change to stream operator
    void print() const;

    bool has_instruction(const instruction* ins) const;

    private:
    std::unique_ptr<program_impl> impl;
};

} // namespace rtg

#endif
