#ifndef RTG_GUARD_RTGLIB_PROGRAM_HPP
#define RTG_GUARD_RTGLIB_PROGRAM_HPP

#include <list>
#include <unordered_map>
#include <rtg/operation.hpp>
#include <rtg/literal.hpp>
#include <rtg/builtin.hpp>
#include <rtg/instruction_ref.hpp>
#include <rtg/target.hpp>
#include <algorithm>
#include <iostream>

namespace rtg {

struct program_impl;

/**
 * @brief Stores the instruction stream
 */
struct program
{
    program();
    program(program&&) noexcept;
    program& operator=(program&&) noexcept;
    ~program() noexcept;

    template <class... Ts>
    instruction_ref add_instruction(operation op, Ts... args)
    {
        return add_instruction(op, {args...});
    }
    instruction_ref add_instruction(operation op, std::vector<instruction_ref> args);

    template <class... Ts>
    instruction_ref insert_instruction(instruction_ref ins, operation op, Ts... args)
    {
        return insert_instruction(ins, op, {args...});
    }
    instruction_ref
    insert_instruction(instruction_ref ins, operation op, std::vector<instruction_ref> args);

    template <class... Ts>
    instruction_ref replace_instruction(instruction_ref ins, operation op, Ts... args)
    {
        return replace_instruction(ins, op, {args...});
    }
    instruction_ref
    replace_instruction(instruction_ref ins, operation op, std::vector<instruction_ref> args);

    template <class... Ts>
    instruction_ref add_literal(Ts&&... xs)
    {
        return add_literal(literal{std::forward<Ts>(xs)...});
    }

    instruction_ref add_literal(literal l);
    
    instruction_ref add_outline(shape s);

    instruction_ref add_parameter(std::string name, shape s);

    shape get_parameter_shape(std::string name);

    literal eval(std::unordered_map<std::string, argument> params) const;

    friend std::ostream& operator<<(std::ostream& os, const program& p);

    bool has_instruction(instruction_ref ins) const;

    instruction_ref begin();
    instruction_ref end();

    instruction_ref validate() const;

    void compile(const target& t);

    private:
    std::unique_ptr<program_impl> impl;
};

} // namespace rtg

#endif
