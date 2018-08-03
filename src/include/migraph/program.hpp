#ifndef MIGRAPH_GUARD_MIGRAPHLIB_PROGRAM_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_PROGRAM_HPP

#include <list>
#include <unordered_map>
#include <migraph/operation.hpp>
#include <migraph/literal.hpp>
#include <migraph/builtin.hpp>
#include <migraph/instruction_ref.hpp>
#include <migraph/target.hpp>
#include <algorithm>
#include <iostream>

namespace migraph {

struct program_impl;

const operation& get_operation(instruction_ref ins);

/**
 * @brief Stores the instruction stream
 */
struct program
{
    program();
    program(program&&) noexcept;
    program& operator=(program&&) noexcept;
    ~program() noexcept;

    using parameter_map = std::unordered_map<std::string, argument>;

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

    instruction_ref remove_instruction(instruction_ref ins);
    instruction_ref remove_instructions(instruction_ref first, instruction_ref last);

    instruction_ref move_instruction(instruction_ref src, instruction_ref dst);

    template <class... Ts>
    instruction_ref add_literal(Ts&&... xs)
    {
        return add_literal(literal{std::forward<Ts>(xs)...});
    }

    instruction_ref add_literal(literal l);

    instruction_ref add_outline(shape s);

    instruction_ref add_parameter(std::string name, shape s);

    shape get_parameter_shape(std::string name);

    argument eval(parameter_map params) const;

    bool has_instruction(instruction_ref ins) const;

    instruction_ref begin();
    instruction_ref end();

    shape get_shape() const;

    instruction_ref validate() const;

    void compile(const target& t);

    friend std::ostream& operator<<(std::ostream& os, const program& p);
    friend bool operator==(const program& x, const program& y);
    friend bool operator!=(const program& x, const program& y) { return !(x == y); }

    private:
    std::unique_ptr<program_impl> impl;
};

} // namespace migraph

#endif
