#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_PROGRAM_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_PROGRAM_HPP

#include <list>
#include <unordered_map>
#include <migraphx/operation.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/builtin.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/target.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/env.hpp>
#include <migraphx/config.hpp>
#include <algorithm>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_COMPILE)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_EVAL)

struct program_impl;

const operation& get_operation(instruction_ref ins);

/**
 * @brief Stores the instruction stream
 */
struct program
{
    program();

    // move constructor
    program(program&&) noexcept;

    // copy constructor
    program(const program&);

    // copy assignment operator
    program& operator=(program);

    ~program() noexcept;

    using parameter_map = std::unordered_map<std::string, argument>;

    template <class... Ts>
    instruction_ref add_instruction(operation op, Ts... args)
    {
        return add_instruction(op, {args...});
    }
    instruction_ref add_instruction(const operation& op, std::vector<instruction_ref> args);

    template <class... Ts>
    instruction_ref insert_instruction(instruction_ref ins, operation op, Ts... args)
    {
        return insert_instruction(ins, op, {args...});
    }
    instruction_ref
    insert_instruction(instruction_ref ins, const operation& op, std::vector<instruction_ref> args);

    template <class... Ts>
    instruction_ref replace_instruction(instruction_ref ins, operation op, Ts... args)
    {
        return replace_instruction(ins, op, {args...});
    }
    instruction_ref replace_instruction(instruction_ref ins,
                                        const operation& op,
                                        std::vector<instruction_ref> args);

    instruction_ref replace_instruction(instruction_ref ins, instruction_ref rep);

    instruction_ref remove_instruction(instruction_ref ins);
    instruction_ref remove_instructions(instruction_ref first, instruction_ref last);

    instruction_ref move_instruction(instruction_ref src, instruction_ref dst);

    template <class... Ts>
    instruction_ref add_literal(Ts&&... xs)
    {
        return add_literal(literal{std::forward<Ts>(xs)...});
    }

    instruction_ref add_literal(literal l);

    instruction_ref add_outline(const shape& s);

    instruction_ref add_parameter(std::string name, shape s);

    shape get_parameter_shape(std::string name) const;

    instruction_ref get_parameter(std::string name) const;

    std::unordered_map<std::string, shape> get_parameter_shapes() const;

    // argument eval(parameter_map params) const;
    std::vector<argument> eval(parameter_map params) const;

    bool has_instruction(instruction_ref ins) const;

    std::size_t size() const;
    instruction_ref begin() const;
    instruction_ref end() const;

    // shape get_shape() const;

    // support multiple program outputs
    std::vector<shape> get_shapes() const;

    context& get_context() const;

    instruction_ref validate() const;

    void compile(const target& t, compile_options options = compile_options{});

    void finalize();

    void perf_report(std::ostream& os, std::size_t n, parameter_map params) const;

    void debug_print() const;
    void debug_print(instruction_ref ins) const;
    void debug_print(const std::vector<instruction_ref>& inss) const;
    void print_graph(std::ostream& os, bool brief = false) const;
    void print_cpp(std::ostream& os) const;

    void dry_run(parameter_map params) const;

    void annotate(std::ostream& os, std::function<void(instruction_ref)> a) const;

    friend std::ostream& operator<<(std::ostream& os, const program& p);
    friend bool operator==(const program& x, const program& y);
    friend bool operator!=(const program& x, const program& y) { return !(x == y); }

    private:
    void assign(const program& p);

    private:
    std::unique_ptr<program_impl> impl;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
