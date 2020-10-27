#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_PROGRAM_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_PROGRAM_HPP

#include <list>
#include <unordered_map>
#include <migraphx/operation.hpp>
#include <migraphx/module.hpp>
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

    std::vector<std::string> get_parameter_names() const;

    shape get_parameter_shape(std::string name) const;

    instruction_ref get_parameter(std::string name) const;

    std::unordered_map<std::string, shape> get_parameter_shapes() const;

    std::vector<argument> eval(parameter_map params) const;

    bool has_instruction(instruction_ref ins) const;

    std::size_t size() const;
    instruction_ref begin() const;
    instruction_ref end() const;

    std::vector<shape> get_output_shapes() const;

    context& get_context() const;

    instruction_ref validate() const;

    void compile(const target& t, compile_options options = compile_options{});

    bool is_compiled() const;

    void finalize();

    void perf_report(std::ostream& os, std::size_t n, parameter_map params) const;

    value to_value() const;
    void from_value(const value& v);

    void debug_print() const;
    void debug_print(instruction_ref ins) const;
    void debug_print(const std::vector<instruction_ref>& inss) const;
    void print_graph(std::ostream& os, bool brief = false) const;
    void print_cpp(std::ostream& os) const;

    void dry_run(parameter_map params) const;

    void annotate(std::ostream& os, std::function<void(instruction_ref)> a) const;

    program& sort();

    friend std::ostream& operator<<(std::ostream& os, const program& p);
    friend bool operator==(const program& x, const program& y);
    friend bool operator!=(const program& x, const program& y) { return !(x == y); }

    module& get_main_module() { return main_module; }

    private:
    void assign(const program& p);
    module main_module;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
