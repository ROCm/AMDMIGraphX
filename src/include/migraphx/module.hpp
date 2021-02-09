#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_MODULE_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_MODULE_HPP

#include <list>
#include <unordered_set>
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

const operation& get_operation(instruction_ref ins);

struct module_impl;

using parameter_map = std::unordered_map<std::string, argument>;

using module_ref = module*;

/**
 * @brief Stores the instruction stream
 */
struct module
{
    module(const std::string& name = "");

    // move constructor
    module(module&&) noexcept;

    // copy constructor
    module(const module&);

    // copy assignment operator
    module& operator=(module);

    ~module() noexcept;

    std::string name() const;

    template <class... Ts>
    instruction_ref add_instruction(operation op, Ts... args)
    {
        return add_instruction(op, {args...});
    }

    instruction_ref add_instruction(const operation& op, std::vector<instruction_ref> args);

    instruction_ref add_instruction(const operation& op,
                                    std::vector<instruction_ref> args,
                                    std::vector<module_ref> module_args);

    template <class... Ts>
    instruction_ref insert_instruction(instruction_ref ins, operation op, Ts... args)
    {
        return insert_instruction(ins, op, {args...});
    }
    instruction_ref
    insert_instruction(instruction_ref ins, const operation& op, std::vector<instruction_ref> args);

    instruction_ref insert_instruction(instruction_ref ins,
                                       const operation& op,
                                       std::vector<instruction_ref> args,
                                       std::vector<module_ref> module_args);

    template <class... Ts>
    instruction_ref replace_instruction(instruction_ref ins, operation op, Ts... args)
    {
        return replace_instruction(ins, op, {args...});
    }
    instruction_ref replace_instruction(instruction_ref ins,
                                        const operation& op,
                                        std::vector<instruction_ref> args) MIGRAPHX_TIDY_CONST;

    instruction_ref replace_instruction(instruction_ref ins,
                                        const operation& op,
                                        std::vector<instruction_ref> args,
                                        std::vector<module_ref> module_args) const;

    instruction_ref replace_instruction(instruction_ref ins, instruction_ref rep);

    instruction_ref remove_instruction(instruction_ref ins);
    instruction_ref remove_instructions(instruction_ref first, instruction_ref last);

    instruction_ref move_instruction(instruction_ref src, instruction_ref dst);
    instruction_ref move_instructions(instruction_ref src, instruction_ref dst);

    template <class... Ts>
    instruction_ref add_literal(Ts&&... xs)
    {
        return add_literal(literal{std::forward<Ts>(xs)...});
    }

    instruction_ref add_literal(literal l);

    instruction_ref add_outline(const shape& s);

    instruction_ref add_parameter(std::string name, shape s);

    instruction_ref add_return(std::vector<instruction_ref> args);

    std::vector<std::string> get_parameter_names() const;

    shape get_parameter_shape(std::string name) const;

    instruction_ref get_parameter(std::string name) const;

    std::unordered_map<std::string, shape> get_parameter_shapes() const;

    bool has_instruction(instruction_ref ins) const;

    std::size_t size() const;
    instruction_ref begin() const;
    instruction_ref end() const;

    std::vector<shape> get_output_shapes() const;

    instruction_ref validate() const;

    void finalize(context& ctx);

    value to_value(value& v, std::unordered_map<instruction_ref, std::string> names) const;
    void from_value(const value& v, std::unordered_map<std::string, instruction_ref>& instructions, const std::unordered_map<std::string, module_ref>& map_mods = {});

    void debug_print() const;
    void debug_print(instruction_ref ins) const;
    void debug_print(instruction_ref ins,
                     std::unordered_map<instruction_ref, std::string>& names) const;
    void debug_print(const std::vector<instruction_ref>& inss) const;
    void print(std::unordered_map<instruction_ref, std::string>& names,
               const std::function<void(instruction_ref)>& print_func) const;

    void print_graph(std::ostream& os, bool brief = false) const;
    void print_cpp(std::ostream& os) const;

    void annotate(std::ostream& os, std::function<void(instruction_ref)> a) const;

    module& sort();

    friend std::ostream& operator<<(std::ostream& os, const module& m);
    friend bool operator==(const module& x, const module& y);
    friend bool operator!=(const module& x, const module& y) { return !(x == y); }

    void assign(const module& m,
                std::unordered_map<instruction_ref, instruction_ref> ins_map,
                const std::unordered_map<module_ref, module_ref>& mod_map);

    private:
    std::unordered_set<module_ref> get_parent_modules() const;

    std::unique_ptr<module_impl> impl;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
