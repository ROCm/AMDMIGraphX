/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_MODULE_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_MODULE_HPP

#include <algorithm>
#include <iostream>
#include <list>
#include <unordered_set>
#include <unordered_map>

#include <migraphx/config.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/builtin.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/target.hpp>
#include <migraphx/module_ref.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/env.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

const operation& get_operation(instruction_ref ins);

struct module_impl;

using parameter_map = std::unordered_map<std::string, argument>;
using ins_dep_map   = std::unordered_map<instruction_ref, std::unordered_set<instruction_ref>>;

/**
 * @brief Stores the instruction stream
 */
struct module
{
    MIGRAPHX_EXPORT module(const std::string& name = "");

    // move constructor
    MIGRAPHX_EXPORT module(module&&) noexcept;

    // copy constructor
    MIGRAPHX_EXPORT module(const module&);

    // copy assignment operator
    MIGRAPHX_EXPORT module& operator=(module);

    MIGRAPHX_EXPORT ~module() noexcept;

    MIGRAPHX_EXPORT std::string name() const;

    MIGRAPHX_EXPORT bool bypass() const;
    MIGRAPHX_EXPORT void set_bypass(bool b = true);

    template <class... Ts, MIGRAPHX_REQUIRES(std::is_same<Ts, instruction_ref>{}...)>
    instruction_ref add_instruction(operation op, Ts... args)
    {
        return add_instruction(op, {args...});
    }

    MIGRAPHX_EXPORT instruction_ref add_instruction(const operation& op, std::vector<instruction_ref> args);

    MIGRAPHX_EXPORT instruction_ref add_instruction(const operation& op,
                                    std::vector<instruction_ref> args,
                                    std::vector<module_ref> module_args);

    template <class... Ts, MIGRAPHX_REQUIRES(std::is_same<Ts, instruction_ref>{}...)>
    instruction_ref insert_instruction(instruction_ref ins, operation op, Ts... args)
    {
        return insert_instruction(ins, op, {args...});
    }
    MIGRAPHX_EXPORT instruction_ref
    insert_instruction(instruction_ref ins, const operation& op, std::vector<instruction_ref> args);

    MIGRAPHX_EXPORT instruction_ref insert_instruction(instruction_ref ins,
                                       const operation& op,
                                       std::vector<instruction_ref> args,
                                       std::vector<module_ref> module_args);

    template <class... Ts, MIGRAPHX_REQUIRES(std::is_same<Ts, instruction_ref>{}...)>
    instruction_ref replace_instruction(instruction_ref ins, operation op, Ts... args)
    {
        return replace_instruction(ins, op, {args...});
    }
    MIGRAPHX_EXPORT instruction_ref replace_instruction(instruction_ref ins,
                                        const operation& op,
                                        std::vector<instruction_ref> args) MIGRAPHX_TIDY_CONST;

    MIGRAPHX_EXPORT instruction_ref replace_instruction(instruction_ref ins,
                                        const operation& op,
                                        std::vector<instruction_ref> args,
                                        std::vector<module_ref> module_args) MIGRAPHX_TIDY_CONST;

    MIGRAPHX_EXPORT instruction_ref replace_instruction(instruction_ref ins, instruction_ref rep);

    MIGRAPHX_EXPORT instruction_ref remove_instruction(instruction_ref ins);
    MIGRAPHX_EXPORT instruction_ref remove_instructions(instruction_ref first, instruction_ref last);

    MIGRAPHX_EXPORT instruction_ref move_instruction(instruction_ref src, instruction_ref dst);
    MIGRAPHX_EXPORT instruction_ref move_instructions(instruction_ref src, instruction_ref dst);

    MIGRAPHX_EXPORT std::vector<instruction_ref>
    add_instructions(const std::vector<instruction_ref>& instructions,
                     std::unordered_map<instruction_ref, instruction_ref> map_ins = {});

    MIGRAPHX_EXPORT std::vector<instruction_ref>
    add_instructions(const_module_ref m,
                     std::unordered_map<instruction_ref, instruction_ref> map_ins = {});

    MIGRAPHX_EXPORT std::vector<instruction_ref>
    add_instructions(instruction_ref start,
                     instruction_ref last,
                     std::unordered_map<instruction_ref, instruction_ref> map_ins = {});

    MIGRAPHX_EXPORT std::vector<instruction_ref>
    insert_instructions(instruction_ref ins,
                        const std::vector<instruction_ref>& instructions,
                        std::unordered_map<instruction_ref, instruction_ref> map_ins = {});

    MIGRAPHX_EXPORT std::vector<instruction_ref>
    insert_instructions(instruction_ref ins,
                        const_module_ref m,
                        std::unordered_map<instruction_ref, instruction_ref> map_ins = {});

    MIGRAPHX_EXPORT std::vector<instruction_ref>
    insert_instructions(instruction_ref ins,
                        instruction_ref start,
                        instruction_ref last,
                        std::unordered_map<instruction_ref, instruction_ref> map_ins = {});

    template <class... Ts>
    instruction_ref add_literal(Ts&&... xs)
    {
        return add_literal(literal{std::forward<Ts>(xs)...});
    }

    MIGRAPHX_EXPORT instruction_ref add_literal(literal l);

    MIGRAPHX_EXPORT instruction_ref add_outline(const shape& s);

    MIGRAPHX_EXPORT instruction_ref add_parameter(std::string name, shape s);

    MIGRAPHX_EXPORT instruction_ref add_return(std::vector<instruction_ref> args);

    MIGRAPHX_EXPORT instruction_ref replace_return(std::vector<instruction_ref> args);

    MIGRAPHX_EXPORT instruction_ref insert_literal(instruction_ref ins, literal l);

    MIGRAPHX_EXPORT instruction_ref insert_parameter(instruction_ref ins, std::string name, shape s);

    MIGRAPHX_EXPORT std::vector<std::string> get_parameter_names() const;

    MIGRAPHX_EXPORT shape get_parameter_shape(std::string name) const;

    MIGRAPHX_EXPORT instruction_ref get_parameter(std::string name) const;

    MIGRAPHX_EXPORT std::unordered_map<std::string, shape> get_parameter_shapes() const;

    MIGRAPHX_EXPORT bool has_instruction(instruction_ref ins) const;

    MIGRAPHX_EXPORT std::vector<instruction_ref> get_returns() const;

    MIGRAPHX_EXPORT std::size_t size() const;
    MIGRAPHX_EXPORT instruction_ref begin() const;
    MIGRAPHX_EXPORT instruction_ref end() const;

    MIGRAPHX_EXPORT std::vector<shape> get_output_shapes() const;

    MIGRAPHX_EXPORT instruction_ref validate() const;
    MIGRAPHX_EXPORT instruction_ref find_dangling_reference() const;

    MIGRAPHX_EXPORT void finalize(context& ctx);

    MIGRAPHX_EXPORT void debug_print() const;
    MIGRAPHX_EXPORT void debug_print(instruction_ref ins) const;
    MIGRAPHX_EXPORT void debug_print(instruction_ref ins,
                     std::unordered_map<instruction_ref, std::string>& names) const;
    MIGRAPHX_EXPORT void debug_print(const std::vector<instruction_ref>& inss) const;

    MIGRAPHX_EXPORT std::unordered_map<instruction_ref, std::string> print(
        const std::function<void(
            instruction_ref, const std::unordered_map<instruction_ref, std::string>&)>& print_func,
        std::unordered_map<instruction_ref, std::string> names) const;
    MIGRAPHX_EXPORT void print(const std::function<void(instruction_ref,
                                        const std::unordered_map<instruction_ref, std::string>&)>&
                   print_func) const;

    MIGRAPHX_EXPORT void print_graph(std::ostream& os, bool brief = false) const;

    MIGRAPHX_EXPORT void print_py(std::ostream& os) const;
    MIGRAPHX_EXPORT std::unordered_map<instruction_ref, std::string>
    print_py(std::ostream& os,
             const std::string& mname,
             std::unordered_map<instruction_ref, std::string> names) const;

    MIGRAPHX_EXPORT void print_cpp(std::ostream& os) const;
    MIGRAPHX_EXPORT std::unordered_map<instruction_ref, std::string>
    print_cpp(std::ostream& os,
              const std::string& mname,
              std::unordered_map<instruction_ref, std::string> names) const;

    MIGRAPHX_EXPORT void annotate(std::ostream& os, std::function<void(instruction_ref)> a) const;

    [[nodiscard]] MIGRAPHX_EXPORT std::vector<module_ref> get_sub_modules(bool shallow = false) const;
    MIGRAPHX_EXPORT module& sort();
    [[nodiscard]] MIGRAPHX_EXPORT ins_dep_map calc_implicit_deps() const;

    MIGRAPHX_EXPORT friend std::ostream& operator<<(std::ostream& os, const module& m);
    MIGRAPHX_EXPORT friend bool operator==(const module& x, const module& y);
    friend bool operator!=(const module& x, const module& y) { return not(x == y); }

    private:
    void assign(const module& m);
    void calc_implicit_deps(const module& smod,
                            const module& pmod,
                            instruction_ref ins,
                            ins_dep_map& deps) const;

    std::unique_ptr<module_impl> impl;
};

inline module& get_module(module& m) { return m; }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
