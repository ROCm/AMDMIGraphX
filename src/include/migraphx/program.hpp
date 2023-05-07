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
#include <migraphx/target_assignments.hpp>
#include <migraphx/assignment_options.hpp>
#include <migraphx/env.hpp>
#include <migraphx/config.hpp>
#include <migraphx/execution_environment.hpp>
#include <algorithm>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_COMPILE)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_EVAL)

struct program_impl;

struct marker;

/**
 * @brief Stores the instruction stream
 */
struct program
{
    MIGRAPHX_EXPORT program();

    // move constructor
    MIGRAPHX_EXPORT program(program&&) noexcept;

    // copy constructor
    MIGRAPHX_EXPORT program(const program&);

    // copy assignment operator
    MIGRAPHX_EXPORT program& operator=(program);

    MIGRAPHX_EXPORT ~program() noexcept;

    MIGRAPHX_EXPORT std::vector<std::string> get_parameter_names() const;

    MIGRAPHX_EXPORT shape get_parameter_shape(std::string name) const;

    MIGRAPHX_EXPORT instruction_ref get_parameter(std::string name) const;

    MIGRAPHX_EXPORT std::unordered_map<std::string, shape> get_parameter_shapes() const;

    MIGRAPHX_EXPORT std::vector<argument> eval(parameter_map params,
                               execution_environment exec_env = execution_environment{}) const;
    MIGRAPHX_EXPORT std::size_t size() const;

    MIGRAPHX_EXPORT std::vector<shape> get_output_shapes() const;

    MIGRAPHX_EXPORT context& get_context() const;

    MIGRAPHX_EXPORT instruction_ref validate() const;

    MIGRAPHX_EXPORT target_assignments get_target_assignments(const std::vector<target>& targets,
                                              assignment_options options = assignment_options{});

    MIGRAPHX_EXPORT void compile(const target& t, compile_options options = compile_options{});

    MIGRAPHX_EXPORT bool is_compiled() const;

    MIGRAPHX_EXPORT void finalize();

    MIGRAPHX_EXPORT void
    perf_report(std::ostream& os, std::size_t n, parameter_map params, std::size_t batch = 1) const;

    MIGRAPHX_EXPORT void mark(const parameter_map& params, marker&& m);

    MIGRAPHX_EXPORT value to_value() const;
    MIGRAPHX_EXPORT void from_value(const value& v);

    MIGRAPHX_EXPORT void debug_print() const;
    MIGRAPHX_EXPORT void debug_print(instruction_ref ins) const;
    MIGRAPHX_EXPORT void print(std::unordered_map<instruction_ref, std::string>& names,
               const std::function<void(instruction_ref,
                                        std::unordered_map<instruction_ref, std::string>)>&
                   print_func) const;
    MIGRAPHX_EXPORT void print(const std::function<void(instruction_ref ins,
                                        std::unordered_map<instruction_ref, std::string>)>&
                   print_func) const;

    MIGRAPHX_EXPORT void print_graph(std::ostream& os, bool brief = false) const;
    MIGRAPHX_EXPORT void print_py(std::ostream& os) const;
    MIGRAPHX_EXPORT void print_cpp(std::ostream& os) const;

    MIGRAPHX_EXPORT void dry_run(parameter_map params) const;

    MIGRAPHX_EXPORT void annotate(std::ostream& os, const std::function<void(instruction_ref)>& a) const;

    MIGRAPHX_EXPORT program& sort();

    MIGRAPHX_EXPORT friend std::ostream& operator<<(std::ostream& os, const program& p);
    MIGRAPHX_EXPORT friend bool operator==(const program& x, const program& y);
    MIGRAPHX_EXPORT friend bool operator!=(const program& x, const program& y) { return not(x == y); }

    // module related api
    MIGRAPHX_EXPORT module* create_module(const std::string& name);
    MIGRAPHX_EXPORT module* get_module(const std::string& name);
    MIGRAPHX_EXPORT const module* get_module(const std::string& name) const;

    MIGRAPHX_EXPORT module* get_main_module();
    MIGRAPHX_EXPORT const module* get_main_module() const;

    MIGRAPHX_EXPORT std::vector<const module*> get_modules() const;
    MIGRAPHX_EXPORT std::vector<module*> get_modules();

    MIGRAPHX_EXPORT std::unordered_multimap<module_ref, module_ref> get_module_tree();

    MIGRAPHX_EXPORT void remove_module(const std::string& name);
    MIGRAPHX_EXPORT void remove_unused_modules();

    private:
    void assign(const program& p);
    std::unique_ptr<program_impl> impl;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
