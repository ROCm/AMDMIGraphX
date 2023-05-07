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
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_INSTRUCTION_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_INSTRUCTION_HPP

#include <string>
#include <utility>

#include <migraphx/config.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/module_ref.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/erase.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_EXPORT shape compute_shape(const operation& op, const std::vector<instruction_ref>& args);
MIGRAPHX_EXPORT shape compute_shape(const operation& op,
                    const std::vector<instruction_ref>& args,
                    const std::vector<module_ref>& mods);
MIGRAPHX_EXPORT std::vector<shape> to_shapes(const std::vector<instruction_ref>& args);
MIGRAPHX_EXPORT std::vector<shape> try_compute_shape(const operation& op, const std::vector<shape>& inputs);

struct instruction
{
    instruction() = default;

    MIGRAPHX_EXPORT instruction(operation o, shape r, std::vector<instruction_ref> args);

    MIGRAPHX_EXPORT instruction(operation o,
                shape r,
                std::vector<instruction_ref> args,
                std::vector<module_ref> modules);

    MIGRAPHX_EXPORT instruction(literal l);

    MIGRAPHX_EXPORT void replace(operation o);

    MIGRAPHX_EXPORT void recompute_shape();

    MIGRAPHX_EXPORT void clear_arguments();

    friend bool operator==(const instruction& i, instruction_ref ref);

    MIGRAPHX_EXPORT bool valid(instruction_ref start, bool check_order = false) const;

    MIGRAPHX_EXPORT bool valid() const;

    MIGRAPHX_EXPORT shape get_shape() const;
    MIGRAPHX_EXPORT const literal& get_literal() const;

    MIGRAPHX_EXPORT const operation& get_operator() const;

    MIGRAPHX_EXPORT std::string name() const;

    MIGRAPHX_EXPORT const std::vector<instruction_ref>& inputs() const;

    MIGRAPHX_EXPORT const std::vector<module_ref>& module_inputs() const;

    MIGRAPHX_EXPORT const std::vector<instruction_ref>& outputs() const;

    friend bool operator==(const instruction& x, const instruction& y);

    friend bool operator!=(const instruction& x, const instruction& y);

    friend bool operator==(instruction_ref ref, const instruction& i);

    friend bool operator!=(const instruction& i, instruction_ref ref);

    friend bool operator!=(instruction_ref ref, const instruction& i);

    MIGRAPHX_EXPORT void add_output(instruction_ref ins);

    template <class T>
    void remove_output(const T& ins)
    {
        migraphx::erase(output, ins);
    }

    MIGRAPHX_EXPORT static void replace_refs(instruction_ref ins,
                             const std::unordered_map<instruction_ref, instruction_ref>& map_insts,
                             const std::unordered_map<module_ref, module_ref>& map_mods);

    MIGRAPHX_EXPORT static void backreference(instruction_ref ref);

    MIGRAPHX_EXPORT static void replace_argument(instruction_ref ins, instruction_ref old, instruction_ref new_ins);

    MIGRAPHX_EXPORT static void replace_mod_argument(instruction_ref ins, module_ref old, module_ref new_mod);

    MIGRAPHX_EXPORT static void
    replace(instruction_ref ins, operation o, const shape& r, std::vector<instruction_ref> args);

    MIGRAPHX_EXPORT static void replace(instruction_ref ins,
                        operation o,
                        const shape& r,
                        std::vector<instruction_ref> args,
                        std::vector<module_ref> module_args);

    MIGRAPHX_EXPORT bool can_eval() const;

    MIGRAPHX_EXPORT bool is_undefined() const;

    MIGRAPHX_EXPORT argument eval(bool check_eval = true) const;

    MIGRAPHX_EXPORT void finalize(context& ctx);

    MIGRAPHX_EXPORT static instruction_ref get_output_alias(instruction_ref ins, bool shallow = false);

    MIGRAPHX_EXPORT void set_normalized(bool value = true);
    MIGRAPHX_EXPORT bool is_normalized() const;

    MIGRAPHX_EXPORT bool need_normalization() const;

    MIGRAPHX_EXPORT operation normalized_operator() const;

    MIGRAPHX_EXPORT void debug_print() const;

    MIGRAPHX_EXPORT static void print(std::ostream& os,
                      instruction_ref ins,
                      const std::unordered_map<instruction_ref, std::string>& names);

    private:
    // internal
    void replace(operation o, const shape& r, std::vector<instruction_ref> args);

    // internal
    void replace(operation o,
                 const shape& r,
                 std::vector<instruction_ref> args,
                 std::vector<module_ref> mdl_args);

    // internal
    void replace(std::vector<instruction_ref> args);

    // internal
    void replace(std::vector<instruction_ref> args, std::vector<module_ref> mdl_args);

    // internal
    void replace_argument(instruction_ref old, instruction_ref new_ins);

    // internal
    void replace_mod_argument(module_ref old, module_ref new_mod);

    void replace(const shape& r);

    operation op;
    shape result{};
    std::vector<instruction_ref> output;
    std::vector<instruction_ref> arguments;
    std::vector<module_ref> module_args;
    literal lit;
    bool normalized = false;
};
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
