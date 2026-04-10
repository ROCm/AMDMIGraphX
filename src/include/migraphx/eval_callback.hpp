/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_EVAL_CALLBACK_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_EVAL_CALLBACK_HPP

#include <functional>
#include <string>
#include <unordered_set>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// Callback invoked per-instruction during program::eval() with the result
/// buffer copied to host memory.
///
/// Two optional filters restrict which instructions trigger the callback.
/// When both are empty every instruction fires. When either contains a match
/// the callback fires (OR semantics). Non-matching instructions skip the
/// device-sync and host-copy entirely.
///
///   Name filter   – matches the operator name (e.g. "gpu::code_object")
///                   or the kernel symbol name (e.g. "concat_kernel")
///   Instruction filter – matches one specific instruction in the graph
struct eval_callback
{
    using callback_function = std::function<void(instruction_ref ins, const argument& output)>;

    eval_callback() {}

    explicit eval_callback(callback_function f,
                           std::unordered_set<std::string> names            = {},
                           std::unordered_set<instruction_ref> instructions = {})
        : cb(std::move(f)), name_filter(std::move(names)), ins_filter(std::move(instructions))
    {
    }

    bool enabled() const { return cb != nullptr; }

    bool matches(instruction_ref ins) const
    {
        if(name_filter.empty() and ins_filter.empty())
            return true;
        if(ins_filter.count(ins) > 0)
            return true;
        if(name_filter.empty())
            return false;
        if(name_filter.count(ins->name()) > 0)
            return true;
        auto sym = get_symbol_name(ins->get_operator());
        return not sym.empty() and name_filter.count(sym) > 0;
    }

    void operator()(instruction_ref ins, const argument& output) const
    {
        if(cb)
            cb(ins, output);
    }

    static std::string get_symbol_name(const operation& op)
    {
        return op.to_value().get("symbol_name", std::string{});
    }

    private:
    callback_function cb = nullptr;
    std::unordered_set<std::string> name_filter;
    std::unordered_set<instruction_ref> ins_filter;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
