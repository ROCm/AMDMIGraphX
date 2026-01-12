/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <unordered_map>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

// NOLINTNEXTLINE(modernize-return-braced-init-list)
value get_default_options() { return value::object{}; }

static std::unordered_map<std::string, op_builder_if>& builder_map()
{
    static std::unordered_map<std::string, op_builder_if> m; // NOLINT
    return m;
}

bool has_op_builder(const std::string& name) { return builder_map().count(name) == 1; }

void register_builder(const std::string& name, op_builder_if opb_if)
{
    builder_map()[name] = std::move(opb_if);
}

value get_op_builder_value(const std::string& name)
{
    if(not has_op_builder(name))
        MIGRAPHX_THROW("GET_OP_BUILDER_VALUE: OpBuilder not found: " + name);
    return builder_map().at(name).to_val_func();
}

static std::vector<instruction_ref> default_op_builder(module& m,
                                                       const std::vector<instruction_ref>& args,
                                                       const std::string& name,
                                                       const value& options)
{
    const auto& op = make_op(name, options);
    return {m.add_instruction(op, args)};
}

std::vector<instruction_ref> insert(const std::string& name,
                                    module& m,
                                    instruction_ref ins,
                                    const std::vector<instruction_ref>& args,
                                    const value& options)
{
    return has_op_builder(name) ? builder_map()[name].bld_func(m, ins, args, {}, options)
                                : default_op_builder(m, args, name, options);
}

std::vector<instruction_ref> insert(const std::string& name,
                                    module& m,
                                    instruction_ref ins,
                                    const std::vector<instruction_ref>& args,
                                    const std::vector<module_ref>& module_args,
                                    const value& options)
{
    return has_op_builder(name) ? builder_map()[name].bld_func(m, ins, args, module_args, options)
                                : default_op_builder(m, args, name, options);
}

std::vector<instruction_ref> add(const std::string& name,
                                 module& m,
                                 const std::vector<instruction_ref>& args,
                                 const value& options)
{
    return has_op_builder(name) ? builder_map()[name].bld_func(m, m.end(), args, {}, options)
                                : default_op_builder(m, args, name, options);
}

std::vector<instruction_ref> add(const std::string& name,
                                 module& m,
                                 const std::vector<instruction_ref>& args,
                                 const std::vector<module_ref>& module_args,
                                 const value& options)
{
    return has_op_builder(name)
               ? builder_map()[name].bld_func(m, m.end(), args, module_args, options)
               : default_op_builder(m, args, name, options);
}

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
