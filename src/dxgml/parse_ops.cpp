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
#include "dxgml_parser.hpp"
#include <migraphx/errors.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/literal.hpp>

#include <unordered_map>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

// Collect input instruction_refs from comma-separated SSA operand names.
// operands_raw is the content of the parentheses: "%a, %b, %c" or
// "#dxgml.constant_resource<...>" (for constant ops).
static std::vector<std::string> split_operands(const std::string& s)
{
    std::vector<std::string> names;
    std::size_t pos = 0;
    while(pos < s.size())
    {
        auto pct = s.find('%', pos);
        if(pct == std::string::npos)
            break;
        auto end = pct + 1;
        while(end < s.size() && s[end] != ',' && s[end] != ' ' && s[end] != ')')
            ++end;
        names.push_back(s.substr(pct + 1, end - pct - 1));
        pos = end;
    }
    return names;
}

// Collect instruction_refs for SSA operands from value_map.
// Returns only those that start with '%' (skips attribute operands like #dxgml.*).
static std::vector<instruction_ref>
collect_inputs(const std::string& operands_raw,
               const std::unordered_map<std::string, instruction_ref>& value_map,
               const std::string& op_name)
{
    auto names = split_operands(operands_raw);
    std::vector<instruction_ref> inputs;
    for(const auto& n : names)
    {
        auto it = value_map.find(n);
        if(it == value_map.end())
            MIGRAPHX_THROW("DxGML op '" + op_name + "': undefined SSA value: %" + n);
        inputs.push_back(it->second);
    }
    return inputs;
}

// ---------------------------------------------------------------------------
// Constant: dxgml_op.constant(#dxgml.constant_resource<NAME : TYPE>)
// ---------------------------------------------------------------------------
static instruction_ref parse_constant(dxgml_parser& self, const std::string& operands_raw)
{
    // operands_raw = "#dxgml.constant_resource<NAME : !dxgml.tensor<...>>"
    // (possibly prefixed by the dialect hash)
    const std::string pfx = "constant_resource<";
    auto pos = operands_raw.find(pfx);
    if(pos == std::string::npos)
        MIGRAPHX_THROW("DxGML: constant missing constant_resource attribute: " + operands_raw);

    auto start = pos + pfx.size();
    // Find the matching '>' (may contain nested '<>')
    int depth = 1;
    auto end  = start;
    while(end < operands_raw.size() && depth > 0)
    {
        if(operands_raw[end] == '<')      ++depth;
        else if(operands_raw[end] == '>') --depth;
        if(depth > 0) ++end;
    }
    std::string inner = operands_raw.substr(start, end - start);

    // inner = "NAME : !dxgml.tensor<...>"
    const std::string sep = " : ";
    auto colon_pos = inner.find(sep);
    if(colon_pos == std::string::npos)
        MIGRAPHX_THROW("DxGML: cannot parse constant_resource in: " + operands_raw);

    std::string param_name = inner.substr(0, colon_pos);
    std::string type_str   = inner.substr(colon_pos + sep.size());

    shape sh = self.parse_tensor_type(type_str);
    // Constants must be appended after entry-point arg parameters, not prepended.
    // insert_parameter(end(),...) places them after all currently-existing instructions.
    return self.mm->insert_parameter(self.mm->end(), param_name, sh);
}

// ---------------------------------------------------------------------------
// Helpers for reading typed attributes from the attrs_block text
// ---------------------------------------------------------------------------

// Get a dense_integer_elements attribute value by key.
static std::vector<std::size_t>
get_dense_int(dxgml_parser& self, const std::string& attrs, const std::string& key)
{
    std::string val = self.get_attr_str(attrs, key);
    if(val.empty())
        MIGRAPHX_THROW("DxGML: missing attribute '" + key + "' in: " + attrs);
    return self.parse_dense_int_vec(val);
}

// Get an integer scalar attribute value by key.
static int64_t
get_int(dxgml_parser& self, const std::string& attrs, const std::string& key)
{
    std::string val = self.get_attr_str(attrs, key);
    if(val.empty())
        MIGRAPHX_THROW("DxGML: missing attribute '" + key + "' in: " + attrs);
    return self.parse_int_scalar(val);
}

// Get the result type from the type signature.
// type_sig looks like "(TYPE, TYPE) -> RETTYPE" or "(!dxgml.tensor<...>) -> !dxgml.tensor<...>"
// Returns the return type string.
static std::string extract_ret_type(const std::string& type_sig)
{
    auto arrow = type_sig.rfind("->");
    if(arrow == std::string::npos)
        return type_sig; // whole thing is the type
    auto after = type_sig.find_first_not_of(" \t", arrow + 2);
    if(after == std::string::npos)
        return {};
    return type_sig.substr(after);
}

// ---------------------------------------------------------------------------
// Main op dispatch
// ---------------------------------------------------------------------------

instruction_ref dxgml_parser::parse_dxgml_op(const std::string& name,
                                               const std::string& operands_raw,
                                               const std::string& attrs_block,
                                               const std::string& type_sig)
{
    // --- Constant (weight / bias) ---
    if(name == "constant")
        return parse_constant(*this, operands_raw);

    // Collect tensor inputs from the SSA operand list
    auto inputs = collect_inputs(operands_raw, value_map, name);

    // --- Unary elementwise ---
    static const std::unordered_map<std::string, std::string> unary_map = {
        {"relu",    "relu"},
        {"sigmoid", "sigmoid"},
        {"tanh",    "tanh"},
        {"erf",     "erf"},
        {"exp",     "exp"},
        {"log",     "log"},
        {"sqrt",    "sqrt"},
        {"abs",     "abs"},
        {"ceil",    "ceil"},
        {"floor",   "floor"},
        {"neg",     "neg"},
        {"rsqrt",   "rsqrt"},
        {"recip",   "recip"},
    };
    {
        auto it = unary_map.find(name);
        if(it != unary_map.end())
            return mm->add_instruction(make_op(it->second), inputs);
    }

    // --- Binary elementwise ---
    static const std::unordered_map<std::string, std::string> binary_map = {
        {"add",      "add"},
        {"subtract", "sub"},
        {"multiply", "mul"},
        {"divide",   "div"},
        {"pow",      "pow"},
        {"max",      "max"},
        {"min",      "min"},
    };
    {
        auto it = binary_map.find(name);
        if(it != binary_map.end())
            return mm->add_instruction(make_op(it->second), inputs);
    }

    // --- Convolution ---
    if(name == "convolution")
    {
        auto strides   = get_dense_int(*this, attrs_block, "strides");
        auto dilations = get_dense_int(*this, attrs_block, "dilations");
        auto pad_start = get_dense_int(*this, attrs_block, "start_padding");
        auto pad_end   = get_dense_int(*this, attrs_block, "end_padding");
        auto groups    = get_int(*this, attrs_block, "group_count");

        // MIGraphX interleaved padding: [top, bottom, left, right] for 2D
        // DxGML: start_padding=[top,left], end_padding=[bottom,right]
        std::vector<std::size_t> padding;
        for(std::size_t i = 0; i < pad_start.size(); ++i)
        {
            padding.push_back(pad_start[i]);
            padding.push_back(pad_end[i]);
        }

        // MIGraphX convolution takes exactly 2 inputs (input + filter).
        // DxGML may supply a third bias operand — use only the first two.
        std::vector<instruction_ref> conv_inputs = {inputs[0], inputs[1]};
        return mm->add_instruction(
            make_op("convolution",
                    {{"stride",   strides},
                     {"dilation", dilations},
                     {"padding",  padding},
                     {"group",    static_cast<int>(groups)}}),
            conv_inputs);
    }

    // --- Gemm / dot ---
    if(name == "gemm" || name == "dot")
        return mm->add_instruction(make_op("dot"), inputs);

    // --- Reshape ---
    if(name == "reshape")
    {
        std::string ret = extract_ret_type(type_sig);
        shape out_shape = parse_tensor_type(ret);
        return mm->add_instruction(
            make_op("reshape", {{"dims", out_shape.lens()}}), inputs);
    }

    // --- Transpose ---
    if(name == "transpose")
    {
        auto perm = get_dense_int(*this, attrs_block, "perm");
        std::vector<int64_t> permutation(perm.begin(), perm.end());
        return mm->add_instruction(
            make_op("transpose", {{"permutation", permutation}}), inputs);
    }

    // --- Cast / convert ---
    if(name == "cast")
    {
        std::string ret = extract_ret_type(type_sig);
        shape out_shape = parse_tensor_type(ret);
        return mm->add_instruction(
            make_op("convert", {{"target_type", out_shape.type()}}), inputs);
    }

    // --- Softmax ---
    if(name == "softmax" || name == "log_softmax")
    {
        int64_t axis         = get_int(*this, attrs_block, "axis");
        std::string mgx_name = (name == "softmax") ? "softmax" : "logsoftmax";
        return mm->add_instruction(make_op(mgx_name, {{"axis", axis}}), inputs);
    }

    // --- Pooling ---
    if(name == "max_pooling" || name == "average_pooling")
    {
        auto win  = get_dense_int(*this, attrs_block, "window_size");
        auto strd = get_dense_int(*this, attrs_block, "strides");
        auto pad  = get_dense_int(*this, attrs_block, "padding");
        std::string mode = (name == "max_pooling") ? "max" : "average";
        return mm->add_instruction(
            make_op("pooling",
                    {{"mode", mode}, {"lengths", win}, {"stride", strd}, {"padding", pad}}),
            inputs);
    }

    if(name == "global_avg_pool")
    {
        auto in_shape = inputs[0]->get_shape();
        std::vector<std::size_t> lengths(in_shape.lens().begin() + 2, in_shape.lens().end());
        return mm->add_instruction(
            make_op("pooling", {{"mode", "average"}, {"lengths", lengths}}), inputs);
    }

    // --- Concat ---
    if(name == "concat")
    {
        int64_t axis = get_int(*this, attrs_block, "axis");
        return mm->add_instruction(make_op("concat", {{"axis", axis}}), inputs);
    }

    // --- Slice ---
    if(name == "slice")
    {
        auto axes   = get_dense_int(*this, attrs_block, "axes");
        auto starts = get_dense_int(*this, attrs_block, "starts");
        auto ends   = get_dense_int(*this, attrs_block, "ends");
        return mm->add_instruction(
            make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}), inputs);
    }

    // --- Reduce ops ---
    static const std::unordered_map<std::string, std::string> reduce_map = {
        {"reduce_sum",  "reduce_sum"},
        {"reduce_mean", "reduce_mean"},
        {"reduce_max",  "reduce_max"},
        {"reduce_min",  "reduce_min"},
        {"reduce_prod", "reduce_prod"},
    };
    {
        auto it = reduce_map.find(name);
        if(it != reduce_map.end())
        {
            auto axes     = get_dense_int(*this, attrs_block, "axes");
            int keep_dims = 0;
            std::string kd_str = get_attr_str(attrs_block, "keepdims");
            if(!kd_str.empty())
                keep_dims = (kd_str == "true" || kd_str == "1") ? 1 : 0;
            return mm->add_instruction(
                make_op(it->second, {{"axes", axes}, {"keepdims", keep_dims}}), inputs);
        }
    }

    // --- Squeeze / unsqueeze ---
    if(name == "squeeze")
    {
        auto axes = get_dense_int(*this, attrs_block, "axes");
        return mm->add_instruction(make_op("squeeze", {{"axes", axes}}), inputs);
    }
    if(name == "unsqueeze")
    {
        auto axes = get_dense_int(*this, attrs_block, "axes");
        return mm->add_instruction(make_op("unsqueeze", {{"axes", axes}}), inputs);
    }

    // --- Flatten ---
    if(name == "flatten")
    {
        int64_t axis = get_int(*this, attrs_block, "axis");
        return mm->add_instruction(make_op("flatten", {{"axis", axis}}), inputs);
    }

    // --- Unknown ---
    if(!opts.skip_unknown_operators)
        MIGRAPHX_THROW("DxGML: unhandled op: " + name);

    // Skip: return first input as passthrough (or literal if no inputs)
    if(inputs.empty())
        return mm->add_literal(literal{});
    return inputs[0];
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
