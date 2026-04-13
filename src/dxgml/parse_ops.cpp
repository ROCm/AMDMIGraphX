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
#include <migraphx/op/common.hpp>

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
        // Allow '#' in names to support multi-result refs like %548#0
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
                                               const std::string& type_sig,
                                               const std::string& result_base_name,
                                               int num_results)
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
    // DxGML binary ops implicitly broadcast; MIGraphX requires identical shapes.
    // If shapes differ, derive the output shape from the type signature and broadcast.
    auto broadcast_to = [&](instruction_ref in,
                             const std::vector<std::size_t>& out_lens) -> instruction_ref {
        if(in->get_shape().lens() == out_lens)
            return in;
        return mm->add_instruction(make_op("multibroadcast", {{"out_lens", out_lens}}), in);
    };
    auto get_out_lens = [&]() -> std::vector<std::size_t> {
        std::string ret = extract_ret_type(type_sig);
        if(ret.empty())
            return {};
        return parse_tensor_type(ret).lens();
    };

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
        {
            if(inputs.size() == 2 &&
               inputs[0]->get_shape().lens() != inputs[1]->get_shape().lens())
            {
                auto out_lens = get_out_lens();
                if(!out_lens.empty())
                {
                    inputs[0] = broadcast_to(inputs[0], out_lens);
                    inputs[1] = broadcast_to(inputs[1], out_lens);
                }
            }
            return mm->add_instruction(make_op(it->second), inputs);
        }
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
    // MIGraphX dot requires same ndims; broadcast batch dims if needed.
    if(name == "gemm" || name == "dot")
    {
        auto a = inputs[0];
        auto b = inputs[1];
        std::size_t na = a->get_shape().ndim();
        std::size_t nb = b->get_shape().ndim();
        if(na > nb)
        {
            // Prepend (na - nb) leading axes of size 1 to b, then broadcast to match a's batch dims
            std::vector<int64_t> new_axes;
            for(std::size_t i = 0; i < na - nb; ++i)
                new_axes.push_back(static_cast<int64_t>(i));
            b = mm->add_instruction(make_op("unsqueeze", {{"axes", new_axes}}), b);
            // Broadcast batch dims of b to match a
            auto b_lens = b->get_shape().lens();
            auto a_lens = a->get_shape().lens();
            for(std::size_t i = 0; i < na - nb; ++i)
                b_lens[i] = a_lens[i];
            b = mm->add_instruction(make_op("multibroadcast", {{"out_lens", b_lens}}), b);
        }
        else if(nb > na)
        {
            std::vector<int64_t> new_axes;
            for(std::size_t i = 0; i < nb - na; ++i)
                new_axes.push_back(static_cast<int64_t>(i));
            a = mm->add_instruction(make_op("unsqueeze", {{"axes", new_axes}}), a);
            auto a_lens = a->get_shape().lens();
            auto b_lens = b->get_shape().lens();
            for(std::size_t i = 0; i < nb - na; ++i)
                a_lens[i] = b_lens[i];
            a = mm->add_instruction(make_op("multibroadcast", {{"out_lens", a_lens}}), a);
        }
        return mm->add_instruction(make_op("dot"), a, b);
    }

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
        // DxGML uses "permutation"; fall back to "perm" for robustness.
        std::string perm_key = get_attr_str(attrs_block, "permutation").empty() ? "perm" : "permutation";
        auto perm = get_dense_int(*this, attrs_block, perm_key);
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
        // DxGML uses start_padding / end_padding; fall back to symmetric 'padding' if present.
        std::vector<std::size_t> padding;
        std::string sym_str = get_attr_str(attrs_block, "padding");
        if(!sym_str.empty())
        {
            padding = parse_dense_int_vec(sym_str);
        }
        else
        {
            auto pad_start = get_dense_int(*this, attrs_block, "start_padding");
            auto pad_end   = get_dense_int(*this, attrs_block, "end_padding");
            for(std::size_t i = 0; i < pad_start.size(); ++i)
            {
                padding.push_back(pad_start[i]);
                padding.push_back(pad_end[i]);
            }
        }
        op::pooling_mode mode =
            (name == "max_pooling") ? op::pooling_mode::max : op::pooling_mode::average;
        return mm->add_instruction(
            make_op("pooling",
                    {{"mode", mode}, {"lengths", win}, {"stride", strd}, {"padding", padding}}),
            inputs);
    }

    if(name == "global_avg_pool")
    {
        auto in_shape = inputs[0]->get_shape();
        std::vector<std::size_t> lengths(in_shape.lens().begin() + 2, in_shape.lens().end());
        return mm->add_instruction(
            make_op("pooling", {{"mode", op::pooling_mode::average}, {"lengths", lengths}}),
            inputs);
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
    // MIGraphX reduce ops always drop the reduced axes; if keepdims=1 we add unsqueeze after.
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
            auto axes_u = get_dense_int(*this, attrs_block, "axes");
            std::vector<int64_t> axes(axes_u.begin(), axes_u.end());
            auto result = mm->add_instruction(make_op(it->second, {{"axes", axes}}), inputs);
            // Determine keepdims from attr or by comparing to expected output shape.
            int keep_dims = 0;
            std::string kd_str = get_attr_str(attrs_block, "keepdims");
            if(!kd_str.empty())
                keep_dims = (kd_str == "true" || kd_str == "1") ? 1 : 0;
            else
            {
                auto out_lens = get_out_lens();
                if(!out_lens.empty() && result->get_shape().lens() != out_lens)
                    keep_dims = 1;
            }
            if(keep_dims)
                result = mm->add_instruction(make_op("unsqueeze", {{"axes", axes}}), result);
            return result;
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

    // --- Clip ---
    // MIGraphX clip takes 3 inputs: (x, min, max) — no attributes.
    // DxGML encodes min/max as float attributes; we create scalar literals and broadcast.
    if(name == "clip")
    {
        std::string min_str = get_attr_str(attrs_block, "min");
        std::string max_str = get_attr_str(attrs_block, "max");
        if(min_str.empty() || max_str.empty())
            MIGRAPHX_THROW("DxGML clip: missing min or max attribute in: " + attrs_block);
        double mn = parse_float_scalar(min_str);
        double mx = parse_float_scalar(max_str);
        auto in_shape = inputs[0]->get_shape();
        auto out_lens = in_shape.lens();
        shape::type_t dtype = in_shape.type();
        // Create scalar literals for min/max and broadcast to input shape
        auto mn_lit  = mm->add_literal(literal{shape{dtype, {1}}, {static_cast<float>(mn)}});
        auto mx_lit  = mm->add_literal(literal{shape{dtype, {1}}, {static_cast<float>(mx)}});
        auto mn_bcast = mm->add_instruction(
            make_op("multibroadcast", {{"out_lens", out_lens}}), mn_lit);
        auto mx_bcast = mm->add_instruction(
            make_op("multibroadcast", {{"out_lens", out_lens}}), mx_lit);
        return mm->add_instruction(make_op("clip"), inputs[0], mn_bcast, mx_bcast);
    }

    // --- Batch normalization ---
    // DxGML: batch_normalization(input, scale, bias, mean, variance) { epsilon }
    // Decomposes to: (input - mean) / sqrt(variance + epsilon) * scale + bias
    // Per-channel 1-D params {C} are broadcast to match the input shape {N,C,...}.
    if(name == "batch_normalization")
    {
        std::string eps_str = get_attr_str(attrs_block, "epsilon");
        if(eps_str.empty())
            MIGRAPHX_THROW("DxGML batch_normalization: missing epsilon in: " + attrs_block);
        double epsilon      = parse_float_scalar(eps_str);
        auto input          = inputs[0];
        auto scale          = inputs[1];
        auto bias           = inputs[2];
        auto mean           = inputs[3];
        auto variance       = inputs[4];
        auto in_shape       = input->get_shape();
        shape::type_t dtype = in_shape.type();
        auto in_lens        = in_shape.lens();

        // Epsilon scalar literal
        auto eps_lit = mm->add_literal(
            literal{shape{dtype, {1}}, {static_cast<float>(epsilon)}});

        // Broadcast 1-D per-channel params {C} -> {1, C, 1, 1, ...} to match input rank.
        // We use multibroadcast with out_lens = in_lens.
        auto bcast = [&](instruction_ref p) -> instruction_ref {
            auto p_shape = p->get_shape();
            if(p_shape.lens() == in_lens)
                return p; // already same shape
            // unsqueeze to {1, C, 1, 1, ...} then multibroadcast
            std::size_t rank = in_lens.size();
            std::vector<int64_t> axes;
            for(std::size_t ax = 0; ax < rank; ++ax)
                if(ax != 1) axes.push_back(static_cast<int64_t>(ax));
            auto unsqueezed = mm->add_instruction(make_op("unsqueeze", {{"axes", axes}}), p);
            return mm->add_instruction(
                make_op("multibroadcast", {{"out_lens", in_lens}}), unsqueezed);
        };

        auto scale_b = bcast(scale);
        auto bias_b  = bcast(bias);
        auto mean_b  = bcast(mean);
        // Broadcast eps scalar {1} -> {C} to match variance, then to input shape
        auto eps_c   = mm->add_instruction(
            make_op("multibroadcast", {{"out_lens", variance->get_shape().lens()}}), eps_lit);
        auto var_eps = mm->add_instruction(make_op("add"),   bcast(variance), bcast(eps_c));
        auto rsqrt   = mm->add_instruction(make_op("rsqrt"), var_eps);
        auto norm    = mm->add_instruction(make_op("sub"),   input,  mean_b);
        auto scaled  = mm->add_instruction(make_op("mul"),   norm,   rsqrt);
        auto out     = mm->add_instruction(make_op("mul"),   scaled, scale_b);
        return           mm->add_instruction(make_op("add"),   out,    bias_b);
    }

    // --- Reduce (generic, with reduction_function attr) ---
    // dxgml_op.reduce { axes, reduction_function = #dxgml_op.reduce_function_enum_attr<...> }
    if(name == "reduce")
    {
        auto axes = get_dense_int(*this, attrs_block, "axes");
        std::string rfunc = get_attr_str(attrs_block, "reduction_function");
        // Map enum name to MIGraphX op
        static const std::unordered_map<std::string, std::string> rfunc_map = {
            {"reduce_function_average", "reduce_mean"},
            {"reduce_function_sum",     "reduce_sum"},
            {"reduce_function_max",     "reduce_max"},
            {"reduce_function_min",     "reduce_min"},
            {"reduce_function_prod",    "reduce_prod"},
        };
        std::string mgx_op;
        for(const auto& kv : rfunc_map)
        {
            if(rfunc.find(kv.first) != std::string::npos)
            {
                mgx_op = kv.second;
                break;
            }
        }
        if(mgx_op.empty())
            MIGRAPHX_THROW("DxGML reduce: unknown reduction_function: " + rfunc);
        std::vector<int64_t> axes_i(axes.begin(), axes.end());
        auto result = mm->add_instruction(make_op(mgx_op, {{"axes", axes_i}}), inputs);
        // DxGML reduce always keeps dimensions (keepdims=1).
        // Only unsqueeze if needed (check against expected output shape from type sig).
        auto out_lens = get_out_lens();
        if(!out_lens.empty() && result->get_shape().lens() != out_lens)
            result = mm->add_instruction(make_op("unsqueeze", {{"axes", axes_i}}), result);
        return result;
    }

    // --- Split ---
    // dxgml_op.split(input) { axis } -> T0, T1, ..., TN
    // Registers each result as <base_name>#<i> in value_map.
    // Returns the first slice (caller skips value_map registration for multi-result).
    if(name == "split")
    {
        int64_t axis = get_int(*this, attrs_block, "axis");
        auto in_shape = inputs[0]->get_shape();
        std::size_t ax = (axis < 0)
            ? in_shape.ndim() + static_cast<std::size_t>(axis)
            : static_cast<std::size_t>(axis);
        std::size_t total = in_shape.lens()[ax];
        // Assume equal split (all output types should be uniform)
        std::size_t chunk = (num_results > 0) ? total / static_cast<std::size_t>(num_results) : total;

        instruction_ref first{};
        for(int i = 0; i < num_results; ++i)
        {
            std::size_t start = static_cast<std::size_t>(i) * chunk;
            std::size_t end   = start + chunk;
            auto slice = mm->add_instruction(
                make_op("slice", {{"axes", {ax}}, {"starts", {start}}, {"ends", {end}}}),
                inputs);
            if(i == 0)
                first = slice;
            // Register each result under "<base_name>#<i>"
            if(!result_base_name.empty())
                value_map[result_base_name + "#" + std::to_string(i)] = slice;
        }
        return first;
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
