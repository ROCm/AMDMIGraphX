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
#include <sstream>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static std::string trim(const std::string& s)
{
    auto b = s.find_first_not_of(" \t\r\n");
    if(b == std::string::npos) return {};
    auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

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
//        or dxgml_op.constant(#dxgml.dense_integer_elements<[v0,v1,...]> : TYPE)
// ---------------------------------------------------------------------------
static instruction_ref parse_constant(dxgml_parser& self, const std::string& operands_raw)
{
    // --- Handle dense_integer_elements inline constant ---
    const std::string dense_pfx = "dense_integer_elements<";
    {
        auto dp = operands_raw.find(dense_pfx);
        if(dp != std::string::npos)
        {
            auto list_start = dp + dense_pfx.size();
            // Find '[' ... ']'
            auto lb = operands_raw.find('[', list_start);
            auto rb = operands_raw.find(']', lb != std::string::npos ? lb : list_start);
            std::vector<int64_t> vals;
            if(lb != std::string::npos && rb != std::string::npos)
            {
                std::string list = operands_raw.substr(lb + 1, rb - lb - 1);
                std::istringstream ss(list);
                std::string tok;
                while(std::getline(ss, tok, ','))
                {
                    tok = trim(tok);
                    if(!tok.empty())
                        vals.push_back(std::stoll(tok));
                }
            }
            // Find type after '>'  ':'  TYPE
            auto colon_pos = operands_raw.find(':', rb != std::string::npos ? rb : list_start);
            if(colon_pos != std::string::npos)
            {
                std::string type_str = trim(operands_raw.substr(colon_pos + 1));
                try
                {
                    shape sh = self.parse_tensor_type(type_str);
                    return self.mm->add_literal(literal{sh, vals.begin(), vals.end()});
                }
                catch(...)
                {
                }
            }
            // Fallback: return 1D int64 literal
            if(!vals.empty())
            {
                shape sh{shape::int64_type, {vals.size()}};
                return self.mm->add_literal(literal{sh, vals.begin(), vals.end()});
            }
            return self.mm->add_literal(literal{});
        }
    }

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

static std::vector<std::size_t>
get_dense_int_or(dxgml_parser& self,
                 const std::string& attrs,
                 const std::string& key,
                 std::vector<std::size_t> default_val)
{
    std::string val = self.get_attr_str(attrs, key);
    if(val.empty())
        return default_val;
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

static int64_t
get_int_or(dxgml_parser& self, const std::string& attrs, const std::string& key, int64_t default_val)
{
    std::string val = self.get_attr_str(attrs, key);
    if(val.empty())
        return default_val;
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

    // Returns true if `in_lens` can be numpy-broadcast to `out_lens`.
    auto is_broadcastable = [](const std::vector<std::size_t>& in_lens,
                                const std::vector<std::size_t>& out_lens) -> bool {
        if(in_lens.size() > out_lens.size())
            return false;
        std::size_t off = out_lens.size() - in_lens.size();
        for(std::size_t i = 0; i < in_lens.size(); ++i)
        {
            if(in_lens[i] != 1 && in_lens[i] != out_lens[off + i])
                return false;
        }
        return true;
    };

    auto broadcast_to = [&](instruction_ref in,
                             const std::vector<std::size_t>& out_lens) -> instruction_ref {
        if(in->get_shape().lens() == out_lens)
            return in;
        if(!is_broadcastable(in->get_shape().lens(), out_lens))
            return in; // cannot broadcast; return as-is
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
            if(inputs.size() == 2)
            {
                // Shape broadcast: if dims differ, try to broadcast to output shape
                if(inputs[0]->get_shape().lens() != inputs[1]->get_shape().lens())
                {
                    auto out_lens = get_out_lens();
                    if(!out_lens.empty())
                    {
                        inputs[0] = broadcast_to(inputs[0], out_lens);
                        inputs[1] = broadcast_to(inputs[1], out_lens);
                    }
                }
                // Type coercion: MIGraphX requires matching element types
                if(inputs[0]->get_shape().type() != inputs[1]->get_shape().type())
                    inputs[1] = mm->add_instruction(
                        make_op("convert", {{"target_type", inputs[0]->get_shape().type()}}),
                        inputs[1]);
            }
            return mm->add_instruction(make_op(it->second), inputs);
        }
    }

    // --- Convolution ---
    if(name == "convolution")
    {
        auto strides   = get_dense_int(*this, attrs_block, "strides");
        auto pad_start = get_dense_int(*this, attrs_block, "start_padding");
        auto pad_end   = get_dense_int(*this, attrs_block, "end_padding");
        // dilations and group_count are optional
        std::vector<std::size_t> default_dil(strides.size(), 1);
        auto dilations = get_dense_int_or(*this, attrs_block, "dilations", default_dil);
        auto groups    = get_int_or(*this, attrs_block, "group_count", 1);

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
        // MIGraphX dot requires matching types; convert weight to activation type if needed
        if(a->get_shape().type() != b->get_shape().type())
            b = mm->add_instruction(
                make_op("convert", {{"target_type", a->get_shape().type()}}), b);
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

    // --- null_ptr: DxGML optional/null sentinel ---
    if(name == "null_ptr")
        return mm->add_literal(literal{});

    // --- shape ---
    // dxgml_op.shape(%x) -> !dxgml.tensor<Nx!dxgml.int64>
    // Returns a 1-D int64 tensor containing the runtime shape of the input.
    // Map to MIGraphX dimensions_of which returns all dims as a 1-D int64 tensor.
    if(name == "shape")
        return mm->add_instruction(make_op("dimensions_of", {{"start", 0}, {"end", static_cast<int64_t>(inputs[0]->get_shape().ndim())}}), inputs[0]);

    // --- depth_to_space ---
    // dxgml_op.depth_to_space(%x) { block_size, depth_space_order } -> T
    // Decomposes into reshape -> transpose -> reshape (same as ONNX DepthToSpace).
    // DxGML orders: depth_space_order_column_row_depth  -> DCR perm {0,3,4,1,5,2}
    //               depth_space_order_row_column_depth  -> CRD perm {0,1,4,2,5,3}
    if(name == "depth_to_space")
    {
        int64_t blocksize = get_int(*this, attrs_block, "block_size");
        std::string order = get_attr_str(attrs_block, "depth_space_order");
        auto s            = inputs[0]->get_shape();
        auto lens1        = s.lens();
        auto lens2        = s.lens();
        std::size_t divisor = static_cast<std::size_t>(blocksize * blocksize);
        if((lens2[1] % divisor) != 0)
            MIGRAPHX_THROW("DxGML depth_to_space: channels not divisible by block_size^2");
        lens2[1] /= divisor;
        lens2[2] = lens2[2] * static_cast<std::size_t>(blocksize);
        lens2[3] = lens2[3] * static_cast<std::size_t>(blocksize);
        lens1.push_back(lens1[2]);
        lens1.push_back(lens1[3]);
        lens1[2] = static_cast<std::size_t>(blocksize);
        std::vector<int64_t> perm;
        // column_row_depth = DCR: expand as (N, bs, bs, C/(bs^2), H, W), perm 0,3,4,1,5,2
        if(order.find("column_row_depth") != std::string::npos)
        {
            lens1[3] = lens1[1] / divisor;
            lens1[1] = static_cast<std::size_t>(blocksize);
            perm     = {0, 3, 4, 1, 5, 2};
        }
        // row_column_depth = CRD: expand as (N, C/(bs^2), bs, bs, H, W), perm 0,1,4,2,5,3
        else if(order.find("row_column_depth") != std::string::npos)
        {
            lens1[1] /= divisor;
            lens1[3] = static_cast<std::size_t>(blocksize);
            perm     = {0, 1, 4, 2, 5, 3};
        }
        else
            MIGRAPHX_THROW("DxGML depth_to_space: unknown order: " + order);
        auto r1 = mm->add_instruction(make_op("reshape", {{"dims", lens1}}), inputs[0]);
        auto r2 = mm->add_instruction(make_op("transpose", {{"permutation", perm}}), r1);
        return mm->add_instruction(make_op("reshape", {{"dims", lens2}}), r2);
    }

    // --- dequantize_linear ---
    // dxgml_op.dequantize_linear(x, scale [, zero_point]) { axis, block_size } -> T
    // Maps directly to MIGraphX dequantizelinear (2 or 3 inputs).
    // Note: block_size is a DxGML attribute that the MIGraphX op handles implicitly via
    // the scale tensor shape; no extra attribute needed on the MIGraphX side.
    if(name == "dequantize_linear")
        return mm->add_instruction(make_op("dequantizelinear"), inputs);

    // --- rotary_embedding ---
    // dxgml_op.rotary_embedding(input, cos_cache, sin_cache, pos_ids)
    //   { interleaved, num_heads, rotary_embedding_dim }
    // input    : (B, S, num_heads * head_size)  [BSH layout]
    // cos/sin  : (max_seq_len, rotary_embedding_dim/2)
    // pos_ids  : (B, S) int64
    // output   : same shape as input
    //
    // Decomposition (non-interleaved, matching ONNX RotaryEmbedding parser):
    //   1. Gather cos/sin rows indexed by pos_ids -> (B, S, rot_dim/2)
    //   2. Repeat last dim: concat(cache, cache) -> (B, S, rot_dim)
    //   3. Broadcast to (B, num_heads, S, rot_dim)
    //   4. split input into rotary part and tail (if head_dim > rot_dim)
    //   5. neg_half = concat(-x[rot/2:rot], x[0:rot/2])  (90-degree rotation)
    //   6. out = x * cos + neg_half * sin
    //   7. concat rotary result with tail, reshape back to BSH
    if(name == "rotary_embedding")
    {
        auto input      = inputs[0]; // (B, S, H)
        auto cos_cache  = inputs[1]; // (max_seq, rot_dim/2)
        auto sin_cache  = inputs[2]; // (max_seq, rot_dim/2)
        auto pos_ids    = inputs[3]; // (B, S)

        int64_t num_heads   = get_int(*this, attrs_block, "num_heads");
        int64_t rot_dim     = get_int(*this, attrs_block, "rotary_embedding_dim");
        int64_t interleaved = get_int(*this, attrs_block, "interleaved");

        auto in_lens    = input->get_shape().lens();
        int64_t B       = static_cast<int64_t>(in_lens[0]);
        int64_t S       = static_cast<int64_t>(in_lens[1]);
        int64_t H       = static_cast<int64_t>(in_lens[2]);
        int64_t head_sz = H / num_heads;

        // Reshape input to (B, num_heads, S, head_sz) for per-head processing
        auto x = mm->add_instruction(
            make_op("reshape", {{"dims", {B, num_heads, S, head_sz}}}), input);

        // Slice rotary and tail parts along head_sz axis
        auto x_rot = mm->add_instruction(
            make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {rot_dim}}}), x);
        instruction_ref x_tail;
        bool has_tail = (head_sz > rot_dim);
        if(has_tail)
            x_tail = mm->add_instruction(
                make_op("slice", {{"axes", {3}}, {"starts", {rot_dim}}, {"ends", {head_sz}}}), x);

        // Gather cache rows: pos_ids (B,S) -> indices into cache (max_seq, rot_dim/2)
        // Reshape pos_ids to (B, S, 1) for gathernd
        auto pos_rs = mm->add_instruction(
            make_op("reshape", {{"dims", {B, S, 1}}}), pos_ids);
        auto cos_g = mm->add_instruction(make_op("gathernd", {{"batch_dims", 0}}), cos_cache, pos_rs);
        auto sin_g = mm->add_instruction(make_op("gathernd", {{"batch_dims", 0}}), sin_cache, pos_rs);
        // cos_g / sin_g: (B, S, rot_dim/2) — duplicate to (B, S, rot_dim)
        cos_g = mm->add_instruction(make_op("concat", {{"axis", 2}}), cos_g, cos_g);
        sin_g = mm->add_instruction(make_op("concat", {{"axis", 2}}), sin_g, sin_g);
        // Reshape to (B, 1, S, rot_dim) and broadcast to (B, num_heads, S, rot_dim)
        cos_g = mm->add_instruction(make_op("reshape", {{"dims", {B, 1, S, rot_dim}}}), cos_g);
        sin_g = mm->add_instruction(make_op("reshape", {{"dims", {B, 1, S, rot_dim}}}), sin_g);
        cos_g = mm->add_instruction(
            make_op("multibroadcast", {{"out_lens", {B, num_heads, S, rot_dim}}}), cos_g);
        sin_g = mm->add_instruction(
            make_op("multibroadcast", {{"out_lens", {B, num_heads, S, rot_dim}}}), sin_g);

        instruction_ref rot_out;
        if(interleaved)
        {
            // Interleaved: pair (x[0],x[1]), (x[2],x[3])... rotate each pair
            // neg_half: reshape to (..., rot_dim/2, 2), swap, neg odds, flatten back
            auto rs = mm->add_instruction(
                make_op("reshape", {{"dims", {B, num_heads, S, rot_dim / 2, 2}}}), x_rot);
            auto evens = mm->add_instruction(
                make_op("slice", {{"axes", {4}}, {"starts", {0}}, {"ends", {1}}}), rs);
            auto odds  = mm->add_instruction(
                make_op("slice", {{"axes", {4}}, {"starts", {1}}, {"ends", {2}}}), rs);
            auto neg_odds = mm->add_instruction(make_op("neg"), odds);
            auto swapped  = mm->add_instruction(make_op("concat", {{"axis", 4}}), neg_odds, evens);
            auto neg_half = mm->add_instruction(
                make_op("reshape", {{"dims", {B, num_heads, S, rot_dim}}}), swapped);
            auto mul_cos  = mm->add_instruction(make_op("mul"), x_rot, cos_g);
            auto mul_sin  = mm->add_instruction(make_op("mul"), neg_half, sin_g);
            rot_out = mm->add_instruction(make_op("add"), mul_cos, mul_sin);
        }
        else
        {
            // Non-interleaved: neg_half = concat(-x[rot/2:rot], x[0:rot/2])
            auto x_pos = mm->add_instruction(
                make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {rot_dim / 2}}}), x_rot);
            auto x_neg = mm->add_instruction(
                make_op("slice", {{"axes", {3}}, {"starts", {rot_dim / 2}}, {"ends", {rot_dim}}}), x_rot);
            auto neg_x_neg = mm->add_instruction(make_op("neg"), x_neg);
            auto neg_half  = mm->add_instruction(make_op("concat", {{"axis", 3}}), neg_x_neg, x_pos);
            auto mul_cos   = mm->add_instruction(make_op("mul"), x_rot, cos_g);
            auto mul_sin   = mm->add_instruction(make_op("mul"), neg_half, sin_g);
            rot_out = mm->add_instruction(make_op("add"), mul_cos, mul_sin);
        }

        // Reassemble: concat rotary result with tail (if any)
        instruction_ref full;
        if(has_tail)
            full = mm->add_instruction(make_op("concat", {{"axis", 3}}), rot_out, x_tail);
        else
            full = rot_out;

        // Reshape back to BSH layout (B, S, H)
        return mm->add_instruction(make_op("reshape", {{"dims", {B, S, H}}}), full);
    }

    // --- Unknown ---
    if(!opts.skip_unknown_operators)
        MIGRAPHX_THROW("DxGML: unhandled op: " + name);

    // Skip: return a placeholder with the declared output shape when possible,
    // so downstream shape/type checks don't fail.
    {
        std::string ret_type_str = extract_ret_type(type_sig);
        if(!ret_type_str.empty())
        {
            try
            {
                shape out_shape = parse_tensor_type(ret_type_str);
                // Use the first input if its shape already matches, else add a
                // fresh parameter so the correct shape propagates downstream.
                if(!inputs.empty() &&
                   out_shape.lens() == inputs[0]->get_shape().lens() &&
                   out_shape.type() == inputs[0]->get_shape().type())
                    return inputs[0];
                static std::size_t skip_counter = 0;
                std::string ph_name =
                    "__skipped_" + name + "_" + std::to_string(skip_counter++);
                return mm->add_parameter(ph_name, out_shape);
            }
            catch(...)
            {
            }
        }
    }
    if(inputs.empty())
        return mm->add_literal(literal{});
    return inputs[0];
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
