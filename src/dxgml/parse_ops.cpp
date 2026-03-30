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

// MLIR C API — no DxGML or LLVM C++ headers needed
#include <mlir-c/IR.h>
#include <mlir-c/BuiltinAttributes.h>

#include <unordered_map>
#include <string>
#include <vector>
#include <sstream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

// Print MlirAttribute to string (defined in dxgml_parser.cpp, replicated here
// as a static to avoid exposing internal linkage across TUs).
static void print_callback_ops(MlirStringRef chunk, void* user)
{
    static_cast<std::string*>(user)->append(chunk.data, chunk.length);
}

static std::string attr_str(MlirAttribute a)
{
    std::string s;
    mlirAttributePrint(a, print_callback_ops, &s);
    return s;
}

// Get a named attribute from an op; returns null-attribute if missing.
static MlirAttribute get_attr(MlirOperation op, const char* name)
{
    return mlirOperationGetAttributeByName(op, mlirStringRefCreateFromCString(name));
}

// Get a named attribute; throw if missing.
static MlirAttribute require_attr(MlirOperation op, const char* name)
{
    MlirAttribute a = get_attr(op, name);
    if(mlirAttributeIsNull(a))
    {
        // Retrieve op name for the error message
        std::string op_name(
            mlirIdentifierStr(mlirOperationGetName(op)).data,
            mlirIdentifierStr(mlirOperationGetName(op)).length);
        MIGRAPHX_THROW("DxGML op '" + op_name + "' missing attribute '" + name + "'");
    }
    return a;
}

// ---------------------------------------------------------------------------
// Constant: #dxgml.constant_resource<name : !dxgml.tensor<...>>
//
// Format of printed attribute:
//   #dxgml.constant_resource<_conv1.weight : !dxgml.tensor<32x4x3x3x!dxgml.float16>>
//
// We expose the weight as a named parameter so callers supply data at runtime.
// ---------------------------------------------------------------------------
static instruction_ref parse_constant(dxgml_parser& self, MlirOperation op)
{
    MlirAttribute val_attr = get_attr(op, "value");
    if(mlirAttributeIsNull(val_attr))
        MIGRAPHX_THROW("DxGML: dxgml_op.constant missing 'value' attribute");

    // Print the attribute and parse its content
    // Format: #dxgml.constant_resource<NAME : TYPE>
    std::string s = attr_str(val_attr);

    // Find "constant_resource<"
    const std::string prefix = "constant_resource<";
    auto pos = s.find(prefix);
    if(pos == std::string::npos)
        MIGRAPHX_THROW("DxGML: unexpected constant attribute format: " + s);

    // Everything inside the outermost < > (accounting for nested <>)
    auto start = pos + prefix.size();
    int depth  = 1;
    auto end   = start;
    while(end < s.size() && depth > 0)
    {
        if(s[end] == '<')
            ++depth;
        else if(s[end] == '>')
            --depth;
        if(depth > 0)
            ++end;
    }
    std::string inner = s.substr(start, end - start);

    // Split at first " : " to get name and type
    const std::string sep = " : ";
    auto colon_pos = inner.find(sep);
    if(colon_pos == std::string::npos)
        MIGRAPHX_THROW("DxGML: cannot parse constant_resource name/type in: " + s);

    std::string param_name = inner.substr(0, colon_pos);
    std::string type_str   = inner.substr(colon_pos + sep.size());

    // type_str is something like "!dxgml.tensor<32x4x3x3x!dxgml.float16>"
    shape sh = self.mlir_type_to_shape(mlirValueGetType(mlirOperationGetResult(op, 0)));

    // If we couldn't derive shape from the result type (DxGML opaque type),
    // parse the type string directly.  The result type IS the tensor type.
    // Use the shape from the result value type — that's the authoritative source.
    return self.mm->add_parameter(param_name, sh);
}

// ---------------------------------------------------------------------------
// Main op dispatch
// ---------------------------------------------------------------------------

instruction_ref dxgml_parser::parse_dxgml_op(const std::string& name,
                                               MlirOperation op,
                                               const std::vector<instruction_ref>& inputs)
{
    // --- Constant (weight) ---
    if(name == "constant")
        return parse_constant(*this, op);

    // --- Unary elementwise ---
    static const std::unordered_map<std::string, std::string> unary_map = {
        {"relu", "relu"},
        {"sigmoid", "sigmoid"},
        {"tanh", "tanh"},
        {"erf", "erf"},
        {"exp", "exp"},
        {"log", "log"},
        {"sqrt", "sqrt"},
        {"abs", "abs"},
        {"ceil", "ceil"},
        {"floor", "floor"},
        {"neg", "neg"},
        {"rsqrt", "rsqrt"},
        {"recip", "recip"},
    };
    {
        auto it = unary_map.find(name);
        if(it != unary_map.end())
            return mm->add_instruction(make_op(it->second), inputs);
    }

    // --- Binary elementwise ---
    static const std::unordered_map<std::string, std::string> binary_map = {
        {"add", "add"},
        {"subtract", "sub"},
        {"multiply", "mul"},
        {"divide", "div"},
        {"pow", "pow"},
        {"max", "max"},
        {"min", "min"},
    };
    {
        auto it = binary_map.find(name);
        if(it != binary_map.end())
            return mm->add_instruction(make_op(it->second), inputs);
    }

    // --- Convolution ---
    if(name == "convolution")
    {
        auto strides   = get_dense_int_vec(require_attr(op, "strides"));
        auto dilations = get_dense_int_vec(require_attr(op, "dilations"));
        auto pad_start = get_dense_int_vec(require_attr(op, "start_padding"));
        auto pad_end   = get_dense_int_vec(require_attr(op, "end_padding"));
        auto groups    = get_int_scalar(require_attr(op, "group_count"));

        // MIGraphX interleaved padding: [top, bottom, left, right] for 2D
        // DxGML: start_padding=[top,left], end_padding=[bottom,right]
        std::vector<std::size_t> padding;
        for(std::size_t i = 0; i < pad_start.size(); ++i)
        {
            padding.push_back(pad_start[i]);
            padding.push_back(pad_end[i]);
        }

        return mm->add_instruction(
            make_op("convolution",
                    {{"stride", strides},
                     {"dilation", dilations},
                     {"padding", padding},
                     {"group", static_cast<int>(groups)}}),
            inputs);
    }

    // --- Gemm / dot ---
    if(name == "gemm" || name == "dot")
        return mm->add_instruction(make_op("dot"), inputs);

    // --- Reshape ---
    if(name == "reshape")
    {
        MlirValue res       = mlirOperationGetResult(op, 0);
        shape out_shape     = mlir_type_to_shape(mlirValueGetType(res));
        return mm->add_instruction(
            make_op("reshape", {{"dims", out_shape.lens()}}), inputs);
    }

    // --- Transpose ---
    if(name == "transpose")
    {
        auto perm = get_dense_int_vec(require_attr(op, "perm"));
        std::vector<int64_t> permutation(perm.begin(), perm.end());
        return mm->add_instruction(
            make_op("transpose", {{"permutation", permutation}}), inputs);
    }

    // --- Cast / convert ---
    if(name == "cast")
    {
        MlirValue res    = mlirOperationGetResult(op, 0);
        shape out_shape  = mlir_type_to_shape(mlirValueGetType(res));
        return mm->add_instruction(
            make_op("convert", {{"target_type", out_shape.type()}}), inputs);
    }

    // --- Softmax ---
    if(name == "softmax" || name == "log_softmax")
    {
        int64_t axis         = get_int_scalar(require_attr(op, "axis"));
        std::string mgx_name = (name == "softmax") ? "softmax" : "logsoftmax";
        return mm->add_instruction(make_op(mgx_name, {{"axis", axis}}), inputs);
    }

    // --- Pooling ---
    if(name == "max_pooling" || name == "average_pooling")
    {
        auto win  = get_dense_int_vec(require_attr(op, "window_size"));
        auto strd = get_dense_int_vec(require_attr(op, "strides"));
        auto pad  = get_dense_int_vec(require_attr(op, "padding"));
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
        int64_t axis = get_int_scalar(require_attr(op, "axis"));
        return mm->add_instruction(make_op("concat", {{"axis", axis}}), inputs);
    }

    // --- Slice ---
    if(name == "slice")
    {
        auto axes   = get_dense_int_vec(require_attr(op, "axes"));
        auto starts = get_dense_int_vec(require_attr(op, "starts"));
        auto ends   = get_dense_int_vec(require_attr(op, "ends"));
        return mm->add_instruction(
            make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}), inputs);
    }

    // --- Reduce ops ---
    static const std::unordered_map<std::string, std::string> reduce_map = {
        {"reduce_sum", "reduce_sum"},
        {"reduce_mean", "reduce_mean"},
        {"reduce_max", "reduce_max"},
        {"reduce_min", "reduce_min"},
        {"reduce_prod", "reduce_prod"},
    };
    {
        auto it = reduce_map.find(name);
        if(it != reduce_map.end())
        {
            auto axes = get_dense_int_vec(require_attr(op, "axes"));
            int keep_dims = 0;
            MlirAttribute ka = get_attr(op, "keepdims");
            if(!mlirAttributeIsNull(ka) && mlirAttributeIsABool(ka))
                keep_dims = mlirBoolAttrGetValue(ka) ? 1 : 0;
            return mm->add_instruction(
                make_op(it->second, {{"axes", axes}, {"keepdims", keep_dims}}), inputs);
        }
    }

    // --- Squeeze / unsqueeze ---
    if(name == "squeeze")
    {
        auto axes = get_dense_int_vec(require_attr(op, "axes"));
        return mm->add_instruction(make_op("squeeze", {{"axes", axes}}), inputs);
    }
    if(name == "unsqueeze")
    {
        auto axes = get_dense_int_vec(require_attr(op, "axes"));
        return mm->add_instruction(make_op("unsqueeze", {{"axes", axes}}), inputs);
    }

    // --- Flatten ---
    if(name == "flatten")
    {
        int64_t axis = get_int_scalar(require_attr(op, "axis"));
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
