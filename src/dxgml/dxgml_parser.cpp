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
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>

// MLIR C API — type/attribute access
#include <mlir-c/IR.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/BuiltinAttributes.h>

#include <mutex>
#include <string>
#include <cstring>

#ifdef MIGRAPHX_DXGML_HAS_IR_LIB
// DxGML C++ dialect headers — typed attribute/type API (Option 2)
// Unwrap MlirContext (C API opaque ptr) → mlir::MLIRContext* for C++ dialect registration
#include <mlir/CAPI/IR.h>
#include <mlir/IR/MLIRContext.h>

// DxGML dialect registration
#include <dxgml/dxgml_dialect.h>      // mlir::dxgml::DxGMLDialect
#include <dxgmlOp/DxgmlOpDialect.h>   // mlir::dxgml_op::DxGMLOpDialect

// DxGML typed attribute classes
#include <dxgml/dxgml_attributes.h>   // DenseIntegerElementsAttr, IntegerAttr, ConstantResourceAttr

// DxGML typed type classes
#include <dxgml/dxgml_types.h>        // TensorType, Float16Type, Int8Type, etc.
#endif // MIGRAPHX_DXGML_HAS_IR_LIB

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// ---------------------------------------------------------------------------
// Helper: convert MlirStringRef to std::string
// ---------------------------------------------------------------------------
static std::string to_str(MlirStringRef sr)
{
    return std::string(sr.data, sr.length);
}

// ---------------------------------------------------------------------------
// Helper: print an MlirAttribute/MlirType to a std::string (fallback path)
// ---------------------------------------------------------------------------
static void print_callback(MlirStringRef chunk, void* user)
{
    static_cast<std::string*>(user)->append(chunk.data, chunk.length);
}

static std::string attr_to_string(MlirAttribute a)
{
    std::string s;
    mlirAttributePrint(a, print_callback, &s);
    return s;
}

static std::string type_to_string(MlirType t)
{
    std::string s;
    mlirTypePrint(t, print_callback, &s);
    return s;
}

// ---------------------------------------------------------------------------
// Element type → MIGraphX shape::type_t
// ---------------------------------------------------------------------------

#ifdef MIGRAPHX_DXGML_HAS_IR_LIB
// Typed path (Option 2): use llvm::isa<> on the mlir::Type (C++ wrapper of MlirType)
static shape::type_t dxgml_cpp_elem_to_migraphx(mlir::Type t)
{
    if(llvm::isa<mlir::dxgml::Float32Type>(t))  return shape::float_type;
    if(llvm::isa<mlir::dxgml::Float16Type>(t))  return shape::half_type;
    if(llvm::isa<mlir::dxgml::BFloat16Type>(t)) return shape::bf16_type;
    if(llvm::isa<mlir::dxgml::Float64Type>(t))  return shape::double_type;
    if(llvm::isa<mlir::dxgml::Int8Type>(t))     return shape::int8_type;
    if(llvm::isa<mlir::dxgml::Int16Type>(t))    return shape::int16_type;
    if(llvm::isa<mlir::dxgml::Int32Type>(t))    return shape::int32_type;
    if(llvm::isa<mlir::dxgml::Int64Type>(t))    return shape::int64_type;
    if(llvm::isa<mlir::dxgml::UInt8Type>(t))    return shape::uint8_type;
    if(llvm::isa<mlir::dxgml::UInt16Type>(t))   return shape::uint16_type;
    if(llvm::isa<mlir::dxgml::UInt32Type>(t))   return shape::uint32_type;
    if(llvm::isa<mlir::dxgml::UInt64Type>(t))   return shape::uint64_type;
    if(llvm::isa<mlir::dxgml::BoolType>(t))     return shape::bool_type;
    // Quantized/narrow float types — map to nearest supported MIGraphX type
    if(llvm::isa<mlir::dxgml::Int4Type>(t) || llvm::isa<mlir::dxgml::Int2Type>(t))
        return shape::int8_type;   // narrower int → promote to int8
    if(llvm::isa<mlir::dxgml::UInt4Type>(t) || llvm::isa<mlir::dxgml::UInt2Type>(t))
        return shape::uint8_type;
    MIGRAPHX_THROW("DxGML: unsupported element type: " + [&]{
        std::string s;
        mlirTypePrint({t.getImpl()}, print_callback, &s);
        return s;
    }());
}
#endif // MIGRAPHX_DXGML_HAS_IR_LIB

// Fallback string-based element type mapping (Option 3, no dxgml.lib)
static shape::type_t dxgml_elem_str_to_migraphx(const std::string& elem)
{
    if(elem == "!dxgml.float32" || elem == "f32")        return shape::float_type;
    if(elem == "!dxgml.float16" || elem == "f16")        return shape::half_type;
    if(elem == "!dxgml.bfloat16" || elem == "bf16")      return shape::bf16_type;
    if(elem == "!dxgml.float64" || elem == "f64")        return shape::double_type;
    if(elem == "!dxgml.int8")                            return shape::int8_type;
    if(elem == "!dxgml.int16")                           return shape::int16_type;
    if(elem == "!dxgml.int32")                           return shape::int32_type;
    if(elem == "!dxgml.int64")                           return shape::int64_type;
    if(elem == "!dxgml.uint8")                           return shape::uint8_type;
    if(elem == "!dxgml.uint16")                          return shape::uint16_type;
    if(elem == "!dxgml.uint32")                          return shape::uint32_type;
    if(elem == "!dxgml.uint64")                          return shape::uint64_type;
    if(elem == "!dxgml.bool")                            return shape::bool_type;
    MIGRAPHX_THROW("DxGML: unsupported element type string: " + elem);
}

// Parse "!dxgml.tensor<AxBxCx!dxgml.float16>" into a shape (string-based fallback)
static shape parse_dxgml_tensor_type_str(const std::string& ts)
{
    auto lt = ts.find('<');
    auto gt = ts.rfind('>');
    if(lt == std::string::npos || gt == std::string::npos || gt <= lt)
        MIGRAPHX_THROW("DxGML: malformed tensor type: " + ts);

    std::string inner = ts.substr(lt + 1, gt - lt - 1);

    std::vector<std::string> tokens;
    {
        std::string tok;
        for(char c : inner)
        {
            if(c == 'x' && !tok.empty() && tok.find('!') == std::string::npos)
            {
                tokens.push_back(tok);
                tok.clear();
            }
            else
            {
                tok += c;
            }
        }
        if(!tok.empty())
            tokens.push_back(tok);
    }

    if(tokens.size() < 2)
        MIGRAPHX_THROW("DxGML: cannot parse tensor type: " + ts);

    std::string elem = tokens.back();
    tokens.pop_back();

    std::vector<std::size_t> lens;
    for(const auto& t : tokens)
        lens.push_back(static_cast<std::size_t>(std::stoull(t)));

    return shape{dxgml_elem_str_to_migraphx(elem), lens};
}

// ---------------------------------------------------------------------------
// Type conversion from MlirType → migraphx::shape
// ---------------------------------------------------------------------------

shape dxgml_parser::mlir_type_to_shape(MlirType t) const
{
#ifdef MIGRAPHX_DXGML_HAS_IR_LIB
    // Option 2: typed DxGML C++ API — mlir::Type is a value-type wrapping the impl ptr
    mlir::Type cpp_type = mlir::Type::getFromOpaquePointer(t.ptr);
    if(auto tt = llvm::dyn_cast<mlir::dxgml::TensorType>(cpp_type))
    {
        llvm::ArrayRef<int64_t> sizes = tt.getSizes();
        std::vector<std::size_t> lens(sizes.begin(), sizes.end());
        mlir::Type dtype = tt.getDtype();
        return shape{dxgml_cpp_elem_to_migraphx(dtype), lens};
    }
    // Also handle standard MLIR RankedTensorType (unlikely in DxGML but defensive)
    if(mlirTypeIsARankedTensor(t))
    {
        intptr_t rank = mlirShapedTypeGetRank(t);
        std::vector<std::size_t> lens;
        for(intptr_t i = 0; i < rank; ++i)
            lens.push_back(static_cast<std::size_t>(mlirShapedTypeGetDimSize(t, i)));
        MlirType elem = mlirShapedTypeGetElementType(t);
        return shape{mlir_element_type_to_migraphx(elem), lens};
    }
    MIGRAPHX_THROW("DxGML: unsupported type: " + type_to_string(t));
#else
    // Option 3 fallback: string-based parsing
    if(mlirTypeIsARankedTensor(t))
    {
        intptr_t rank = mlirShapedTypeGetRank(t);
        std::vector<std::size_t> lens;
        for(intptr_t i = 0; i < rank; ++i)
            lens.push_back(static_cast<std::size_t>(mlirShapedTypeGetDimSize(t, i)));
        MlirType elem = mlirShapedTypeGetElementType(t);
        return shape{mlir_element_type_to_migraphx(elem), lens};
    }
    return parse_dxgml_tensor_type_str(type_to_string(t));
#endif
}

shape::type_t dxgml_parser::mlir_element_type_to_migraphx(MlirType elem_type) const
{
    // Standard MLIR built-in float/int types (always available via C API)
    if(mlirTypeIsAF32(elem_type))  return shape::float_type;
    if(mlirTypeIsAF16(elem_type))  return shape::half_type;
    if(mlirTypeIsABF16(elem_type)) return shape::bf16_type;
    if(mlirTypeIsAF64(elem_type))  return shape::double_type;
    if(mlirTypeIsAInteger(elem_type))
    {
        unsigned w      = mlirIntegerTypeGetWidth(elem_type);
        bool is_uns     = mlirIntegerTypeIsUnsigned(elem_type);
        if(w == 8  && !is_uns)  return shape::int8_type;
        if(w == 16 && !is_uns)  return shape::int16_type;
        if(w == 32 && !is_uns)  return shape::int32_type;
        if(w == 64 && !is_uns)  return shape::int64_type;
        if(w == 8  && is_uns)   return shape::uint8_type;
        if(w == 16 && is_uns)   return shape::uint16_type;
        if(w == 32 && is_uns)   return shape::uint32_type;
        if(w == 64 && is_uns)   return shape::uint64_type;
    }
#ifdef MIGRAPHX_DXGML_HAS_IR_LIB
    // DxGML-specific scalar types
    mlir::Type cpp_type = mlir::Type::getFromOpaquePointer(elem_type.ptr);
    return dxgml_cpp_elem_to_migraphx(cpp_type);
#else
    return dxgml_elem_str_to_migraphx(type_to_string(elem_type));
#endif
}

// ---------------------------------------------------------------------------
// Attribute helpers
// ---------------------------------------------------------------------------

std::vector<std::size_t> dxgml_parser::get_dense_int_vec(MlirAttribute a) const
{
#ifdef MIGRAPHX_DXGML_HAS_IR_LIB
    // Option 2: typed DenseIntegerElementsAttr — getValue() returns ArrayRef<int64_t>
    mlir::Attribute cpp_attr = mlir::Attribute::getFromOpaquePointer(a.ptr);
    if(auto dia = llvm::dyn_cast<mlir::dxgml::DenseIntegerElementsAttr>(cpp_attr))
    {
        llvm::ArrayRef<int64_t> vals = dia.getValue();
        std::vector<std::size_t> out;
        out.reserve(vals.size());
        for(int64_t v : vals)
            out.push_back(static_cast<std::size_t>(v));
        return out;
    }
#endif
    // Standard MLIR DenseIntElements (rare but defensive)
    if(mlirAttributeIsADenseIntElements(a))
    {
        intptr_t n = mlirElementsAttrGetNumElements(a);
        std::vector<std::size_t> out;
        out.reserve(static_cast<std::size_t>(n));
        for(intptr_t i = 0; i < n; ++i)
            out.push_back(static_cast<std::size_t>(mlirDenseElementsAttrGetInt64Value(a, i)));
        return out;
    }

    // String fallback: "#dxgml.dense_integer_elements<[2, 2]> : ..."
    std::string s = attr_to_string(a);
    auto lb = s.find('[');
    auto rb = s.find(']');
    if(lb == std::string::npos || rb == std::string::npos)
        MIGRAPHX_THROW("DxGML: cannot parse dense int vec from: " + s);

    std::string inner = s.substr(lb + 1, rb - lb - 1);
    std::vector<std::size_t> out;
    std::size_t pos = 0;
    while(pos < inner.size())
    {
        while(pos < inner.size() && (inner[pos] == ' ' || inner[pos] == ','))
            ++pos;
        if(pos >= inner.size())
            break;
        std::size_t end = pos;
        while(end < inner.size() && inner[end] != ',' && inner[end] != ' ')
            ++end;
        if(end > pos)
            out.push_back(static_cast<std::size_t>(std::stoull(inner.substr(pos, end - pos))));
        pos = end;
    }
    return out;
}

int64_t dxgml_parser::get_int_scalar(MlirAttribute a) const
{
#ifdef MIGRAPHX_DXGML_HAS_IR_LIB
    // Option 2: typed IntegerAttr — getValue() returns APInt
    mlir::Attribute cpp_attr = mlir::Attribute::getFromOpaquePointer(a.ptr);
    if(auto ia = llvm::dyn_cast<mlir::dxgml::IntegerAttr>(cpp_attr))
        return ia.getValue().getSExtValue();
#endif
    // Standard MLIR IntegerAttr
    if(mlirAttributeIsAInteger(a))
        return mlirIntegerAttrGetValueInt(a);

    // String fallback: "#dxgml.integer<1 : !dxgml.int64>"
    std::string s = attr_to_string(a);
    auto lt    = s.find('<');
    auto colon = s.find(':', lt != std::string::npos ? lt : 0);
    if(lt == std::string::npos || colon == std::string::npos)
        MIGRAPHX_THROW("DxGML: cannot parse int scalar from: " + s);

    std::string val_str = s.substr(lt + 1, colon - lt - 1);
    auto start = val_str.find_first_not_of(" \t");
    if(start == std::string::npos)
        MIGRAPHX_THROW("DxGML: empty int scalar in: " + s);
    return std::stoll(val_str.substr(start));
}

// ---------------------------------------------------------------------------
// Shared context — created once per process with DxGML dialects registered
// ---------------------------------------------------------------------------

static MlirContext get_dxgml_context()
{
    static std::once_flag flag;
    static MlirContext ctx{nullptr};
    std::call_once(flag, [] {
        ctx = mlirContextCreateWithThreading(false);
        if(!ctx.ptr)
            MIGRAPHX_THROW("DxGML: mlirContextCreate() returned null");

#ifdef MIGRAPHX_DXGML_HAS_IR_LIB
        // Option 2: register DxGML dialects so ops/types/attrs are parsed as typed objects
        mlir::MLIRContext* cpp_ctx = unwrap(ctx);
        cpp_ctx->getOrLoadDialect<mlir::dxgml::DxGMLDialect>();
        cpp_ctx->getOrLoadDialect<mlir::dxgml_op::DxGMLOpDialect>();
#else
        // Option 3 fallback: allow unknown dialects, parse as opaque strings
        mlirContextSetAllowUnregisteredDialects(ctx, true);
#endif
    });
    return ctx;
}

// ---------------------------------------------------------------------------
// Top-level parse
// ---------------------------------------------------------------------------

void dxgml_parser::parse_from_string(const std::string& mlir_text)
{
    MlirContext ctx = get_dxgml_context();
    mm              = prog.get_main_module();

    MlirModule mod =
        mlirModuleCreateParse(ctx, mlirStringRefCreate(mlir_text.data(), mlir_text.size()));
    if(mlirModuleIsNull(mod))
        MIGRAPHX_THROW("DxGML: failed to parse MLIR text");

    // Walk top-level module body looking for dxgml.entry_point
    MlirBlock body = mlirModuleGetBody(mod);
    for(MlirOperation op = mlirBlockGetFirstOperation(body); !mlirOperationIsNull(op);
        op               = mlirOperationGetNextInBlock(op))
    {
        MlirIdentifier id   = mlirOperationGetName(op);
        std::string op_name = to_str(mlirIdentifierStr(id));
        if(op_name == "dxgml.entry_point")
        {
            parse_entry_point(op);
            break;
        }
    }

    mlirModuleDestroy(mod);
}

// ---------------------------------------------------------------------------
// Entry point: register block arguments as parameters, then walk body
// ---------------------------------------------------------------------------

void dxgml_parser::parse_entry_point(MlirOperation ep)
{
    if(mlirOperationGetNumRegions(ep) == 0)
        MIGRAPHX_THROW("DxGML: entry_point has no regions");

    MlirRegion region = mlirOperationGetRegion(ep, 0);
    MlirBlock  body   = mlirRegionGetFirstBlock(region);
    if(mlirBlockIsNull(body))
        MIGRAPHX_THROW("DxGML: entry_point region has no blocks");

    // Add input parameters from entry_point block arguments
    intptr_t num_args = mlirBlockGetNumArguments(body);
    for(intptr_t i = 0; i < num_args; ++i)
    {
        MlirValue arg          = mlirBlockGetArgument(body, i);
        shape s                = mlir_type_to_shape(mlirValueGetType(arg));
        auto param_name        = "arg" + std::to_string(i);
        instruction_ref ir     = mm->add_parameter(param_name, s);
        value_map[value_id(arg)] = ir;
    }

    // Walk ops in the body block
    for(MlirOperation op = mlirBlockGetFirstOperation(body); !mlirOperationIsNull(op);
        op               = mlirOperationGetNextInBlock(op))
    {
        parse_op(op);
    }
}

// ---------------------------------------------------------------------------
// Per-op dispatch
// ---------------------------------------------------------------------------

void dxgml_parser::parse_op(MlirOperation op)
{
    std::string full_name = to_str(mlirIdentifierStr(mlirOperationGetName(op)));

    // dxgml.return — collect operands and add @return
    if(full_name == "dxgml.return")
    {
        std::vector<instruction_ref> rets;
        intptr_t n = mlirOperationGetNumOperands(op);
        for(intptr_t i = 0; i < n; ++i)
        {
            MlirValue operand = mlirOperationGetOperand(op, i);
            auto it = value_map.find(value_id(operand));
            if(it == value_map.end())
                MIGRAPHX_THROW("DxGML: undefined SSA value in dxgml.return");
            rets.push_back(it->second);
        }
        mm->add_return(rets);
        return;
    }

    // Skip other dxgml.* non-op namespace ops
    if(full_name.substr(0, 6) == "dxgml." && full_name.substr(0, 9) != "dxgml_op.")
        return;

    // dxgml_op.* ops: strip prefix and dispatch
    const std::string prefix = "dxgml_op.";
    if(full_name.substr(0, prefix.size()) != prefix)
    {
        if(!opts.skip_unknown_operators)
            MIGRAPHX_THROW("DxGML: unsupported op: " + full_name);
        return;
    }

    std::string op_name = full_name.substr(prefix.size());

    // Collect inputs from SSA operands
    std::vector<instruction_ref> inputs;
    intptr_t num_operands = mlirOperationGetNumOperands(op);
    for(intptr_t i = 0; i < num_operands; ++i)
    {
        MlirValue operand = mlirOperationGetOperand(op, i);
        auto it = value_map.find(value_id(operand));
        if(it == value_map.end())
            MIGRAPHX_THROW("DxGML: undefined SSA value for op: " + op_name);
        inputs.push_back(it->second);
    }

    // Dispatch to op-specific handler
    instruction_ref result = parse_dxgml_op(op_name, op, inputs);

    // Store result (assume single result — all tensor ops have exactly one)
    if(mlirOperationGetNumResults(op) > 0)
    {
        MlirValue res            = mlirOperationGetResult(op, 0);
        value_map[value_id(res)] = result;
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
