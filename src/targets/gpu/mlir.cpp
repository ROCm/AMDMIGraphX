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
#include <migraphx/gpu/mlir.hpp>

#ifdef MIGRAPHX_MLIR
#include <mlir-c/IR.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Diagnostics.h>
#include <mlir-c/Dialect/MIGraphX.h>
#include <mlir-c/IntegerSet.h>
#include <mlir-c/Pass.h>
#include <mlir-c/Registration.h>
#endif

#include <migraphx/env.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/config.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/gpu/perfdb.hpp>
#include <deque>
#include <variant>

#if defined(MLIR_MIGRAPHX_DIALECT_API_VERSION) && MLIR_MIGRAPHX_DIALECT_API_VERSION >= 2
#define MIGRAPHX_MLIR_BARE_POINTER
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_MLIR);

#ifdef MIGRAPHX_MLIR
template <class T, class F, F f> // NOLINT
struct mlir_handle
{
    struct ptr
    {
        ptr() = default;
        ptr(std::nullptr_t) {}
        ptr(T x) : obj(x) {}

        std::intptr_t get_value() const
        {
            static_assert(sizeof(T) == sizeof(std::intptr_t), "MLIR Handle different size");
            return reinterpret_cast<const std::intptr_t&>(obj);
        }

        T get() const { return obj; }

        friend bool operator==(ptr x, ptr y) { return x.get_value() == y.get_value(); }

        friend bool operator!=(ptr x, ptr y) { return not(x == y); }
        T obj{};
    };

    struct deleter
    {
        using pointer = ptr;

        void operator()(pointer x) const
        {
            if(x != nullptr)
            {
                (void)f(x.obj);
            }
        }
    };

    mlir_handle() : handle(nullptr) {}

    mlir_handle(T p) : handle(ptr{p}) {}

    T get() const { return handle.get().get(); }

    T release() { return handle.release().get(); }

    private:
    std::unique_ptr<ptr, deleter> handle;
};

#define MIGRAPHX_MANAGE_MLIR_HANDLE(T, F) migraphx::gpu::mlir_handle<T, decltype(&F), &F> // NOLINT

using mlir_context           = MIGRAPHX_MANAGE_MLIR_HANDLE(MlirContext, mlirContextDestroy);
using mlir_module            = MIGRAPHX_MANAGE_MLIR_HANDLE(MlirModule, mlirModuleDestroy);
using mlir_operation         = MIGRAPHX_MANAGE_MLIR_HANDLE(MlirOperation, mlirOperationDestroy);
using mlir_op_printing_flags = MIGRAPHX_MANAGE_MLIR_HANDLE(MlirOpPrintingFlags,
                                                           mlirOpPrintingFlagsDestroy);
using mlir_region            = MIGRAPHX_MANAGE_MLIR_HANDLE(MlirRegion, mlirRegionDestroy);
using mlir_block             = MIGRAPHX_MANAGE_MLIR_HANDLE(MlirBlock, mlirBlockDestroy);
using mlir_pass_manager      = MIGRAPHX_MANAGE_MLIR_HANDLE(MlirPassManager, mlirPassManagerDestroy);

std::string_view to_string_view(MlirStringRef s) { return {s.data, s.length}; }

MlirStringRef make_mlir_string_ref(const std::string_view& s)
{
    return mlirStringRefCreate(s.data(), s.size());
}

template <class F, class T, class Printer>
void mlir_print(F f, T x, Printer printer)
{
    f(
        x,
        +[](MlirStringRef s, void* data) {
            (*reinterpret_cast<Printer*>(data))(to_string_view(s));
        },
        &printer);
}

template <class F, class T>
void mlir_print(F f, T x, std::ostream& os)
{
    mlir_print(f, x, [&](auto s) { os << s; });
}

template <class F, class T>
std::string mlir_print(F f, T x)
{
    std::stringstream ss;
    mlir_print(f, x, [&](auto s) { ss << s; });
    return ss.str();
}

const std::unordered_set<std::string>& get_xdlops_archs()
{
    static std::unordered_set<std::string> supported_archs{"gfx908", "gfx90a"};
    return supported_archs;
}

struct mlir_program
{
    mlir_program()
        : ctx(mlirContextCreate()),
          location(mlirLocationUnknownGet(ctx.get())),
          mmodule(mlirModuleCreateEmpty(location))
    {
        MlirDialectHandle mixr_handle = mlirGetDialectHandle__migraphx__();
        mlirDialectHandleRegisterDialect(mixr_handle, ctx.get());
        mlirRegisterAllDialects(ctx.get());
        mlirContextSetAllowUnregisteredDialects(ctx.get(), true /*allow*/);
    }

    MlirType make_type(shape::type_t t) const
    {
        MlirType result;
        shape::visit(t, [&](auto as) {
            if(as.type_enum() == shape::float_type)
                result = mlirF32TypeGet(ctx.get());
            else if(as.type_enum() == shape::half_type)
                result = mlirF16TypeGet(ctx.get());
            else if(as.type_enum() == shape::double_type)
                result = mlirF64TypeGet(ctx.get());
            else if(as.is_integral())
            {
                if(as.is_signed())
                    result = mlirIntegerTypeSignedGet(ctx.get(), as.size() * 8);
                else
                    result = mlirIntegerTypeGet(ctx.get(), as.size() * 8);
            }
            else
                MIGRAPHX_THROW("Unsupported type: " + std::to_string(as.type_enum()));
        });
        return result;
    }

    MlirType make_tensor(const shape& s) const
    {
        assert(s.standard());
        std::vector<int64_t> lens(s.lens().begin(), s.lens().end());
        return mlirRankedTensorTypeGet(
            lens.size(), lens.data(), make_type(s.type()), mlirAttributeGetNull());
    }

    template <class Range>
    std::vector<MlirType> make_tensors(const Range& r)
    {
        std::vector<MlirType> result;
        std::transform(r.begin(), r.end(), std::back_inserter(result), [&](const auto& s) {
            return make_tensor(s);
        });
        return result;
    }

    MlirType make_function_type(const std::vector<shape>& inputs, const std::vector<shape>& outputs)
    {
        auto in  = make_tensors(inputs);
        auto out = make_tensors(outputs);
        return mlirFunctionTypeGet(ctx.get(), in.size(), in.data(), out.size(), out.data());
    }

    MlirIdentifier id(const std::string_view& s) const
    {
        return mlirIdentifierGet(ctx.get(), make_mlir_string_ref(s));
    }

    MlirAttribute attribute(std::int64_t i) const
    {
        if(i < 0)
            MIGRAPHX_THROW("MLIR cant handle negative values since they are ambiguous");
        return mlirIntegerAttrGet(mlirIntegerTypeGet(ctx.get(), 64), i);
    }
    MlirAttribute attribute(std::uint64_t i) const
    {
        if(i > (std::numeric_limits<std::uint64_t>::max() / 2))
            MIGRAPHX_THROW("MLIR cant handle large integer values since they are ambiguous");
        return mlirIntegerAttrGet(mlirIntegerTypeGet(ctx.get(), 64), i);
    }
    MlirAttribute attribute(unsigned char i) const { return attribute(std::uint64_t(i)); }
    MlirAttribute attribute(bool b) const { return mlirBoolAttrGet(ctx.get(), b ? 1 : 0); }
    MlirAttribute attribute(double d) const
    {
        return mlirFloatAttrDoubleGet(ctx.get(), mlirF64TypeGet(ctx.get()), d);
    }
    MlirAttribute attribute(const std::string& s) const
    {
        return mlirStringAttrGet(ctx.get(), make_mlir_string_ref(s));
    }
    MlirAttribute attribute(std::nullptr_t) const { return {}; }
    template <class T>
    MlirAttribute attribute(const std::vector<T>& v) const
    {
        std::vector<MlirAttribute> attributes;
        attributes.reserve(v.size());
        std::transform(v.begin(), v.end(), std::back_inserter(attributes), [&](auto&& x) {
            return attribute(x);
        });
        return mlirArrayAttrGet(ctx.get(), attributes.size(), attributes.data());
    }
    MlirAttribute attribute(const value& v) const
    {
        MlirAttribute attr;
        v.visit_value([&](auto&& x) { attr = attribute(x); });
        return attr;
    }
    MlirAttribute attribute(const std::vector<value>& v) const
    {
        if(v.empty())
        {
            return mlirArrayAttrGet(ctx.get(), 0, nullptr);
        }
        if(not v.front().get_key().empty())
        {
            std::vector<MlirNamedAttribute> attributes = name_attributes(v);
            return mlirDictionaryAttrGet(ctx.get(), attributes.size(), attributes.data());
        }
        else
        {
            std::vector<MlirAttribute> attributes;
            attributes.reserve(v.size());
            std::transform(v.begin(), v.end(), std::back_inserter(attributes), [&](auto&& x) {
                return attribute(x);
            });
            return mlirArrayAttrGet(ctx.get(), attributes.size(), attributes.data());
        }
    }

    MlirAttribute attribute(MlirType t) const { return mlirTypeAttrGet(t); }

    MlirAttribute attribute(MlirAttribute a) const { return a; }

    template <class T>
    MlirNamedAttribute name_attribute(const std::string_view& key, const T& x) const
    {
        MlirNamedAttribute attr;
        attr.name      = id(key);
        attr.attribute = attribute(x);
        return attr;
    }

    using attribute_t       = std::variant<std::nullptr_t,
                                     std::uint64_t,
                                     unsigned char,
                                     bool,
                                     double,
                                     std::string,
                                     value,
                                     std::vector<value>,
                                     MlirType>;
    using named_attribute_t = std::pair<std::string_view, attribute_t>;

    MlirNamedAttribute name_attribute(const named_attribute_t& na) const
    {
        return name_attribute(na.first,
                              std::visit([&](const auto& x) { return attribute(x); }, na.second));
    }

    std::vector<MlirNamedAttribute>
    name_attributes(const std::vector<named_attribute_t>& named_attrs) const
    {
        std::vector<MlirNamedAttribute> attributes;
        attributes.reserve(named_attrs.size());
        std::transform(named_attrs.begin(),
                       named_attrs.end(),
                       std::back_inserter(attributes),
                       [&](const named_attribute_t& a) { return name_attribute(a); });
        return attributes;
    }

    std::vector<MlirNamedAttribute> name_attributes(const value& v) const
    {
        std::vector<MlirNamedAttribute> attributes;
        attributes.reserve(v.size());
        std::transform(v.begin(), v.end(), std::back_inserter(attributes), [&](const value& x) {
            return name_attribute(x.get_key(), x.without_key());
        });
        return attributes;
    }

    struct mlir_operation_state
    {
        mlir_operation_state(mlir_program& p, const std::string_view& name)
            : prog(&p), op_state(mlirOperationStateGet(make_mlir_string_ref(name), p.location))
        {
        }

        mlir_operation_state& add_attributes(const std::vector<named_attribute_t>& named_attrs)
        {
            auto attributes = prog->name_attributes(named_attrs);
            mlirOperationStateAddAttributes(&op_state, attributes.size(), attributes.data());
            return *this;
        }

        mlir_operation_state& add_attribute_value(const value& v)
        {
            auto attributes = prog->name_attributes(v);
            mlirOperationStateAddAttributes(&op_state, attributes.size(), attributes.data());
            return *this;
        }

        mlir_operation_state& add_regions(std::vector<mlir_region> rs)
        {
            regions = std::move(rs);
            return *this;
        }

        mlir_operation_state& add_region(mlir_region r)
        {
            regions.emplace_back(std::move(r));
            return *this;
        }

        mlir_operation_state& add_results(const std::vector<shape>& outputs)
        {
            auto x = prog->make_tensors(outputs);
            mlirOperationStateAddResults(&op_state, x.size(), x.data());
            return *this;
        }

        mlir_operation_state& add_operands(const std::vector<MlirValue>& inputs)
        {
            mlirOperationStateAddOperands(&op_state, inputs.size(), inputs.data());
            return *this;
        }

        mlir_operation create_operation()
        {
            std::vector<MlirRegion> mregions(regions.size());
            std::transform(regions.begin(), regions.end(), mregions.begin(), [](const auto& r) {
                return r.get();
            });
            mlirOperationStateAddOwnedRegions(&op_state, mregions.size(), mregions.data());
            mlir_operation op(mlirOperationCreate(&op_state));
            // Release memory since mlir_operation owns it
            for(auto& r : regions)
                r.release();
            regions.clear();
            return op;
        }

        mlir_program* prog;
        MlirOperationState op_state;
        std::vector<mlir_region> regions = {};
    };

    mlir_operation_state create_operation_state(const std::string_view& name)
    {
        return {*this, name};
    }

    std::vector<MlirValue> insert(MlirBlock body, mlir_operation_state ops)
    {
        std::vector<MlirValue> result;
        mlir_operation op = ops.create_operation();
        auto weak_op      = op.get();
        mlirBlockAppendOwnedOperation(body, op.release());

        auto n = mlirOperationGetNumResults(weak_op);
        result.reserve(n);
        transform(range(n), std::back_inserter(result), [&](auto i) {
            return mlirOperationGetResult(weak_op, i);
        });
        return result;
    }

    MlirBlock
    insert(MlirBlock body, const module& m, std::unordered_map<instruction_ref, MlirValue>& ins_map)
    {
        auto names = m.get_parameter_names();
        std::sort(names.begin(), names.end());
        std::vector<shape> inputs;
        std::transform(names.begin(),
                       names.end(),
                       std::back_inserter(inputs),
                       [&](const std::string& name) { return m.get_parameter_shape(name); });
        std::vector<shape> outputs = m.get_output_shapes();

        std::vector<MlirLocation> arg_locs(inputs.size(), location);
        auto body_inputs   = make_tensors(inputs);
        mlir_region region = mlirRegionCreate();
        mlir_block fbody = mlirBlockCreate(body_inputs.size(), body_inputs.data(), arg_locs.data());
        MlirBlock result = fbody.get();
        mlirRegionAppendOwnedBlock(region.get(), fbody.release());

        auto ops = create_operation_state("func.func");
        ops.add_attributes({{"function_type", make_function_type(inputs, outputs)},
                            {"sym_name", std::string("main")},
                            {"kernel", std::string("mixr")}});
        ops.add_region(std::move(region));
        insert(body, std::move(ops));

        for(auto i : range(names.size()))
            ins_map[m.get_parameter(names[i])] = mlirBlockGetArgument(result, i);
        return result;
    }

    static std::string get_name(instruction_ref ins)
    {
        if(ins->name() == "@return")
            return "func.return";
        return "migraphx." + ins->name();
    }

    static value get_operator_value(const operation& op)
    {
        auto v = op.to_value();
        if(op.name() == "convolution")
        {
            // Adjust symetrical padding
            if(v.at("padding").size() == v.at("stride").size())
            {
                auto padding = v.at("padding");
                std::copy(padding.begin(), padding.end(), std::back_inserter(v.at("padding")));
            }
        }
        return v;
    }

    static shape get_shape(instruction_ref ins)
    {
        if(ins->name() == "@return")
        {
            assert(ins->inputs().size() == 1);
            return ins->inputs().front()->get_shape();
        }
        return ins->get_shape();
    }

    void parse(const module& m)
    {
        auto mbody = mlirModuleGetBody(mmodule.get());
        std::unordered_map<instruction_ref, MlirValue> ins_map;
        auto fbody = insert(mbody, m, ins_map);
        for(auto ins : iterator_for(m))
        {
            if(ins->name() == "@param")
                continue;
            auto name = get_name(ins);
            auto ops  = create_operation_state(name);
            ops.add_attribute_value(get_operator_value(ins->get_operator()));
            if(ins->name() != "@return")
                ops.add_results({get_shape(ins)});
            if(ins->name() == "convolution")
            {
                pp =
                    problem_params{ins->get_operator(), to_shapes(ins->inputs()), ins->get_shape()};
                std::string tuned = get_tune_params();
                if(not tuned.empty())
                    ops.add_attributes({{"perf_config", tuned}});
                // check if HW supports xdlops
                if(contains(get_xdlops_archs(), target_name))
                    ops.add_attributes({{"xdlopsV2", true}});
            }

            std::vector<MlirValue> inputs;
            transform(
                ins->inputs(), std::back_inserter(inputs), [&](auto i) { return ins_map.at(i); });
            ops.add_operands(inputs);

            auto outputs = insert(fbody, std::move(ops));
            if(ins->name() != "@return")
            {
                assert(outputs.size() == 1);
                ins_map[ins] = outputs.front();
            }
        }
    }

    code_object_op compile() MIGRAPHX_TIDY_CONST
    {
        mlir_pass_manager pm{mlirPassManagerCreate(ctx.get())};
        // 1st pipeline to call
        mlirMIGraphXAddHighLevelPipeline(pm.get());
        // 2nd pipeline to call
        mlirMIGraphXAddBackendPipeline(pm.get(), target_name.c_str(), "amdgcn-amd-amdhsa", "");
        mlirPassManagerRun(pm.get(), mmodule.get());

        code_object_op op{};
        op.symbol_name                = "main";
        op.code_object                = get_binary();
        std::tie(op.global, op.local) = get_launch_params();
        return op;
    }

    void find_target()
    {
        std::string tname = get_device_name();
        // HACK: Since MLIR can't handle the full target name
        target_name = trim(split_string(tname, ':').front());
        if(tname.size() != target_name.size())
            std::cout
                << "*************** WARNING: MLIR may not compile the correct target features for: "
                << tname << std::endl;
    }

    std::pair<std::size_t, std::size_t> get_launch_params() const
    {
        uint32_t attrs[2];
        // returns block and grid sizes
        mlirGetKernelAttrs(mmodule.get(), attrs);
        std::size_t local  = attrs[0];
        std::size_t global = local * attrs[1];
        return {global, local};
    }

    value::binary get_binary() const
    {
        int size = 0;
        mlirGetBinary(mmodule.get(), &size, nullptr);
        value::binary result(size);
        if(mlirGetBinary(mmodule.get(), &size, reinterpret_cast<char*>(result.data())))
            return result;
        MIGRAPHX_THROW("Failed to compile mlir program");
    }

    std::string get_tune_params() { return get_mlir_perf_for_conv(pp); }

    mlir_context ctx;
    MlirLocation location;
    mlir_module mmodule;
    problem_params pp;
    std::deque<std::string> strings{};
    std::string target_name;
};

std::string dump_mlir(const module& m)
{
    mlir_program mp;
    mp.parse(m);
    auto mod_op = mlirModuleGetOperation(mp.mmodule.get());
    return mlir_print(&mlirOperationPrint, mod_op);
}

code_object_op compile_mlir(const context&, const module& m)
{
    const bool trace = enabled(MIGRAPHX_TRACE_MLIR{});
    if(trace)
        std::cout << m << std::endl;
    mlir_program mp;
    mp.find_target();
    mp.parse(m);
    auto mod_op = mlirModuleGetOperation(mp.mmodule.get());
    if(trace)
        std::cout << mlir_print(&mlirOperationPrint, mod_op) << std::endl;
    auto co   = mp.compile();
    co.output = m.get_output_shapes().front();
    return co;
}

instruction_ref insert_mlir(module& m,
                            instruction_ref ins,
                            code_object_op co,
                            const std::vector<instruction_ref>& inputs)
{

    std::vector<instruction_ref> refs;
    std::size_t last = 0;
#ifdef MIGRAPHX_MLIR_BARE_POINTER
    refs.reserve(inputs.size());
    std::copy(inputs.begin(), inputs.end(), std::back_inserter(refs));
    last = refs.size() - 1;
#else
    refs.reserve(inputs.size() * 15);
    std::unordered_map<uint64_t, instruction_ref> literal_map{};
    auto get_literal = [&](uint64_t value) {
        auto fi = literal_map.find(value);
        if(fi != literal_map.end())
            return fi->second;
        auto lit = m.add_literal(value);
        literal_map.emplace(value, lit);
        return lit;
    };

    for(auto input : inputs)
    {
        const size_t offset = 0;
        auto s              = input->get_shape();
        last                = refs.size();
        refs.push_back(input);
        refs.push_back(input);
        refs.push_back(get_literal(offset)); // offset

        // dim sizes
        std::transform(s.lens().begin(),
                       s.lens().end(),
                       std::back_inserter(refs),
                       [&](const auto& lval) { return get_literal(lval); });
        // refs.push_back(get_literal(1)); // G

        // dim strides
        std::transform(s.strides().begin(),
                       s.strides().end(),
                       std::back_inserter(refs),
                       [&](const auto& lval) { return get_literal(lval); });
        // refs.push_back(get_literal(1)); // G
    }
#endif
    co.expected_inputs = to_shapes(refs);
    co.output_arg      = last;
    return m.insert_instruction(ins, co, refs);
}

#else

std::string dump_mlir(const module&) { return {}; }

code_object_op compile_mlir(const context&, const module&) { return {}; }

template <class T>
void use(T&)
{
}

instruction_ref
// cppcheck-suppress funcArgNamesDifferent
insert_mlir(module& m, instruction_ref, code_object_op co, const std::vector<instruction_ref>&)
{
    use(co);
    return m.end();
}

#endif

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
