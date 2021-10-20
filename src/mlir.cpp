
#include <mlir-c/IR.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Diagnostics.h>
#include <mlir-c/Dialect/Standard.h>
#include <mlir-c/Dialect/MIGraphX.h>
#include <mlir-c/IntegerSet.h>
#include <mlir-c/Registration.h>

#include <migraphx/manage_ptr.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/config.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/iterator_for.hpp>
#include <deque>
#include <variant>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class T, class F, F f>
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

        friend bool operator!=(ptr x, ptr y) { return !(x == y); }
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

#define MIGRAPHX_MANAGE_MLIR_HANDLE(T, F) migraphx::mlir_handle<T, decltype(&F), &F> // NOLINT

using mlir_context           = MIGRAPHX_MANAGE_MLIR_HANDLE(MlirContext, mlirContextDestroy);
using mlir_module            = MIGRAPHX_MANAGE_MLIR_HANDLE(MlirModule, mlirModuleDestroy);
using mlir_operation         = MIGRAPHX_MANAGE_MLIR_HANDLE(MlirOperation, mlirOperationDestroy);
using mlir_op_printing_flags = MIGRAPHX_MANAGE_MLIR_HANDLE(MlirOpPrintingFlags,
                                                           mlirOpPrintingFlagsDestroy);
using mlir_region            = MIGRAPHX_MANAGE_MLIR_HANDLE(MlirRegion, mlirRegionDestroy);
using mlir_block             = MIGRAPHX_MANAGE_MLIR_HANDLE(MlirBlock, mlirBlockDestroy);

std::string_view to_string_view(MlirStringRef s) { return {s.data, s.length}; }

MlirStringRef make_mlir_string_ref(const std::string_view& s)
{
    return mlirStringRefCreate(s.data(), s.size());
}

template <class F, class T, class Printer>
void mlir_print(F f, T x, Printer printer)
{
    f(x,
      +[](MlirStringRef s, void* data) { (*reinterpret_cast<Printer*>(data))(to_string_view(s)); },
      &printer);
}

template <class F, class T>
void mlir_print(F f, T x, std::ostream& os)
{
    mlir_print(f, x, [&](auto s) { os << s; });
}

struct mlir_program
{
    mlir_program() : ctx(mlirContextCreate()), location(mlirLocationUnknownGet(ctx.get())), mmodule(mlirModuleCreateEmpty(location))
    {
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
        return mlirRankedTensorTypeGet(lens.size(), lens.data(), make_type(s.type()));
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
        return mlirIntegerAttrGet(mlirIntegerTypeSignedGet(ctx.get(), 64), i);
    }
    MlirAttribute attribute(std::uint64_t i) const { return attribute(std::int64_t(i)); }
    MlirAttribute attribute(unsigned char i) const { return attribute(std::int64_t(i)); }
    MlirAttribute attribute(bool b) const { return mlirBoolAttrGet(ctx.get(), b); }
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

    using attribute_t = std::variant<std::nullptr_t, std::uint64_t, unsigned char, bool, double, std::string, value, std::vector<value>, MlirType>;
    using named_attribute_t = std::pair<std::string_view, attribute_t>;

    MlirNamedAttribute name_attribute(const named_attribute_t& na) const
    {
        return name_attribute(na.first, std::visit([&](const auto& x) { return attribute(x); }, na.second));
    }

    std::vector<MlirNamedAttribute> name_attributes(const std::vector<named_attribute_t>& named_attrs) const
    {
        std::vector<MlirNamedAttribute> attributes;
        attributes.reserve(named_attrs.size());
        std::transform(named_attrs.begin(), named_attrs.end(), std::back_inserter(attributes), [&](const named_attribute_t& a) {
            return name_attribute(a);
        });
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
        {}

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
            for(auto& r:regions)
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
        auto weak_op = op.get();
        mlirBlockInsertOwnedOperation(body, 0, op.release());

        auto n = mlirOperationGetNumResults(weak_op);
        result.reserve(n);
        transform(range(n), std::back_inserter(result), [&](auto i) {
            return mlirOperationGetResult(weak_op, i);
        });
        return result;
    }

    MlirBlock insert(MlirBlock body, const module& m, std::unordered_map<instruction_ref, MlirValue>& ins_map)
    {
        auto names = m.get_parameter_names();
        std::vector<shape> inputs;
        std::transform(names.begin(), names.end(), std::back_inserter(inputs), [&](const std::string& name) {
            return m.get_parameter_shape(name);
        });
        std::vector<shape> outputs = m.get_output_shapes();

        auto body_inputs = make_tensors(inputs);
        mlir_region region = mlirRegionCreate();
        mlir_block fbody = mlirBlockCreate(body_inputs.size(), body_inputs.data());
        MlirBlock result = fbody.get();
        mlirRegionAppendOwnedBlock(region.get(), fbody.release());

        auto ops = create_operation_state("builtin.func");
        ops.add_attributes({{"type", make_function_type(inputs, outputs)}, {"sym_name", "\"main\""}});
        ops.add_region(std::move(region));
        insert(body, std::move(ops));

        for(auto i:range(names.size()))
            ins_map[m.get_parameter(names[i])] = mlirBlockGetArgument(result, i);
        return result;
    }

    void parse(const module& m)
    {
        auto mbody = mlirModuleGetBody(mmodule.get());
        std::unordered_map<instruction_ref, MlirValue> ins_map;
        auto fbody = insert(mbody, m, ins_map);
        for(auto ins:iterator_for(m))
        {
            auto name = "migraphx." + ins->name();
            auto ops = create_operation_state(name);
            ops.add_attribute_value(ins->get_operator().to_value());
            ops.add_results({ins->get_shape()});

            std::vector<MlirValue> inputs;
            transform(ins->inputs(), std::back_inserter(inputs), [&](auto i) {
                return ins_map.at(i);
            });
            ops.add_operands(inputs);

            auto outputs = insert(fbody, std::move(ops));
            assert(outputs.size() == 1);
            ins_map[ins] = outputs.front();
        }
    }

    mlir_context ctx;
    MlirLocation location;
    mlir_module mmodule;
    std::deque<std::string> strings{};
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
