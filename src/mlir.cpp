
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
#include <migraphx/config.hpp>

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

    T release() const { return handle.release().get(); }

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
    mlir_program() : ctx(mlirContextCreate())
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
            std::vector<MlirNamedAttribute> attributes;
            attributes.reserve(v.size());
            std::transform(v.begin(), v.end(), std::back_inserter(attributes), [&](auto&& x) {
                return name_attribute(x.get_key(), x.without_key());
            });
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

    template <class T>
    MlirNamedAttribute name_attribute(const std::string_view& key, const T& x) const
    {
        MlirNamedAttribute attr;
        attr.name      = id(key);
        attr.attribute = attribute(x);
        return attr;
    }

    mlir_context ctx;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
