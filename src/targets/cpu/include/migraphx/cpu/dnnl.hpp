#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_DNNL_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_DNNL_HPP

#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/check_shapes.hpp>
#include <unordered_map>
#include <dnnl.hpp>
#include <migraphx/errors.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct dnnl_context
{
    dnnl::engine engine;
    dnnl::stream stream;
    dnnl_context() : engine(dnnl::engine::kind::cpu, 0), stream(engine) {}
};

inline dnnl_context& get_dnnl_context()
{
    static dnnl_context ctx{}; // NOLINT
    return ctx;
}
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-enum"
#endif
inline dnnl::memory::data_type to_dnnl_memory_data_type(shape::type_t t)
{
    using dt = dnnl::memory::data_type;
    using st = shape::type_t;
    switch(t)
    {
    case st::half_type: return dt::f16;
    case st::float_type: return dt::f32;
    case st::int32_type: return dt::s32;
    case st::int8_type: return dt::s8;
    case st::uint8_type: return dt::u8;
    default: MIGRAPHX_THROW("Unsupported data type");
    }
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif

inline dnnl::memory::format_tag to_dnnl_memory_format_tag(std::size_t n)
{
    switch(n)
    {
    case 1: return dnnl::memory::format_tag::a;
    case 2: return dnnl::memory::format_tag::ab;
    case 3: return dnnl::memory::format_tag::abc;
    case 4: return dnnl::memory::format_tag::abcd;
    case 5: return dnnl::memory::format_tag::abcde;
    case 6: return dnnl::memory::format_tag::abcdef;
    default: MIGRAPHX_THROW("Unsupported tensor size: " + std::to_string(n));
    }
}

template <class R>
inline dnnl::memory::dims to_dnnl_dims(R&& r)
{
    return {r.begin(), r.end()};
}

inline dnnl::memory::desc to_dnnl_memory_desc(const shape& s)
{
    return {to_dnnl_dims(s.lens()), to_dnnl_memory_data_type(s.type()), to_dnnl_dims(s.strides())};
}

inline dnnl::memory to_dnnl_memory(const dnnl::memory::desc& desc, const argument& a)
{
    return dnnl::memory(desc, get_dnnl_context().engine, a.data());
}

inline dnnl::memory to_dnnl_memory(const argument& a)
{
    return to_dnnl_memory(to_dnnl_memory_desc(a.get_shape()), a);
}

template <class Derived, class Primitive>
struct dnnl_op : auto_register_op<Derived>
{
    std::function<argument(context& ctx, const std::vector<argument>& args)> execute;

    static std::vector<shape> to_shapes(const std::vector<argument>& args)
    {
        std::vector<shape> shapes(args.size());
        std::transform(args.begin(), args.end(), shapes.begin(), [](const argument& a) {
            return a.get_shape();
        });
        return shapes;
    }
    // Map arg index to arg in dnnl
    std::vector<int> arg_map(int size) const
    {
        std::vector<int> result(size);
        std::iota(result.begin(), result.end(), DNNL_ARG_SRC_0);
        return result;
    }
    shape base_adjust_shape(const shape& s) const
    {
        if(s.broadcasted())
        {
            auto lens    = s.lens();
            auto strides = s.strides();
            std::transform(strides.begin(),
                           strides.end(),
                           lens.begin(),
                           lens.begin(),
                           [](auto stride, auto len) -> std::size_t {
                               if(stride == 0)
                                   return 1;
                               else
                                   return len;
                           });
            return shape{s.type(), lens};
        }
        return s;
    }
    shape adjust_shape(const shape& s, int) const { return base_adjust_shape(s); }
    std::unordered_map<int, dnnl::memory::desc>
    to_memory_desc(const shape& output_shape, const std::vector<shape>& inputs) const
    {
        const auto& self = static_cast<const Derived&>(*this);
        std::unordered_map<int, dnnl::memory::desc> result;
        result[DNNL_ARG_DST] = to_dnnl_memory_desc(self.adjust_shape(output_shape, inputs.size()));
        auto m               = self.arg_map(inputs.size());
        assert(m.size() >= inputs.size());
        for(int i = 0; i < inputs.size(); i++)
        {
            result[m[i]] = to_dnnl_memory_desc(self.adjust_shape(inputs[i], i));
        }
        return result;
    }
    template <class T>
    auto get_primitive_desc(const T& desc) const
        -> decltype(typename Primitive::primitive_desc(desc, get_dnnl_context().engine))
    {
        return typename Primitive::primitive_desc(desc, get_dnnl_context().engine);
    }
    Primitive get_primitive(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        const auto& self = static_cast<const Derived&>(*this);
        auto desc        = self.get_desc(m);
        auto pd          = self.get_primitive_desc(desc);
        return Primitive(pd);
    }
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        return execute(ctx, args);
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }

    void finalize(context&, const shape& output_shape, std::vector<shape> inputs)
    {
        // Compensate for allocation
        inputs.pop_back();
        const auto& self = static_cast<const Derived&>(*this);
        auto name        = self.name();
        auto md          = to_memory_desc(output_shape, inputs);
        auto prim        = get_primitive(md);
        auto arg_lookup  = self.arg_map(inputs.size());
        execute          = [=](context&, const std::vector<argument>& args) {
#ifndef NDEBUG
            // Check that the memory descriptors have not changed
            auto debug_args = args;
            debug_args.pop_back();
            auto debug_md = to_memory_desc(output_shape, to_shapes(debug_args));
            for(auto&& p : debug_md)
            {
                if(md.count(p.first) == 0)
                    MIGRAPHX_THROW(name +
                                   ": Missing memory descriptor for: " + std::to_string(p.first));
                if(p.second == md.at(p.first))
                    continue;
                MIGRAPHX_THROW(name +
                               ": Memory descriptor has changed for: " + std::to_string(p.first));
            }
#endif
            std::unordered_map<int, dnnl::memory> m;
            m[DNNL_ARG_DST] = to_dnnl_memory(md.at(DNNL_ARG_DST), args.back());
            for(int i = 0; i < args.size() - 1; i++)
                m[arg_lookup[i]] = to_dnnl_memory(md.at(arg_lookup[i]), args[i]);
            prim.execute(get_dnnl_context().stream, m);
            return args.back();
        };
    }
};

template <class Derived, class Primitive, class Op>
struct dnnl_extend_op : dnnl_op<Derived, Primitive>
{
    Op op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "dnnl::" + op.name(); }
    shape compute_shape(std::vector<shape> inputs) const
    {
        // Compensate for allocation
        inputs.pop_back();
        // check_shapes(inputs, *this).standard();
        auto r = migraphx::compute_shape(op, inputs);
        // Call to get_primitive to make sure an algo is available
        this->get_primitive(this->to_memory_desc(r, inputs));
        return r;
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
