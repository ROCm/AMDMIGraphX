#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_DNNL_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_DNNL_HPP

#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/check_shapes.hpp>
#include <unordered_map>
#ifdef USE_DNNL
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
    static dnnl_context ctx{};
    return ctx;
}

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
    if(not s.standard())
        MIGRAPHX_THROW("Unsupported layout");
    return dnnl::memory::desc(to_dnnl_dims(s.lens()),
                              to_dnnl_memory_data_type(s.type()),
                              to_dnnl_memory_format_tag(s.lens().size()));
}

inline dnnl::memory to_dnnl_memory(const dnnl::memory::desc& desc, const argument& a)
{
    return dnnl::memory(desc, get_dnnl_context().engine, a.data());
}

inline dnnl::memory to_dnnl_memory(const argument& a)
{
    return to_dnnl_memory(to_dnnl_memory_desc(a.get_shape()), a);
}

// Remove
inline dnnl::memory to_dnnl_memory(const argument& a, const dnnl::engine& engine)
{
    return dnnl::memory(to_dnnl_memory_desc(a.get_shape()), engine, a.data());
}

template <class Primitive, class Context>
auto execute_dnnl(Context& ctx, std::unordered_map<int, argument> args)
{
    using primitive_desc = typename Primitive::primitive_desc;
    return [&ctx, args](auto f) {
        std::unordered_map<int, dnnl::memory> m;
        for(auto&& p : args)
            m[p.first] = to_dnnl_memory(p.second, get_dnnl_context().engine);
        auto desc = f(m);
        auto pd   = primitive_desc(desc, get_dnnl_context().engine);
        auto prim = Primitive(pd);
        prim.execute(get_dnnl_context().stream, m);
    };
}

template<class Derived, class Primitive, class Op>
struct dnnl_op : auto_register_op<Derived>
{
    Op op;
    // std::function<argument(const std::vector<argument>&)> execute;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }
    static std::vector<shape> to_shapes(const std::vector<argument>& args)
    {
        std::vector<shape> shapes(args.size());
        std::transform(
            args.begin(), args.end(), shapes.begin(), [](argument a) { return a.get_shape(); });
        return shapes;
    }
    // Map arg index to arg in dnnl
    std::vector<int> arg_map(int size) const
    {
        std::vector<int> result(size);
        std::iota(result.begin(), result.end(), 0);
        return result;
    }
    shape base_adjust_shape(const shape& s) const
    {
        // TODO: Compensate for broadcast shapes
        return s;
    }
    shape adjust_shape(shape s) const
    {
        return base_adjust_shape(std::move(s));
    }
    std::unordered_map<int, dnnl::memory::desc> to_memory_desc(const shape& output_shape, const std::vector<shape>& inputs) const
    {
        const auto& self = static_cast<const Derived&>(*this);
        std::unordered_map<int, dnnl::memory::desc> result;
        result[DNNL_ARG_DST] = to_dnnl_memory_desc(self.adjust_shape(output_shape));
        auto m = self.arg_map(inputs.size());
        for(int i = 0;i< inputs.size();i++)
        {
            result[m[i]] = to_dnnl_memory_desc(self.adjust_shape(inputs[i]));
        }
        return result;
    }
    Primitive get_primitive(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        using primitive_desc = typename Primitive::primitive_desc;
        const auto& self = static_cast<const Derived&>(*this);
        auto desc = self.get_desc(m);
        auto pd   = primitive_desc(desc, get_dnnl_context().engine);
        return Primitive(pd);
    }
    std::string name() const { return "dnnl::" + op.name(); }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes(inputs, *this).standard();
        return op.compute_shape(inputs);
    }
    argument compute(context&, const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        const auto& self = static_cast<const Derived&>(*this);
        auto md = to_memory_desc(output_shape, to_shapes(args));
        auto prim = get_primitive(md);
        auto arg_lookup = self.arg_map(args.size());
        std::unordered_map<int, dnnl::memory> m;
        m[DNNL_ARG_DST] = to_dnnl_memory(md[DNNL_ARG_DST], result);
        for(int i=0;i<args.size();i++)
            m[arg_lookup[i]] = to_dnnl_memory(md[arg_lookup[i]], args[i]);
        prim.execute(get_dnnl_context().stream, m);
        return result;
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif

#endif
