#ifndef MIGRAPHX_GUARD_RTGLIB_UNARY_HPP
#define MIGRAPHX_GUARD_RTGLIB_UNARY_HPP

#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/type_name.hpp>
#include <utility>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

template <class Derived>
struct oper
{
    std::string name() const
    {
        const std::string& name = get_type_name<Derived>();
        // search the namespace gpu (::gpu::)
        auto pos_ns = name.find("::gpu::");
        if(pos_ns != std::string::npos)
        {
            auto pos_name = name.find("hip_", pos_ns + std::string("::gpu::").length());
            if(pos_name != std::string::npos)
            {
                return std::string("gpu::") + name.substr(pos_name + 4);
            }
            else
            {
                return name.substr(pos_ns + 2);
            }
        }

        return "unknown";
    }
};

template <class Derived, void (*F)(hipStream_t, const argument&, const argument&)>
struct unary_device : oper<Derived>
{
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        auto s = inputs.at(0);
        if(s.packed())
        {
            return s;
        }
        else
        {
            return {s.type(), s.lens()};
        }
    }

    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        F(ctx.get_stream().get(), args[1], args[0]);
        return args[1];
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

template <class Derived, void (*F)(hipStream_t, const argument&, const argument&, const argument&)>
struct binary_device : oper<Derived>
{
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        auto s0 = inputs.at(0);
        auto s1 = inputs.at(1);
        if(s0 == s1 and s0.packed())
        {
            return s0;
        }
        else
        {
            return {s0.type(), s0.lens()};
        }
    }

    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        F(ctx.get_stream().get(), args[2], args[1], args[0]);
        return args[2];
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
