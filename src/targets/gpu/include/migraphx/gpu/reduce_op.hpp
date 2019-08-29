#ifndef MIGRAPHX_GUARD_RTGLIB_REDUCE_OP_HPP
#define MIGRAPHX_GUARD_RTGLIB_REDUCE_OP_HPP

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

struct context;

template<class Derived, class Op, void (*F)(hipStream_t, const argument&, const argument&)>
struct reduce_op 
{
    Op op;

    template <class Self, class T>
    static auto reflect(Self& self, T f)
    {
        return migraphx::reflect(self.op, f);
    }

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

    shape compute_shape(std::vector<shape> inputs) const
    {
        std::vector<shape> in_shapes{inputs};
        in_shapes.pop_back();
        return op.compute_shape(in_shapes);
    }

    argument
    compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        F(ctx.get_stream().get(), args[1], args[0]);
        return args[1];
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }

    reduce_op() { }
    reduce_op(const Op& op_ref) : op(op_ref) { }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
