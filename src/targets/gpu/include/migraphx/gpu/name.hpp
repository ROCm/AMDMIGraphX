#ifndef MIGRAPHX_GUARD_RTGLIB_OP_NAME_HPP
#define MIGRAPHX_GUARD_RTGLIB_OP_NAME_HPP

#include <migraphx/config.hpp>
#include <migraphx/type_name.hpp>
#include <utility>

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

        return "unknown_operator_name";
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
