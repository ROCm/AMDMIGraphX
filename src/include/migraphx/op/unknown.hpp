#ifndef MIGRAPHX_GUARD_RTGLIB_UNKNOWN_HPP
#define MIGRAPHX_GUARD_RTGLIB_UNKNOWN_HPP

#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/check_shapes.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct unknown
{
    std::string op;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }
    std::string name() const { return "unknown:" + op; }
    shape compute_shape(std::vector<shape> input) const
    {
        if(input.empty())
            return {};
        else
            return input.front();
    }

    friend std::ostream& operator<<(std::ostream& os, const unknown& x)
    {
        os << x.name();
        return os;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
