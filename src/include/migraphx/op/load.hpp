#ifndef MIGRAPHX_GUARD_OPERATORS_LOAD_HPP
#define MIGRAPHX_GUARD_OPERATORS_LOAD_HPP

#include <array>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct load
{
    shape s;
    std::size_t offset = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.s, "shape"), f(self.offset, "offset"));
    }

    std::string name() const { return "load"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs}.has(1);
        return s;
    }
    argument compute(const shape&, const std::vector<argument>& args) const
    {
        if((offset + s.bytes()) > args[0].get_shape().bytes())
            MIGRAPHX_THROW("Load access is out of bounds");
        return {s, args[0].data() + offset};
    }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }

    friend std::ostream& operator<<(std::ostream& os, const load& op)
    {
        os << op.name() << "[";
        os << "offset=" << op.offset << ",";
        os << "end=" << (op.offset + op.s.bytes()) << "]";
        return os;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
