#ifndef MIGRAPHX_GUARD_OPERATORS_ABNORMAL_OPS_HPP
#define MIGRAPHX_GUARD_OPERATORS_ABNORMAL_OPS_HPP

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

struct not_computable
{
    argument compute(const shape&, const std::vector<argument>&) const
    {
        MIGRAPHX_THROW("not computable");
    }
};

struct undefined
{
    std::string name() const { return "undefined"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(0);
        return {};
    }

    argument compute(const shape&, const std::vector<argument>&) const { return {{}, nullptr}; }
};

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
