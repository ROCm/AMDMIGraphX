#ifndef MIGRAPHX_GUARD_OPERATORS_GET_TUPLE_ELEM_HPP
#define MIGRAPHX_GUARD_OPERATORS_GET_TUPLE_ELEM_HPP

#include "migraphx/errors.hpp"
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct get_tuple_elem
{
    std::size_t index = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.index, "index"));
    }

    std::string name() const { return "get_tuple_elem"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).tuple_type();
        const auto& sub_shapes = inputs.at(0).sub_shapes();
        if(index >= sub_shapes.size())
        {
            MIGRAPHX_THROW("GET_TUPLE_ELEM: index " + std::to_string(index) + " is out of range " +
                           std::to_string(sub_shapes.size()));
        }

        return sub_shapes.at(index);
    }

    argument compute(const shape&, std::vector<argument> args) const
    {
        assert(args.size() == 1);
        auto vec_args = args.at(0).get_sub_objects();
        assert(index < vec_args.size());
        return vec_args.at(index);
    }

    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
