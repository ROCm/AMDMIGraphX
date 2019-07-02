#ifndef MIGRAPHX_GUARD_OPERATORS_MULTIBROADCAST_HPP
#define MIGRAPHX_GUARD_OPERATORS_MULTIBROADCAST_HPP

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

struct multibroadcast
{
    std::vector<std::size_t> output_lens;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_lens, "output_lens"));
    }

    std::string name() const { return "multibroadcast"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto t     = inputs.at(0).type();
        auto input = inputs.at(0);

        if(input.lens().empty())
        {
            MIGRAPHX_THROW("MULTIBROADCAST: inputs dimensions should be > 0");
        }

        if(input.lens().size() > output_lens.size())
        { 
            MIGRAPHX_THROW("MULTIBROADCAST: inputs dimensions should <= output size");
        }

        auto offset = output_lens.size() - input.lens().size();
        for(std::ptrdiff_t i = input.lens().size() - 1; i >= 0; i--)
        {
            if(output_lens[i + offset] != input.lens()[i] and input.lens()[i] != 1)
            {
                MIGRAPHX_THROW("MULTIBROADCAST: input shape {" + 
                    to_string_range(input.lens()) + "} cannot be broadcasted to {" 
                    + to_string_range(output_lens) + "}!");
            }
        }

        std::vector<size_t> bcast_strides(output_lens.size(), 0);
        for(std::ptrdiff_t i = input.lens().size() - 1; i >= 0; i--)
        {
            if(output_lens[i + offset] == input.lens()[i])
            {
                bcast_strides[i + offset] = input.strides()[i];
            }
        }
        return {t, output_lens, bcast_strides};
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.at(0).data)};
    }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
