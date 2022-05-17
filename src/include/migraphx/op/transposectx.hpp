#ifndef MIGRAPHX_GUARD_OPERATORS_TRANSPOSECTX_HPP
#define MIGRAPHX_GUARD_OPERATORS_TRANSPOSECTX_HPP

#include <array>
#include <migraphx/argument.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct transposectx
{
    std::string name() const { return "transposectx"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).standard();
        auto lens = inputs.front().lens();
        std::vector<std::size_t> out_lens{lens[0], lens[2], lens[1], lens[3]};

        return {inputs.front().type(), out_lens};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        // Input:  BxNxSxH
        // Output: BxSxNxH
        argument result{output_shape};
        auto in_s = args.front().get_shape();
        auto lens = in_s.lens();
        visit_all(result, args.front())([&](auto output, const auto input) {
            par_for(output_shape.elements(), [&](auto i) {
                auto idx = in_s.multi(i);

                int n = idx.at(1);
                int s = idx.at(2);
                int b = idx.front();

                int num_heads       = lens.at(1);
                int sequence_length = lens.at(2);
                int head_size = lens.back();

                const int NH        = num_heads * head_size;
                const int NHS       = NH * sequence_length;
                //const int in_offset = s * head_size + n * sequence_length * head_size + b * NHS;

                const int out_offset = n * head_size + s * NH + b * NHS;

                const int j = idx.back();
                output[out_offset + j] = input[i];
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
