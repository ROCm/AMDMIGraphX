#ifndef MIGRAPHX_GUARD_OPERATORS_TRANSPOSEQKV_HPP
#define MIGRAPHX_GUARD_OPERATORS_TRANSPOSEQKV_HPP

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

struct transposeqkv
{
    std::string name() const { return "transposeqkv"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).standard();
        auto lens = inputs.front().lens();
        std::vector<std::size_t> out_lens{lens[2], lens[0], lens[3], lens[1], lens[4]};

        return {inputs.front().type(), out_lens};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        // Input:  BxSxKxNxH
        // Output: KxBxNxSxH
        // K is the number of identical matrix

        auto in_s = args.front().get_shape();
        auto lens = in_s.lens();
        argument result{output_shape};
        visit_all(result, args.front())([&](auto output, const auto input) {
            par_for(output_shape.elements(), [&](auto i) {
                auto idx = in_s.multi(i);

                const int b = idx.front();
                const int s = idx.at(1);
                const int m = idx.at(2);
                const int n = idx.at(3);
                const int j = idx.back();

                const int num_heads = lens[3];

                const int sequence_length = lens[1];
                const int batch_size      = lens[0];
                const int H               = lens.back();
                const int NH              = num_heads * H;
                const int NHS             = NH * sequence_length;

                const int out_offset =
                    s * H + n * sequence_length * H + b * NHS + m * NHS * batch_size;

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
