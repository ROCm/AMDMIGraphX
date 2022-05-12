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
    int head_size    = 64;
    bool reversed_bs = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.head_size, "head_size"), f(self.reversed_bs, "reversed_bs"));
    }

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
        // Input:  BxSxKxNxH or SxBxKxNxH
        // Output: KxBxNxSxH
        // K is the number of identical matrix
        argument result{output_shape};
        visit_all(result, args.front())([&](auto output, const auto input) {
            par_for(output_shape.elements(), [&](auto i) {
                // TODO: calculate in_offet and out_offset
                output[i] = input[i];
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
