#include <migraphx/config.hpp>
#include <migraphx/cpu/pointwise.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct cpu_copy : reduce_dims_base, auto_register_op<cpu_copy>
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }

    std::string name() const { return "cpu::copy"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        return inputs.at(1);
    }
    argument
    // cppcheck-suppress constParameter
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        argument result = get_arg(args, args.size() - 1);

        visit_all(result, get_arg(args, 0))([&](auto output, auto input) {
            pointwise(output, input)(ctx, output.get_shape(), 1024, [](auto& y, auto x) { y = x; });
        });

        return result.reshape(output_shape);
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
