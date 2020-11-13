#include <migraphx/config.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/context.hpp>
#include <migraphx/cpu/context.hpp>
#include <migraphx/cpu/dnnl.hpp>
#include <migraphx/cpu/migemm.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/quant_dot.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

#if USE_DNNL
struct dnnl_gemm : dnnl_op<dnnl_gemm, dnnl::matmul, op::dot>
{
    std::vector<int> arg_map(int) const { return {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS}; }

    // Batch must be a single dimension
    shape adjust_shape(shape x) const
    {
        auto s     = base_adjust_shape(std::move(x));
        auto ndims = s.lens().size();
        if(ndims > 3)
        {
            std::size_t batch = std::accumulate(
                s.lens().begin(), s.lens().begin() + (ndims - 2), 1, std::multiplies<>{});
            shape s3d{s.type(), {batch, s.lens()[ndims - 2], s.lens()[ndims - 1]}};
            return s3d;
        }
        else
        {
            return s;
        }
    }

    dnnl::matmul::desc get_desc(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        return dnnl::matmul::desc(m.at(DNNL_ARG_SRC), m.at(DNNL_ARG_WEIGHTS), m.at(DNNL_ARG_DST));
    }
};
#endif

struct cpu_gemm : auto_register_op<cpu_gemm>
{
    op::dot op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }
    std::string name() const { return "cpu::dot"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.standard();
        inputs.pop_back();
        return op.compute_shape(inputs);
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }

    argument compute(context&, const shape&, std::vector<argument> args) const
    {
        // 3 inputs, it is alpha * A * B + beta * C, then
        // A and B are matrices, and C is of the same shape as A * B
        if(args.size() == 3)
        {
            // no need to consider the value of args[2]
            if(op.beta == 0.0f)
            {
                args.back().visit([&](auto output) { std::fill(output.begin(), output.end(), 0); });
            }
            else
            {
                visit_all(args.back(), args[2])([&](auto output, auto input) {
                    std::copy(input.begin(), input.end(), output.begin());
                });
            }

            migemm(args.back(), args[0], args[1], op.alpha, op.beta);

            return args.back();
        }

        // 2 input arguments
        migemm(args.back(), args[0], args[1], op.alpha, 0.0f);

        return args.back();
    }
};

struct cpu_quant_gemm : auto_register_op<cpu_quant_gemm>
{
    op::quant_dot op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "cpu::quant_dot"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.standard();
        inputs.pop_back();
        return op.compute_shape(inputs);
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }

    argument compute(context&, const shape&, std::vector<argument> args) const
    {
        // 3 inputs, it is alpha * A * B + beta * C, then
        // A and B are matrices, and C is of the same shape to A * B

        // first, convert the args[0] and args[1] from int8_t to int32_t
        argument arg_0{{shape::int32_type, {args.at(0).get_shape().lens()}}};
        argument arg_1{{shape::int32_type, {args.at(1).get_shape().lens()}}};
        arg_0.visit([&](auto output) {
            args.at(0).visit(
                [&](auto input) { std::copy(input.begin(), input.end(), output.begin()); });
        });

        arg_1.visit([&](auto output) {
            args.at(1).visit(
                [&](auto input) { std::copy(input.begin(), input.end(), output.begin()); });
        });

        if(args.size() == 3)
        {
            // no need to consider the value of args[2]
            if(op.beta == 0)
            {
                args.back().visit([&](auto output) { std::fill(output.begin(), output.end(), 0); });
            }
            else
            {
                visit_all(args.back(), args[2])([&](auto output, auto input) {
                    std::copy(input.begin(), input.end(), output.begin());
                });
            }

            migemm(args.back(), arg_0, arg_1, op.alpha, op.beta);

            return args.back();
        }

        // 2 input arguments
        migemm(args.back(), arg_0, arg_1, op.alpha, int32_t{0});

        return args.back();
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
