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

struct dnnl_gemm : auto_register_op<dnnl_gemm>
{
    op::dot op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }
    std::string name() const { return "dnnl::dot"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes(inputs, *this).has(2).standard();
        return op.compute_shape(inputs);
    }

    argument limit3(const argument& a) const
    {
        auto s     = a.get_shape();
        auto ndims = s.lens().size();
        if(ndims > 3)
        {
            std::size_t batch = std::accumulate(
                s.lens().begin(), s.lens().begin() + (ndims - 2), 1, std::multiplies<>{});
            shape s3d{s.type(), {batch, s.lens()[ndims - 2], s.lens()[ndims - 1]}};
            return a.reshape(s3d);
        }
        else
        {
            return a;
        }
    }

    argument compute(context& ctx, const shape& output_shape, std::vector<argument> args) const
    {
        if(args[0].get_shape().type() == shape::type_t::half_type)
            return op.compute(output_shape, args);
        argument result{output_shape};
        execute_dnnl<dnnl::matmul>(ctx,
                                   {{DNNL_ARG_SRC, limit3(args[0])},
                                    {DNNL_ARG_WEIGHTS, limit3(args[1])},
                                    {DNNL_ARG_DST, limit3(result)}})([&](auto m) {
            return dnnl::matmul::desc(m.at(DNNL_ARG_SRC).get_desc(),
                                      m.at(DNNL_ARG_WEIGHTS).get_desc(),
                                      m.at(DNNL_ARG_DST).get_desc());
        });
        return result;
    }
};

struct cpu_gemm : auto_register_op<cpu_gemm>
{
    op::dot op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }
    std::string name() const { return "cpu::dot"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.standard();
        if(inputs.size() == 3)
        {
            auto c_shape = inputs.at(2);
            check_shapes{{c_shape}, *this}.not_broadcasted();
        }
        return op.compute_shape(inputs);
    }

    argument compute(context&, const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        // 3 inputs, it is alpha * A * B + beta * C, then
        // A and B are matrices, and C is of the same shape as A * B
        if(args.size() == 3)
        {
            // no need to consider the value of args[2]
            if(op.beta == 0.0f)
            {
                result.visit([&](auto output) { std::fill(output.begin(), output.end(), 0); });
            }
            else
            {
                visit_all(result, args[2])([&](auto output, auto input) {
                    std::copy(input.begin(), input.end(), output.begin());
                });
            }

            migemm(result, args[0], args[1], op.alpha, op.beta);

            return result;
        }

        // 2 input arguments
        migemm(result, args[0], args[1], op.alpha, 0.0f);

        return result;
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
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.standard();
        if(inputs.size() == 3)
        {
            auto c_shape = inputs.at(2);
            check_shapes{{c_shape}, *this}.not_broadcasted();
        }
        return op.compute_shape(inputs);
    }

    argument compute(context&, const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
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
                result.visit([&](auto output) { std::fill(output.begin(), output.end(), 0); });
            }
            else
            {
                visit_all(result, args[2])([&](auto output, auto input) {
                    std::copy(input.begin(), input.end(), output.begin());
                });
            }

            migemm(result, arg_0, arg_1, op.alpha, op.beta);

            return result;
        }

        // 2 input arguments
        migemm(result, arg_0, arg_1, op.alpha, int32_t{0});

        return result;
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
