
#include <migraph/cpu/cpu_lowering.hpp>
#include <migraph/instruction.hpp>
#include <migraph/dfor.hpp>
#include <migraph/operators.hpp>
#include <migraph/shape_for_each.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/cpu/gemm.hpp>
#include <unordered_map>
#include <utility>

namespace migraph {
namespace cpu {

template <typename T>
T zero(const T&)
{
    return T(0);
}

//
// cpu implemenataion of batch norm for inference
//
// inputs are:
// args[0] -> input data buffer
// args[1] -> mini batch mean
// args[2] -> mini batch variance
// args[3] -> gamma
// args[4] -> bias
//
// The equation to compute batch norm for inference is:
//
// output[i] = bias + gamma * (input[i] + mean) / sqrt(variance + epsilon)
//
// the input data format should be nchw
//
struct cpu_batch_norm_inference
{
    batch_norm_inference op;

    std::string name() const { return "cpu::batch_norm_inference"; }

    shape compute_shape(const std::vector<shape>& inputs) const { return op.compute_shape(inputs); }

    argument compute(context&, const shape& output_shape, std::vector<argument> args) const
    {
        argument output{output_shape};

        double epsilon           = op.epsilon;
        auto input               = args[0];
        auto arg_gamma           = args[1];
        auto arg_bias            = args[2];
        auto mini_batch_mean     = args[3];
        auto mini_batch_variance = args[4];

        auto num_batch    = output_shape.lens()[0];
        auto num_channels = output_shape.lens()[1];
        auto image_height = output_shape.lens()[2];
        auto image_width  = output_shape.lens()[3];

        if(op.bn_mode == batch_norm_inference::spatial)
        {
            visit_all(output, input, mini_batch_mean, mini_batch_variance, arg_gamma, arg_bias)(
                [&](auto result, auto buffer, auto mean, auto variance, auto gamma, auto bias) {

                    dfor(num_batch, num_channels, image_height, image_width)(
                        [&](std::size_t n, std::size_t c, std::size_t h, std::size_t w) {
                            assert((variance(c) + epsilon) > 0);
                            result(n, c, h, w) = gamma(c) * (buffer(n, c, h, w) - mean(c)) /
                                                     std::sqrt(variance(c) + epsilon) +
                                                 bias(c);
                        });
                });
        }

        if(op.bn_mode == batch_norm_inference::per_activation)
        {
            visit_all(output, input, mini_batch_mean, mini_batch_mean, arg_gamma, arg_bias)(
                [&](auto result, auto buffer, auto mean, auto variance, auto gamma, auto bias) {

                    dfor(num_batch, num_channels, image_height, image_width)(
                        [&](std::size_t n, std::size_t c, std::size_t h, std::size_t w) {
                            assert((variance(c, h, w) + epsilon) > 0);
                            result(n, c, h, w) = gamma(c, h, w) *
                                                     (buffer(n, c, h, w) - mean(c, h, w)) /
                                                     std::sqrt(variance(c, h, w) + epsilon) +
                                                 bias(c, h, w);
                        });
                });
        }

        return output;
    }
};

struct cpu_convolution
{
    convolution op;

    std::string name() const { return "cpu::convolution"; }
    shape compute_shape(const std::vector<shape>& inputs) const { return op.compute_shape(inputs); }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0], args[1])([&](auto output, auto input, auto weights) {
            auto in_h = input.get_shape().lens()[2];
            auto in_w = input.get_shape().lens()[3];

            auto wei_c = weights.get_shape().lens()[1];
            auto wei_h = weights.get_shape().lens()[2];
            auto wei_w = weights.get_shape().lens()[3];

            dfor(output_shape.lens()[0],
                 output_shape.lens()[1],
                 output_shape.lens()[2],
                 output_shape.lens()[3])(
                [&](std::size_t o, std::size_t w, std::size_t i, std::size_t j) {
                    const int start_x = i * op.stride[0] - op.padding[0];
                    const int start_y = j * op.stride[1] - op.padding[1];

                    double acc = 0;
                    dfor(wei_c, wei_h, wei_w)([&](std::size_t k, std::size_t x, std::size_t y) {
                        const int in_x = start_x + x;
                        const int in_y = start_y + y;
                        if(in_x >= 0 && in_x < in_h && in_y >= 0 && in_y < in_w)
                        {
                            acc += input(o, k, in_x, in_y) * weights(w, k, x, y);
                        }
                    });
                    output(o, w, i, j) = acc;
                });
        });
        return result;
    }
};

struct cpu_im2col
{
    im2col op;

    static std::string name() { return "cpu::im2col"; }
    shape compute_shape(const std::vector<shape>& inputs) const { return op.compute_shape(inputs); }

    argument compute(context&, const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto input_shape   = args[0].get_shape();
        auto weights_shape = args[1].get_shape();
        visit_all(result, args[0])([&](auto col, auto input) {
            const std::size_t& height   = input_shape.lens()[2];
            const std::size_t& width    = input_shape.lens()[3];
            const std::size_t& channels = weights_shape.lens()[1];
            const std::size_t& kernel_h = weights_shape.lens()[2];
            const std::size_t& kernel_w = weights_shape.lens()[3];
            const std::size_t& pad_h    = op.padding[0];
            const std::size_t& pad_w    = op.padding[1];
            const std::size_t& stride_h = op.stride[0];
            const std::size_t& stride_w = op.stride[1];

            int kdiv2_h, kdiv2_w;
            kdiv2_h = kernel_h / 2;
            kdiv2_w = kernel_w / 2;
            // calculate output sizes
            const std::size_t col_height = (height - kernel_h + 2 * pad_h) / stride_h + 1;
            const std::size_t col_width  = (width - kernel_w + 2 * pad_w) / stride_w + 1;
            // account for padding for the starting position of the input pixels
            std::size_t iinput = kdiv2_h - pad_h;
            // loop over output pixels (ioutput, joutput)
            for(std::size_t ioutput = 0; ioutput < col_height; ioutput++, iinput += stride_h)
            {
                std::size_t jinput = kdiv2_w - pad_w;
                for(std::size_t joutput = 0; joutput < col_width; joutput++, jinput += stride_w)
                {
                    // compute linear index for output
                    std::size_t ldx = ioutput * col_width + joutput;
                    std::size_t p   = 0;
                    dfor(channels,
                         kernel_h,
                         kernel_w)([&](std::size_t c, std::size_t koffset, std::size_t loffset) {
                        int idx     = iinput + koffset - kdiv2_h;
                        int jdx     = jinput + loffset - kdiv2_w;
                        col(ldx, p) = ((idx >= 0) && (idx < height) && (jdx >= 0) && (jdx < width))
                                          ? input(0, c, idx, jdx)
                                          : 0;
                        p++;
                    });
                }
            }
        });
        return result;
    }
};

struct max_pool
{
    static std::string name() { return "max"; }
    static double start() { return std::numeric_limits<double>::lowest(); }

    static double apply(double x, double y)
    {
        double m = std::max(x, y);
        return (m);
    }

    static double final(double x, double) { return (x); }
};

struct avg_pool
{
    static std::string name() { return "average"; }
    static double start() { return 0.0; }

    static double apply(double x, double y) { return x + y; }

    static double final(double x, double y) { return x / y; }
};

template <class Op>
struct cpu_pooling
{
    pooling op;

    std::string name() const { return "cpu::pooling_" + Op::name(); }
    shape compute_shape(const std::vector<shape>& inputs) const { return op.compute_shape(inputs); }
    argument compute(context&, const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0])([&](auto output, auto input) {
            using type = typename decltype(output)::value_type;
            auto in_h  = input.get_shape().lens()[2];
            auto in_w  = input.get_shape().lens()[3];

            dfor(output_shape.lens()[0],
                 output_shape.lens()[1],
                 output_shape.lens()[2],
                 output_shape.lens()[3])(
                [&](std::size_t o, std::size_t w, std::size_t i, std::size_t j) {
                    const int start_x0 = i * op.stride[0] - op.padding[0];
                    const int start_y0 = j * op.stride[1] - op.padding[1];

                    const int hend = std::min(start_x0 + op.lengths[0], in_h);
                    const int wend = std::min(start_y0 + op.lengths[1], in_w);

                    const int start_x = std::max(start_x0, 0);
                    const int start_y = std::max(start_y0, 0);

                    const int w_h       = (hend - start_x);
                    const int w_w       = (wend - start_y);
                    const int pool_size = std::max(w_h * w_w, 1);

                    double acc = Op::start();
                    dfor(w_h, w_w)([&](int x, int y) {
                        const int in_x = start_x + x;
                        const int in_y = start_y + y;
                        if(in_x >= 0 && in_x < in_h && in_y >= 0 && in_y < in_w)
                        {
                            acc = Op::apply(acc, input(o, w, in_x, in_y));
                        }
                    });
                    output(o, w, i, j) = type(Op::final(acc, pool_size));
                });
        });
        return result;
    }
};

struct cpu_contiguous
{
    contiguous op;
    std::string name() const { return "cpu::contiguous"; }
    shape compute_shape(const std::vector<shape>& inputs) const { return op.compute_shape(inputs); }
    argument compute(context&, const shape& output_shape, std::vector<argument> args) const
    {
        assert(output_shape.standard());
        argument result{output_shape};
        visit_all(result, args[0])([&](auto output, auto input) {
            shape_for_each(output.get_shape(), [&](const auto& idx) {
                output(idx.begin(), idx.end()) = input(idx.begin(), idx.end());
            });
        });
        return result;
    }
};

struct cpu_gemm
{
    gemm op;
    std::string name() const { return "cpu::gemm"; }
    shape compute_shape(const std::vector<shape>& inputs) const { return op.compute_shape(inputs); }

    argument compute(context&, const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        migemm(result, args[0], args[1], op.alpha, op.beta);
        return result;
    }
};

struct identity_op
{
    std::string name() const { return "cpu::identity"; }
    auto fcn() const
    {
        return [](auto x) { return x; };
    }
};

struct abs_op
{
    std::string name() const { return "cpu::abs"; }
    auto fcn() const
    {
        return [](auto x) { return std::abs(x); };
    }
};

struct exp_op
{
    std::string name() const { return "cpu::exp"; }
    auto fcn() const
    {
        return [](auto x) { return std::exp(x); };
    }
};

struct sin_op
{
    std::string name() const { return "cpu::sin"; }
    auto fcn() const
    {
        return [](auto x) { return std::sin(x); };
    }
};

struct cos_op
{
    std::string name() const { return "cpu::cos"; }
    auto fcn() const
    {
        return [](auto x) { return std::cos(x); };
    }
};

struct tan_op
{
    std::string name() const { return "cpu::tan"; }
    auto fcn() const
    {
        return [](auto x) { return std::tan(x); };
    }
};

struct asin_op
{
    std::string name() const { return "cpu::asin"; }
    auto fcn() const
    {
        return [](auto x) { return std::asin(x); };
    }
};

struct acos_op
{
    std::string name() const { return "cpu::acos"; }
    auto fcn() const
    {
        return [](auto x) { return std::acos(x); };
    }
};

struct atan_op
{
    std::string name() const { return "cpu::atan"; }
    auto fcn() const
    {
        return [](auto x) { return std::atan(x); };
    }
};

struct tanh_op
{
    std::string name() const { return "cpu::tanh"; }
    auto fcn() const
    {
        return [](auto x) { return std::tanh(x); };
    }
};

struct sigmoid_op
{
    std::string name() const { return "cpu::sigmoid"; }
    auto fcn() const
    {
        return [](auto x) { return 1.f / (1.f + std::exp(-x)); };
    }
};

struct neg_op
{
    std::string name() const { return "cpu::neg"; }
    auto fcn() const
    {
        return [](auto x) { return -x; };
    }
};

struct relu_op
{
    std::string name() const { return "cpu::relu"; }
    auto fcn() const
    {
        return [](auto x) { return x > 0 ? x : 0; };
    }
};

template <typename Op>
struct cpu_unary
{
    Op op;
    std::string name() const { return op.name(); }
    shape compute_shape(const std::vector<shape>& inputs) const { return inputs.front(); }
    argument compute(context&, const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        result.visit([&](auto output) {
            args[0].visit([&](auto input) {
                std::transform(input.begin(), input.end(), output.begin(), op.fcn());
            });
        });
        return result;
    }
};

struct softmax2d
{
    std::string name() const { return "cpu::softmax2d"; }
    shape compute_shape(const std::vector<shape>& inputs) const { return inputs.front(); }
    argument compute(context&, const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0])([&](auto output, auto input) {
            using value_type = typename decltype(input)::value_type;
            auto nb          = input.get_shape().lens()[0];
            auto nc          = input.get_shape().lens()[1];
            auto nh          = input.get_shape().lens()[2];
            auto nw          = input.get_shape().lens()[3];
            dfor(nb, nh, nw)([&](std::size_t b, std::size_t i, std::size_t j) {
                value_type cmax = std::numeric_limits<value_type>::lowest();
                for(int c = 0; c < nc; c++)
                {
                    cmax = std::max(cmax, input(b, c, i, j));
                }
                for(int c = 0; c < nc; c++)
                {
                    output(b, c, i, j) = std::exp(input(b, c, i, j) - cmax);
                }
                value_type sum = value_type(0);
                for(int c = 0; c < nc; c++)
                {
                    sum += output(b, c, i, j);
                }
                for(int c = 0; c < nc; c++)
                {
                    output(b, c, i, j) = output(b, c, i, j) / sum;
                }
            });
        });
        return result;
    }
};

struct add_op
{
    std::string name() const { return "add"; }
    auto fcn() const
    {
        return [](auto x, auto y) { return x + y; };
    }
};

struct sub_op
{
    std::string name() const { return "sub"; }
    auto fcn() const
    {
        return [](auto x, auto y) { return x - y; };
    }
};

struct mul_op
{
    std::string name() const { return "mul"; }
    auto fcn() const
    {
        return [](auto x, auto y) { return x * y; };
    }
};

struct div_op
{
    std::string name() const { return "div"; }
    auto fcn() const
    {
        return [](auto x, auto y) { return x / y; };
    }
};

template <typename Op>
struct cpu_binary
{
    Op op;
    std::string name() const { return op.name(); }
    shape compute_shape(const std::vector<shape>& inputs) const { return inputs.front(); }
    argument compute(context&, const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0], args[1])([&](auto output, auto input1, auto input2) {
            if(input1.get_shape().packed() and input2.get_shape().packed())
            {
                std::transform(
                    input1.begin(), input1.end(), input2.begin(), output.begin(), op.fcn());
            }
            else
            {
                shape_for_each(output.get_shape(), [&](const auto& idx) {
                    output(idx.begin(), idx.end()) =
                        op.fcn()(input1(idx.begin(), idx.end()), input2(idx.begin(), idx.end()));
                });
            }
        });
        return result;
    }
};

struct cpu_apply
{
    program* prog;
    std::unordered_map<std::string, std::function<void(instruction_ref)>> apply_map{};

    template <class T>
    auto simple_op()
    {
        return [this](instruction_ref ins) { apply_simple_op<T>(ins); };
    }

    template <class T, class Op>
    auto extend_op()
    {
        return [this](instruction_ref ins) { apply_extend_op<T, Op>(ins); };
    }

    void init()
    {
        apply_map["im2col"]      = extend_op<cpu_im2col, im2col>();
        apply_map["convolution"] = extend_op<cpu_convolution, convolution>();
        apply_map["gemm"]        = extend_op<cpu_gemm, gemm>();
        apply_map["batch_norm_inference"] =
            extend_op<cpu_batch_norm_inference, batch_norm_inference>();
        apply_map["contiguous"] = extend_op<cpu_contiguous, contiguous>();

        apply_map["identity"] = simple_op<cpu_unary<identity_op>>();
        apply_map["tanh"]     = simple_op<cpu_unary<tanh_op>>();
        apply_map["sigmoid"]  = simple_op<cpu_unary<sigmoid_op>>();
        apply_map["exp"]      = simple_op<cpu_unary<exp_op>>();
        apply_map["neg"]      = simple_op<cpu_unary<neg_op>>();
        apply_map["sin"]      = simple_op<cpu_unary<sin_op>>();
        apply_map["cos"]      = simple_op<cpu_unary<cos_op>>();
        apply_map["tan"]      = simple_op<cpu_unary<tan_op>>();
        apply_map["add"]      = simple_op<cpu_binary<add_op>>();
        apply_map["sub"]      = simple_op<cpu_binary<sub_op>>();
        apply_map["mul"]      = simple_op<cpu_binary<mul_op>>();
        apply_map["div"]      = simple_op<cpu_binary<div_op>>();

        apply_map["softmax"] = simple_op<softmax2d>();
    }

    void apply()
    {
        init();
        for(auto it : iterator_for(*prog))
        {
            if(it->name() == "activation")
            {
                apply_activation(it);
            }
            else if(it->name() == "pooling")
            {
                apply_pooling(it);
            }
            else if(apply_map.count(it->name()) > 0)
            {
                apply_map.at(it->name())(it);
            }
        }
    }

    template <class T>
    void apply_simple_op(instruction_ref ins)
    {
        prog->replace_instruction(ins, T{}, ins->inputs());
    }

    template <class T, class Op>
    void apply_extend_op(instruction_ref ins)
    {
        auto&& op = any_cast<Op>(ins->get_operator());
        prog->replace_instruction(ins, T{op}, ins->inputs());
    }

    void apply_activation(instruction_ref ins)
    {
        auto&& op = any_cast<activation>(ins->get_operator());
        if(op.mode == "relu")
            prog->replace_instruction(ins, cpu_unary<relu_op>{}, ins->inputs());
    }

    void apply_pooling(instruction_ref ins)
    {
        auto&& op = any_cast<pooling>(ins->get_operator());
        if(op.mode == "max")
            prog->replace_instruction(ins, cpu_pooling<max_pool>{op}, ins->inputs());
        else if(op.mode == "average")
            prog->replace_instruction(ins, cpu_pooling<avg_pool>{op}, ins->inputs());
    }
};

void cpu_lowering::apply(program& p) const { cpu_apply{&p}.apply(); }

} // namespace cpu

} // namespace migraph
