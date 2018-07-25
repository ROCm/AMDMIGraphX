
#include <migraph/cpu/cpu_lowering.hpp>
#include <migraph/instruction.hpp>
#include <migraph/dfor.hpp>
#include <migraph/operators.hpp>
#include <migraph/shape_for_each.hpp>
#include <migraph/iterator_for.hpp>
#include <unordered_map>

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

    shape compute_shape(std::vector<shape> inputs) const { return op.compute_shape(inputs); }

    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        argument output{output_shape};

        double epsilon           = op.epsilon;
        auto input               = args[0];
        auto mini_batch_mean     = args[1];
        auto mini_batch_variance = args[2];
        auto gamma               = args[3];
        auto bias                = args[4];

        auto num_batch    = output_shape.lens()[0];
        auto num_channels = output_shape.lens()[1];
        auto image_height = output_shape.lens()[2];
        auto image_width  = output_shape.lens()[3];

        visit_all(output, input, mini_batch_mean, mini_batch_variance, gamma, bias)(
            [&](auto result, auto buffer, auto _mean, auto _variance, auto _gamma, auto _bias) {
                for(size_t n = 0; n < num_batch; n++)
                {
                    size_t stride_n = n * num_channels * image_height * image_width;
                    for(size_t c = 0; c < num_channels; c++)
                    {
                        size_t stride_c = c * image_height * image_width;
                        for(size_t h = 0; h < image_height; h++)
                        {
                            size_t stride_h = h * image_width;
                            for(size_t w = 0; w < image_width; w++)
                            {
                                size_t index  = w + stride_h + stride_c + stride_n;
                                result[index] = _gamma[c] * (buffer[index] - _mean[c]) /
                                                    std::sqrt(_variance[c] + epsilon) +
                                                _bias[c];
                            }
                        }
                    }
                }
            });

        return output;
    }
};

struct cpu_convolution
{
    convolution op;

    std::string name() const { return "cpu::convolution"; }
    shape compute_shape(std::vector<shape> inputs) const { return op.compute_shape(inputs); }
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
    shape compute_shape(std::vector<shape> inputs) const { return op.compute_shape(inputs); }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
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

struct cpu_transpose
{
    transpose op;

    std::string name() const { return "cpu::transpose"; }
    shape compute_shape(std::vector<shape> inputs) const { return op.compute_shape(inputs); }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        return {output_shape, std::move(args.front().data)};
    }
};

struct cpu_contiguous
{
    contiguous op;
    std::string name() const { return "cpu::contiguous"; }
    shape compute_shape(std::vector<shape> inputs) const { return op.compute_shape(inputs); }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0])([&](auto output, auto input) {
            shape_for_each(output.get_shape(), [&](const auto& idx) {
                output(idx.begin(), idx.end()) = input(idx.begin(), idx.end());
            });
        });
        return result;
    }
};

struct cpu_reshape
{
    reshape op;
    std::string name() const { return "cpu::reshape"; }
    shape compute_shape(std::vector<shape> inputs) const { return op.compute_shape(inputs); }

    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        return {output_shape, std::move(args.front().data)};
    }
};

struct cpu_gemm
{
    gemm op;
    std::string name() const { return "cpu::gemm"; }
    shape compute_shape(std::vector<shape> inputs) const { return op.compute_shape(inputs); }

    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0], args[1])([&](auto cmat, auto amat, auto bmat) {
            auto m = amat.get_shape().lens()[0];
            auto n = bmat.get_shape().lens()[1];
            auto k = bmat.get_shape().lens()[0];

            auto a = amat.data();
            auto b = bmat.data();
            auto c = cmat.data();
            for(int ii = 0; ii < m; ii++)
            {
                for(int jj = 0; jj < n; jj++)
                {
                    c[ii * n + jj] = 0;
                }
            }
            for(int ii = 0; ii < m; ii++)
            {
                for(int kk = 0; kk < k; kk++)
                {
                    auto aik  = a[ii * k + kk];
                    auto* bkj = &b[kk * n];
                    auto* cij = &c[ii * n];
                    for(int jj = 0; jj < n; jj++, cij++, bkj++)
                    {
                        *cij += aik * (*bkj);
                    }
                }
            }
        });
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
    shape compute_shape(std::vector<shape> inputs) const { return inputs.front(); }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
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
    shape compute_shape(std::vector<shape> inputs) const { return inputs.front(); }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
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
    shape compute_shape(std::vector<shape> inputs) const { return inputs.front(); }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
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
        apply_map["convolution"] = extend_op<cpu_convolution, convolution>();
        apply_map["gemm"]        = extend_op<cpu_gemm, gemm>();
        apply_map["batch_norm_inference"] =
            extend_op<cpu_batch_norm_inference, batch_norm_inference>();
        apply_map["reshape"]    = extend_op<cpu_reshape, reshape>();
        apply_map["contiguous"] = extend_op<cpu_contiguous, contiguous>();
        apply_map["transpose"]  = extend_op<cpu_transpose, transpose>();

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
            if(it->op.name() == "activation")
            {
                apply_activation(it);
            }
            else if(it->op.name() == "pooling")
            {
                apply_pooling(it);
            }
            else if(apply_map.count(it->op.name()) > 0)
            {
                apply_map.at(it->op.name())(it);
            }
        }
    }

    template <class T>
    void apply_simple_op(instruction_ref ins)
    {
        prog->replace_instruction(ins, T{}, ins->arguments);
    }

    template <class T, class Op>
    void apply_extend_op(instruction_ref ins)
    {
        auto&& op = any_cast<Op>(ins->op);
        prog->replace_instruction(ins, T{op}, ins->arguments);
    }

    void apply_activation(instruction_ref ins)
    {
        auto&& op = any_cast<activation>(ins->op);
        if(op.mode == "relu")
            prog->replace_instruction(ins, cpu_unary<relu_op>{}, ins->arguments);
    }

    void apply_pooling(instruction_ref ins)
    {
        auto&& op = any_cast<pooling>(ins->op);
        if(op.mode == "max")
            prog->replace_instruction(ins, cpu_pooling<max_pool>{op}, ins->arguments);
        else if(op.mode == "average")
            prog->replace_instruction(ins, cpu_pooling<avg_pool>{op}, ins->arguments);
    }
};

void cpu_lowering::apply(program& p) const { cpu_apply{&p}.apply(); }

} // namespace cpu

} // namespace migraph
