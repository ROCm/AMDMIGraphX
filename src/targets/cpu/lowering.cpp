
#include <migraphx/cpu/lowering.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/op/batch_norm.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/deconvolution.hpp>
#include <migraphx/op/quant_convolution.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/quant_dot.hpp>
#include <migraphx/op/elu.hpp>
#include <migraphx/op/im2col.hpp>
#include <migraphx/op/leaky_relu.hpp>
#include <migraphx/op/logsoftmax.hpp>
#include <migraphx/op/lrn.hpp>
#include <migraphx/op/pad.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/op/softmax.hpp>
#include <migraphx/op/argmax.hpp>
#include <migraphx/op/argmin.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/par_dfor.hpp>
#include <migraphx/clamp.hpp>
#include <migraphx/cpu/gemm.hpp>
#include <unordered_map>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

template <typename T>
T zero(const T&)
{
    return T(0);
}

template <class T>
typename std::conditional_t<std::is_integral<T>{}, std::make_signed<T>, std::enable_if<true, T>>::
    type
    make_signed(T x)
{
    return x;
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
    op::batch_norm_inference op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

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

        if(op.bn_mode == op::batch_norm_inference::spatial)
        {
            visit_all(output, input, mini_batch_mean, mini_batch_variance, arg_gamma, arg_bias)(
                [&](auto result, auto buffer, auto mean, auto variance, auto gamma, auto bias) {

                    par_dfor(num_batch, num_channels, image_height, image_width)(
                        [&](std::size_t n, std::size_t c, std::size_t h, std::size_t w) {
                            assert((variance[c] + epsilon) > 0);
                            result(n, c, h, w) = gamma[c] * (buffer(n, c, h, w) - mean[c]) /
                                                     std::sqrt(variance[c] + epsilon) +
                                                 bias[c];
                        });
                });
        }

        if(op.bn_mode == op::batch_norm_inference::per_activation)
        {
            visit_all(output, input, mini_batch_mean, mini_batch_mean, arg_gamma, arg_bias)(
                [&](auto result, auto buffer, auto mean, auto variance, auto gamma, auto bias) {

                    par_dfor(num_batch, num_channels, image_height, image_width)(
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

struct cpu_lrn
{
    op::lrn op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "cpu::lrn"; }
    shape compute_shape(const std::vector<shape>& inputs) const { return op.compute_shape(inputs); }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0])([&](auto output, auto input) {
            int n_batch         = output_shape.lens()[0];
            int channels        = output_shape.lens()[1];
            int height          = output_shape.lens()[2];
            int width           = output_shape.lens()[3];
            float alphaoverarea = op.alpha / float(op.size);
            int radius_lower    = (op.size - 1) / 2;
            int radius_upper    = op.size / 2 + 1;

            par_dfor(n_batch, height, width)([&](int b, int h, int w) {
                float scale = 0;
                dfor(channels)([&](int c) {
                    auto start = (c - radius_lower) < 0 ? 0 : (c - radius_lower);
                    auto end   = (c + radius_upper) > channels ? channels : (c + radius_upper);
                    for(auto k = start; k < end; ++k)
                    {
                        scale += std::pow(input(b, k, h, w), 2);
                    }
                    scale *= alphaoverarea;
                    scale += op.bias;
                    scale              = std::pow(scale, -op.beta);
                    output(b, c, h, w) = input(b, c, h, w) * scale;
                });
            });
        });
        return result;
    }
};

template <class Op>
struct cpu_convolution
{
    Op op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "cpu::" + op.name(); }
    shape compute_shape(const std::vector<shape>& inputs) const { return op.compute_shape(inputs); }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        result.visit([&](auto output) {
            using type = typename decltype(output)::value_type;
            visit_all(args[0], args[1])([&](auto input, auto weights) {
                auto in   = input.get_shape().lens();
                auto in_h = in[2];
                auto in_w = in[3];

                auto wei   = weights.get_shape().lens();
                auto wei_n = wei[0];
                auto wei_c = wei[1];
                auto wei_h = wei[2];
                auto wei_w = wei[3];

                par_dfor(output_shape.lens()[0],
                         output_shape.lens()[1],
                         output_shape.lens()[2],
                         output_shape.lens()[3])(
                    [&](std::size_t o, std::size_t w, std::size_t i, std::size_t j) {
                        const auto start_x  = i * op.stride[0] - op.padding[0];
                        const auto start_y  = j * op.stride[1] - op.padding[1];
                        const auto group_id = w / (wei_n / op.group);

                        type acc = type{0};
                        dfor(wei_c, wei_h, wei_w)([&](std::size_t k, std::size_t x, std::size_t y) {
                            const auto in_x  = start_x + x;
                            const auto in_y  = start_y + y;
                            const auto in_ch = group_id * wei_c + k;
                            if(in_x >= 0 && in_x < in_h && in_y >= 0 && in_y < in_w)
                                acc += input(o, in_ch, in_x, in_y) * weights(w, k, x, y);
                        });
                        output(o, w, i, j) = acc;
                    });
            });
        });
        return result;
    }
};

template <class Op>
struct cpu_deconvolution
{
    Op op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "cpu::" + op.name(); }
    shape compute_shape(const std::vector<shape>& inputs) const { return op.compute_shape(inputs); }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0], args[1])([&](auto output, auto input, auto weights) {
            using type = typename decltype(output)::value_type;

            std::fill(output.begin(), output.end(), type{0});

            auto out_lens = output_shape.lens();
            auto out_h    = out_lens[2];
            auto out_w    = out_lens[3];

            auto in   = input.get_shape().lens();
            auto in_n = in[0];
            auto in_c = in[1];
            auto in_h = in[2];
            auto in_w = in[3];

            auto wei   = weights.get_shape().lens();
            auto wei_n = wei[0];
            auto wei_c = wei[1];
            auto wei_h = wei[2];
            auto wei_w = wei[3];

            par_dfor(in_n, wei_c)([&](std::size_t o, std::size_t k) {

                dfor(in_c, in_h, in_w, wei_h, wei_w)(
                    [&](std::size_t w, std::size_t i, std::size_t j, std::size_t x, std::size_t y) {
                        const int start_x = i * op.stride[0] - op.padding[0];
                        const int start_y = j * op.stride[1] - op.padding[1];
                        const int out_x   = start_x + x * op.dilation[0];
                        const int out_y   = start_y + y * op.dilation[1];

                        const auto group_id = w / (wei_n / op.group);
                        const auto in_ch    = group_id * wei_c + k;

                        if(out_x >= 0 && out_x < out_h && out_y >= 0 && out_y < out_w)
                        {
                            output(o, in_ch, out_x, out_y) +=
                                input(o, w, i, j) * weights(w, k, x, y);
                        }
                    });
            });
        });
        return result;
    }
};

struct cpu_im2col
{
    op::im2col op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

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

            long kdiv2_h = long(kernel_h) / 2;
            long kdiv2_w = long(kernel_w) / 2;
            // calculate output sizes
            const std::size_t col_height = (height - kernel_h + 2 * pad_h) / stride_h + 1;
            const std::size_t col_width  = (width - kernel_w + 2 * pad_w) / stride_w + 1;
            // account for padding for the starting position of the input pixels
            long iinput = kdiv2_h - long(pad_h);
            // loop over output pixels (ioutput, joutput)
            for(std::size_t ioutput = 0; ioutput < col_height; ioutput++, iinput += stride_h)
            {
                long jinput = kdiv2_w - long(pad_w);
                for(std::size_t joutput = 0; joutput < col_width; joutput++, jinput += stride_w)
                {
                    // compute linear index for output
                    std::size_t ldx = ioutput * col_width + joutput;
                    std::size_t p   = 0;
                    dfor(channels,
                         kernel_h,
                         kernel_w)([&](std::size_t c, std::size_t koffset, std::size_t loffset) {
                        auto idx    = iinput + long(koffset) - kdiv2_h;
                        auto jdx    = jinput + long(loffset) - kdiv2_w;
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
    op::pooling op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "cpu::pooling_" + Op::name(); }
    shape compute_shape(const std::vector<shape>& inputs) const { return op.compute_shape(inputs); }
    argument compute(context&, const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0])([&](auto output, auto input) {
            using type = typename decltype(output)::value_type;
            auto in_h  = input.get_shape().lens()[2];
            auto in_w  = input.get_shape().lens()[3];

            par_dfor(output_shape.lens()[0],
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

struct cpu_op
{
    operation op;
    std::string name() const { return "cpu::" + op.name(); }
    shape compute_shape(const std::vector<shape>& inputs) const { return op.compute_shape(inputs); }
    argument compute(context&, const shape& output_shape, const std::vector<argument>& args) const
    {
        return op.compute(output_shape, args);
    }
    friend bool operator==(const cpu_op& x, const cpu_op& y) { return x.op == y.op; }
    friend bool operator==(const cpu_op& x, const operation& y)
    {
        if(x.name() != y.name())
            return false;
        return x == any_cast<cpu_op>(y);
    }
    friend bool operator==(const operation& x, const cpu_op& y) { return y == x; }
};

struct cpu_pad
{
    op::pad op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "cpu::contiguous"; }
    shape compute_shape(const std::vector<shape>& inputs) const { return op.compute_shape(inputs); }
    argument compute(context&, const shape& output_shape, std::vector<argument> args) const
    {
        assert(output_shape.standard());
        argument result{output_shape};
        result.visit([&](auto output) {
            using type = typename decltype(output)::value_type;
            std::fill(output.begin(), output.end(), pad_clamp<type>(op.value));
        });

        visit_all(result, args[0])([&](auto output, auto input) {
            shape_for_each(input.get_shape(), [&](const auto& idx) {
                std::vector<std::size_t> new_idx(idx.size());
                std::transform(
                    idx.begin(), idx.end(), op.pads.begin(), new_idx.begin(), [](auto i, auto j) {
                        return i + j;
                    });
                output(new_idx.begin(), new_idx.end()) = input(idx.begin(), idx.end());
            });
        });

        return result;
    }
};

struct cpu_gemm
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
        if(inputs.size() == 3)
        {
            auto c_shape = inputs.at(2);
            check_shapes{{c_shape}}.not_broadcasted();
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

struct cpu_quant_gemm
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
        if(inputs.size() == 3)
        {
            auto c_shape = inputs.at(2);
            check_shapes{{c_shape}}.not_broadcasted();
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

struct leaky_relu_op
{
    op::leaky_relu op;
    std::string name() const { return "cpu::leaky_relu"; }
    auto fcn() const
    {
        auto a = op.alpha;
        return [a](auto x) { return x > 0 ? x : x * a; };
    }
};

struct elu_op
{
    op::elu op;
    std::string name() const { return "cpu::elu"; }
    auto fcn() const
    {
        auto a = op.alpha;
        return [a](auto x) { return x > 0 ? x : a * std::expm1(x); };
    }
};

template <typename Op>
struct cpu_unary
{
    Op op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op.op, f);
    }
    std::string name() const { return op.name(); }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs}.has(1);
        auto s = inputs.at(0);
        return {s.type(), s.lens()};
    }

    argument compute(context&, const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0])([&](auto output, auto input) {
            assert(input.get_shape().standard());
            std::transform(input.begin(), input.end(), output.begin(), op.fcn());
        });

        return result;
    }
};

template <class Op>
struct cpu_softmax
{
    Op op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "cpu::" + op.name(); }
    shape compute_shape(const std::vector<shape>& inputs) const { return op.compute_shape(inputs); }
    argument compute(context&, const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto batch_lens    = output_shape.lens();
        int64_t tuned_axis = (op.axis < 0) ? op.axis + args[0].get_shape().lens().size() : op.axis;
        std::size_t n_dims = batch_lens[tuned_axis];
        batch_lens[tuned_axis] = 1;
        shape batch_shape{shape::int32_type, batch_lens};

        visit_all(result, args[0])([&](auto output, auto input) {
            using value_type = typename decltype(input)::value_type;
            std::vector<value_type> batch_max(batch_shape.elements(),
                                              std::numeric_limits<value_type>::lowest());
            std::vector<value_type> batch_sum(batch_shape.elements(), value_type(0));
            par_for(batch_shape.elements(), [&](auto i) {
                auto idx = batch_shape.multi(i);
                for(std::size_t j = 0; j < n_dims; ++j)
                {
                    idx[tuned_axis] = j;
                    batch_max[i]    = std::max(batch_max[i], input(idx.begin(), idx.end()));
                }

                for(std::size_t j = 0; j < n_dims; ++j)
                {
                    idx[tuned_axis]   = j;
                    std::size_t index = output_shape.index(idx);
                    output[index]     = std::exp(input[index] - batch_max[i]);
                }

                for(std::size_t j = 0; j < n_dims; ++j)
                {
                    idx[tuned_axis] = j;
                    batch_sum[i] += output(idx.begin(), idx.end());
                }

                for(std::size_t j = 0; j < n_dims; ++j)
                {
                    idx[tuned_axis] = j;
                    output(idx.begin(), idx.end()) =
                        op.output()(output(idx.begin(), idx.end()), batch_sum[i]);
                }
            });
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
        apply_map["batch_norm_inference"] =
            extend_op<cpu_batch_norm_inference, op::batch_norm_inference>();
        apply_map["convolution"] = extend_op<cpu_convolution<op::convolution>, op::convolution>();
        apply_map["deconvolution"] =
            extend_op<cpu_deconvolution<op::deconvolution>, op::deconvolution>();
        apply_map["dot"]       = extend_op<cpu_gemm, op::dot>();
        apply_map["quant_dot"] = extend_op<cpu_quant_gemm, op::quant_dot>();
        apply_map["quant_convolution"] =
            extend_op<cpu_convolution<op::quant_convolution>, op::quant_convolution>();
        apply_map["elu"]        = extend_op<cpu_unary<elu_op>, op::elu>();
        apply_map["im2col"]     = extend_op<cpu_im2col, op::im2col>();
        apply_map["leaky_relu"] = extend_op<cpu_unary<leaky_relu_op>, op::leaky_relu>();
        apply_map["logsoftmax"] = extend_op<cpu_softmax<op::logsoftmax>, op::logsoftmax>();
        apply_map["lrn"]        = extend_op<cpu_lrn, op::lrn>();
        apply_map["pad"]        = extend_op<cpu_pad, op::pad>();
        apply_map["softmax"]    = extend_op<cpu_softmax<op::softmax>, op::softmax>();
    }

    void apply()
    {
        init();
        for(auto it : iterator_for(*prog))
        {
            if(it->name() == "pooling")
            {
                apply_pooling(it);
            }
            else if(apply_map.count(it->name()) > 0)
            {
                apply_map.at(it->name())(it);
            }
            else if(is_context_free(it->get_operator()))
            {
                apply_cpu_op(it);
            }
        }
    }

    void apply_cpu_op(instruction_ref ins)
    {
        prog->replace_instruction(ins, cpu_op{ins->get_operator()}, ins->inputs());
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

    void apply_pooling(instruction_ref ins)
    {
        auto&& op = any_cast<op::pooling>(ins->get_operator());
        if(op.mode == "max")
            prog->replace_instruction(ins, cpu_pooling<max_pool>{op}, ins->inputs());
        else if(op.mode == "average")
            prog->replace_instruction(ins, cpu_pooling<avg_pool>{op}, ins->inputs());
    }
};

void lowering::apply(program& p) const { cpu_apply{&p}.apply(); }

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
