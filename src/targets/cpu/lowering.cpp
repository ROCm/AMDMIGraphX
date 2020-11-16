
#include <migraphx/cpu/lowering.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/op/batch_norm_inference.hpp>
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
#include <migraphx/op/rnn_var_sl_last_output.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/par_dfor.hpp>
#include <migraphx/clamp.hpp>
#include <migraphx/cpu/gemm.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <unordered_map>
#include <utility>
#include <iostream>

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

        if(op.bn_mode == op::batch_norm_inference::spatial)
        {
            visit_all(output, input, mini_batch_mean, mini_batch_variance, arg_gamma, arg_bias)(
                [&](auto result, auto buffer, auto mean, auto variance, auto gamma, auto bias) {
                    par_for(output_shape.elements(), [&](auto i) {
                        auto idx = output_shape.multi(i);
                        auto c   = idx[1];
                        assert((variance[c] + epsilon) > 0);
                        result[i] =
                            gamma[c] * (buffer[i] - mean[c]) / std::sqrt(variance[c] + epsilon) +
                            bias[c];
                    });
                });
        }

        if(op.bn_mode == op::batch_norm_inference::per_activation)
        {
            visit_all(output, input, mini_batch_mean, mini_batch_variance, arg_gamma, arg_bias)(
                [&](auto result, auto buffer, auto mean, auto variance, auto gamma, auto bias) {
                    par_for(output_shape.elements(), [&](auto i) {
                        auto idx   = output_shape.multi(i);
                        idx[0]     = 0;
                        auto index = output_shape.index(idx);

                        assert((variance[index] + epsilon) > 0);
                        result[i] = gamma[index] * (buffer[i] - mean[index]) /
                                        std::sqrt(variance[index] + epsilon) +
                                    bias[index];
                    });
                });
        }

        return output;
    }
};
MIGRAPHX_REGISTER_OP(cpu_batch_norm_inference)

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
MIGRAPHX_REGISTER_OP(cpu_lrn)

template <class V, class T, class... Ts>
void visit_quantize_impl(V&& v, T&& x, Ts&&... xs)
{
    x.visit([&](auto y) { visit_all(xs...)([&](auto... ys) { v(y, ys...); }); });
}

template <class T, class... Ts>
auto visit_quantize(T&& x, Ts&&... xs)
{
    return [&](auto v) {
        // Workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70100
        visit_quantize_impl(v, x, xs...);
    };
}

template <class Op>
struct cpu_convolution : auto_register_op<cpu_convolution<Op>>
{
    cpu_convolution() = default;

    cpu_convolution(Op pop) : op(std::move(pop)) {}

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
        visit_quantize(result, args[0], args[1])([&](auto output, auto input, auto weights) {
            auto in_lens = input.get_shape().lens();

            auto wei_lens = weights.get_shape().lens();
            auto wei_n    = wei_lens[0];
            auto wei_c    = wei_lens[1];
            std::vector<std::size_t> win_size(wei_lens.begin() + 1, wei_lens.end());

            par_for(output_shape.elements(), [&](auto i) {
                auto idx_o = output_shape.multi(i);
                auto w     = idx_o[1];
                auto n_dim = idx_o.size();

                std::vector<std::ptrdiff_t> win_start;
                for(std::size_t dim = 2; dim < n_dim; ++dim)
                {
                    auto d_2 = dim - 2;
                    win_start.push_back(std::ptrdiff_t(idx_o[dim] * op.stride[d_2]) -
                                        std::ptrdiff_t(op.padding[d_2]));
                }
                const auto group_id = w / (wei_n / op.group);

                shape win_shape{output_shape.type(), win_size};

                double acc = 0.0;
                shape_for_each(win_shape, [&](auto idx_win) {
                    auto k           = idx_win[0];
                    const auto in_ch = group_id * wei_c + k;
                    std::vector<std::ptrdiff_t> idx(idx_o.begin(), idx_o.end());
                    idx[1] = in_ch;
                    std::transform(idx_win.begin() + 1,
                                   idx_win.end(),
                                   win_start.begin(),
                                   idx.begin() + 2,
                                   [](std::ptrdiff_t ii, std::ptrdiff_t jj) { return ii + jj; });
                    std::vector<std::ptrdiff_t> idx_wei(idx_o.size());
                    idx_wei[0] = w;
                    std::copy(idx_win.begin(), idx_win.end(), idx_wei.begin() + 1);
                    if(std::all_of(idx.begin() + 2, idx.end(), [&](auto ii) { return ii >= 0; }) and
                       std::equal(idx.begin(),
                                  idx.end(),
                                  in_lens.begin(),
                                  in_lens.end(),
                                  std::less<std::ptrdiff_t>{}))
                    {
                        acc +=
                            input(idx.begin(), idx.end()) * weights(idx_wei.begin(), idx_wei.end());
                    }
                });

                output[i] = acc;
            });
        });
        return result;
    }
};

template <class Op>
struct cpu_deconvolution : auto_register_op<cpu_deconvolution<Op>>
{
    cpu_deconvolution() = default;

    cpu_deconvolution(Op pop) : op(std::move(pop)) {}

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

            auto in_lens = input.get_shape().lens();
            auto in_n    = in_lens[0];
            auto in_c    = in_lens[1];

            auto wei   = weights.get_shape().lens();
            auto wei_n = wei[0];
            auto wei_c = wei[1];

            auto out_lens = output_shape.lens();
            auto kdims    = op.kdims();

            std::vector<std::size_t> win_size{in_c};
            std::copy(in_lens.begin() + 2, in_lens.end(), std::back_inserter(win_size));
            std::copy(wei.begin() + 2, wei.end(), std::back_inserter(win_size));
            shape win_shape{output_shape.type(), win_size};

            par_dfor(in_n, wei_c)([&](int o, int k) {

                shape_for_each(win_shape, [&](auto idx_win) {
                    const int w = idx_win[0];

                    auto input_dims_start = idx_win.begin() + 1;
                    auto wei_dims_start   = idx_win.begin() + kdims + 1;

                    std::vector<std::ptrdiff_t> win_start;
                    for(std::size_t n = 0; n < kdims; ++n)
                    {
                        win_start.push_back(std::ptrdiff_t(*(input_dims_start + n) * op.stride[n]) -
                                            std::ptrdiff_t(op.padding[n]));
                    }

                    const int group_id = w / (wei_n / op.group);
                    const int in_ch    = group_id * wei_c + k;

                    std::vector<std::ptrdiff_t> idx_out{o, in_ch};

                    for(size_t n = 0; n < kdims; n++)
                    {
                        idx_out.push_back(win_start[n] + *(wei_dims_start + n) * op.dilation[n]);
                    }

                    std::vector<std::ptrdiff_t> idx_wei{w, k};
                    std::copy(wei_dims_start, idx_win.end(), std::back_inserter(idx_wei));

                    std::vector<std::ptrdiff_t> idx_in{o, w};
                    std::copy(input_dims_start, wei_dims_start, std::back_inserter(idx_in));

                    if(std::all_of(
                           idx_out.begin() + 2, idx_out.end(), [&](auto ii) { return ii >= 0; }) and
                       std::equal(idx_out.begin() + 2,
                                  idx_out.end(),
                                  out_lens.begin() + 2,
                                  out_lens.end(),
                                  std::less<std::ptrdiff_t>{}))
                    {
                        output(idx_out.begin(), idx_out.end()) +=
                            input(idx_in.begin(), idx_in.end()) *
                            weights(idx_wei.begin(), idx_wei.end());
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
MIGRAPHX_REGISTER_OP(cpu_im2col)

struct max_pool
{
    static std::string name() { return "max"; }
    template <class T>
    static T start()
    {
        return std::numeric_limits<T>::lowest();
    }

    static double apply(double x, double y)
    {
        double m = std::max(x, y);
        return (m);
    }

    static double final(double x, std::size_t) { return (x); }
};

struct avg_pool
{
    static std::string name() { return "average"; }

    template <class T>
    static double start()
    {
        return 0.0;
    }

    static double apply(double x, double y) { return x + y; }

    static double final(double x, std::size_t y) { return (y == 0) ? 0.0 : (x / y); }
};

template <class Op>
struct cpu_pooling : auto_register_op<cpu_pooling<Op>>
{
    cpu_pooling() = default;

    cpu_pooling(op::pooling pop) : op(std::move(pop)) {}

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
            using type   = typename decltype(output)::value_type;
            auto in_s    = input.get_shape();
            auto in_lens = in_s.lens();
            std::vector<std::size_t> vec_len(in_lens.begin() + 2, in_lens.end());

            par_for(output_shape.elements(), [&](auto i) {
                auto idx_o = output_shape.multi(i);
                auto n_dim = idx_o.size();
                std::vector<std::size_t> win_start;
                std::vector<std::size_t> win_size;
                for(std::size_t dim = 2; dim < n_dim; ++dim)
                {
                    auto d_2  = dim - 2;
                    int start = static_cast<int>(idx_o[dim] * op.stride[d_2]) -
                                static_cast<int>(op.padding[d_2]);
                    int end = std::min(start + op.lengths[d_2], in_lens[dim]);
                    start   = std::max(start, 0);
                    win_start.push_back(start);
                    win_size.push_back(end - start);
                }

                shape win_shape{output_shape.type(), win_size};
                auto pool_size = win_shape.elements();
                double acc     = Op::template start<type>();
                shape_for_each(win_shape, [&](auto idx_w) {
                    auto idx = idx_o;
                    std::transform(idx_w.begin(),
                                   idx_w.end(),
                                   win_start.begin(),
                                   idx.begin() + 2,
                                   [](auto ii, auto jj) { return ii + jj; });
                    if(std::all_of(idx.begin() + 2, idx.end(), [&](auto ii) { return ii >= 0; }) and
                       idx < in_lens)
                    {
                        acc = Op::apply(acc, input[in_s.index(idx)]);
                    }
                });

                output[i] = type(Op::final(acc, pool_size));
            });
        });

        return result;
    }
};

struct cpu_op
{
    operation op = op::identity{};
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }
    std::string name() const { return "cpu::op"; }
    shape compute_shape(const std::vector<shape>& inputs) const { return op.compute_shape(inputs); }
    argument compute(context&, const shape& output_shape, const std::vector<argument>& args) const
    {
        return op.compute(output_shape, args);
    }
    value to_value() const
    {
        value v;
        v["name"]     = op.name();
        v["operator"] = op.to_value();
        return v;
    }
    void from_value(const value& v)
    {
        op = make_op(v.at("name").to<std::string>(), v.at("operator"));
    }
    friend std::ostream& operator<<(std::ostream& os, const cpu_op& x)
    {
        os << "cpu::" << x.op;
        return os;
    }
};
MIGRAPHX_REGISTER_OP(cpu_op)

struct cpu_pad
{
    op::pad op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "cpu::pad"; }
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
MIGRAPHX_REGISTER_OP(cpu_pad)

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
MIGRAPHX_REGISTER_OP(cpu_gemm)

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
MIGRAPHX_REGISTER_OP(cpu_gemm)

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
struct cpu_unary : auto_register_op<cpu_unary<Op>>
{
    cpu_unary() = default;

    template <class T>
    cpu_unary(T pop) : op(Op{std::move(pop)})
    {
    }

    Op op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op.op, f);
    }
    std::string name() const { return op.name(); }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(1);
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
struct cpu_softmax : auto_register_op<cpu_softmax<Op>>
{
    cpu_softmax() = default;

    cpu_softmax(Op pop) : op(std::move(pop)) {}

    Op op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "cpu::" + op.name(); }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        return op.normalize_compute_shape(inputs);
    }
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

struct cpu_rnn_var_sl_last_output
{
    op::rnn_var_sl_last_output op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "cpu::rnn_var_sl_last_output"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        return op.compute_shape(std::move(inputs));
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto out_comp_lens = args[0].get_shape().lens();
        out_comp_lens[0]   = 1;
        shape out_comp_s{output_shape.type(), out_comp_lens};

        visit_all(result, args[0])([&](auto output, auto input) {
            args[1].visit([&](auto seq_lens) {
                par_for(output_shape.elements(), [&](auto i) {
                    auto idx = out_comp_s.multi(i);
                    auto b   = idx[2];
                    if(op.direction == op::rnn_direction::reverse or idx[1] == 1)
                    {
                        idx[0] = 0;
                    }
                    else
                    {
                        idx[0] = seq_lens[b] - 1;
                    }
                    output[i] = input(idx.begin(), idx.end());
                });
            });
        });

        return result;
    }
};
MIGRAPHX_REGISTER_OP(cpu_rnn_var_sl_last_output)

struct cpu_apply
{
    module* prog;
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
        apply_map["rnn_var_sl_last_output"] =
            extend_op<cpu_rnn_var_sl_last_output, op::rnn_var_sl_last_output>();
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

    void apply_cpu_op(instruction_ref ins) const
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

    void apply_pooling(instruction_ref ins) const
    {
        auto&& op = any_cast<op::pooling>(ins->get_operator());
        if(op.mode == "max")
            prog->replace_instruction(ins, cpu_pooling<max_pool>{op}, ins->inputs());
        else if(op.mode == "average")
            prog->replace_instruction(ins, cpu_pooling<avg_pool>{op}, ins->inputs());
    }
};

void lowering::apply(module& p) const { cpu_apply{&p}.apply(); }

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
