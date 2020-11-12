
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
#include <migraphx/cpu/migemm.hpp>
#include <migraphx/cpu/context.hpp>
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
template struct cpu_deconvolution<op::deconvolution>;

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
struct cpu_unary2 : auto_register_op<cpu_unary2<Op>>
{
    cpu_unary2() = default;

    template <class T>
    cpu_unary2(T pop) : op(Op{std::move(pop)})
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
template struct cpu_unary2<leaky_relu_op>;
template struct cpu_unary2<elu_op>;

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
template struct cpu_softmax<op::softmax>;
template struct cpu_softmax<op::logsoftmax>;

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
    std::unordered_map<std::string, std::function<instruction_ref(instruction_ref)>> apply_map{};

    void extend_op(const std::string& op_name, const std::string& cpu_name)
    {
        apply_map.emplace(op_name, [=](instruction_ref ins) {
            auto&& op = ins->get_operator();
            return prog->replace_instruction(ins, make_op(cpu_name, op.to_value()), ins->inputs());
        });
    }

    void extend_dnnl_op(const std::string& op_name,
                        const std::string& cpu_name,
                        const std::string& dnnl_name)
    {
        apply_map.emplace(op_name, [=](instruction_ref ins) {
            auto&& op = ins->get_operator();
            if(has_op(dnnl_name) and ins->get_shape().type() == shape::type_t::float_type)
                return prog->replace_instruction(
                    ins, make_op(dnnl_name, op.to_value()), ins->inputs());
            return prog->replace_instruction(ins, make_op(cpu_name, op.to_value()), ins->inputs());
        });
    }

    void extend_dnnl_op(const std::string& op_name,
                        const std::string& dnnl_name)
    {
        apply_map.emplace(op_name, [=](instruction_ref ins) {
            auto&& op = ins->get_operator();
            if(has_op(dnnl_name) and ins->get_shape().type() == shape::type_t::float_type)
                return prog->replace_instruction(
                    ins, make_op(dnnl_name, op.to_value()), ins->inputs());
            return ins;
        });
    }

    void init()
    {
        extend_dnnl_op("add", "cpu::add", "dnnl::add");
        extend_dnnl_op("convolution", "cpu::convolution", "dnnl::convolution");
        extend_dnnl_op("dot", "cpu::dot", "dnnl::dot");
        extend_dnnl_op("concat", "dnnl::concat");
        extend_op("add", "cpu::add");
        extend_op("batch_norm_inference", "cpu::batch_norm_inference");
        extend_op("deconvolution", "cpu::deconvolution");
        extend_op("elu", "cpu::elu");
        extend_op("im2col", "cpu::im2col");
        extend_op("leaky_relu", "cpu::leaky_relu");
        extend_op("logsoftmax", "cpu::logsoftmax");
        extend_op("lrn", "cpu::lrn");
        extend_op("pad", "cpu::pad");
        extend_op("quant_convolution", "cpu::quant_convolution");
        extend_op("quant_dot", "cpu::quant_dot");
        extend_op("rnn_var_sl_last_output", "cpu::rnn_var_sl_last_output");
        extend_op("softmax", "cpu::softmax");
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
        }
    }

    instruction_ref apply_pooling(instruction_ref ins) const
    {
        auto&& op = ins->get_operator();
        if(has_op("dnnl::pooling") and ins->get_shape().type() == shape::type_t::float_type)
            return prog->replace_instruction(ins, make_op("dnnl::pooling", op.to_value()), ins->inputs());
        std::string mode = op.to_value()["mode"].to<std::string>();
        if(mode == "max")
            return prog->replace_instruction(ins, make_op("cpu::pooling_max", op.to_value()), ins->inputs());
        else if(mode == "average")
            return prog->replace_instruction(ins, make_op("cpu::pooling_avgerage", op.to_value()), ins->inputs());
        return ins;
    }
};

void lowering::apply(module& p) const { cpu_apply{&p}.apply(); }

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
