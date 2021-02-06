
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
#include <migraphx/cpu/context.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/tune_axis.hpp>
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
        auto batch_lens        = output_shape.lens();
        int64_t tuned_axis     = tune_axis(args[0].get_shape().lens().size(), op.axis, op.name());
        std::size_t n_dims     = batch_lens[tuned_axis];
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

struct cpu_literal
{
    argument data;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.data, "data"));
    }

    std::string name() const { return "cpu::literal"; }

    shape compute_shape(const std::vector<shape>&) const { return data.get_shape(); }

    argument compute(const shape&, const std::vector<argument>&) const { return data; }

    friend std::ostream& operator<<(std::ostream& os, const cpu_literal& x)
    {
        os << x.name();
        return os;
    }
};

struct cpu_apply
{
    module* modl;
    std::unordered_map<std::string, std::function<instruction_ref(instruction_ref)>> apply_map{};
    std::unordered_map<instruction_ref, std::string> prog_output_names{};
    instruction_ref last{};

    void create_output_names()
    {
        this->last = instruction::get_output_alias(std::prev(modl->end()));
        if(this->last->name() == "@return")
        {
            const auto& prog_outputs = last->inputs();
            std::vector<instruction_ref> outputs_alias(prog_outputs.size());

            std::transform(prog_outputs.begin(),
                           prog_outputs.end(),
                           outputs_alias.begin(),
                           [](const auto& i) { return instruction::get_output_alias(i); });

            std::size_t index = 0;
            for(auto ins : outputs_alias)
            {
                prog_output_names[ins] = "#output_" + std::to_string(index++);
            }
        }
    }

    void extend_op(const std::string& op_name, const std::string& cpu_name, bool allocate = true)
    {
        apply_map.emplace(op_name, [=](instruction_ref ins) {
            auto&& op = ins->get_operator();
            if(allocate)
                return replace(ins, make_op(cpu_name, op.to_value()));
            return modl->replace_instruction(ins, make_op(cpu_name, op.to_value()), ins->inputs());
        });
    }

    template <class F>
    void extend_dnnl_algos(const std::string& dnnl_name,
                           F f,
                           const std::vector<std::pair<std::string, std::string>>& algos)
    {
        for(auto&& pp : algos)
        {
            std::string op_name = pp.first;
            std::string algo    = pp.second;
            apply_map.emplace(op_name, [=](instruction_ref ins) {
                auto v = f(ins);
                if(not v.is_object())
                    return ins;
                v["algo"] = algo;
                auto op   = make_op(dnnl_name, v);
                return replace(ins, op);
            });
        }
    }

    void extend_dnnl_algos(const std::string& dnnl_name,
                           const std::vector<std::pair<std::string, std::string>>& algos)
    {
        return extend_dnnl_algos(dnnl_name,
                                 [](auto ins) -> value {
                                     auto v = ins->get_operator().to_value();
                                     if(not v.is_object())
                                         return value::object{};
                                     return v;
                                 },
                                 algos);
    }

    void init()
    {
        create_output_names();
        extend_dnnl_algos("dnnl::binary",
                          {
                              {"add", "binary_add"},
                              {"div", "binary_div"},
                              {"max", "binary_max"},
                              {"min", "binary_min"},
                              {"mul", "binary_mul"},
                          });

        extend_dnnl_algos("dnnl::eltwise",
                          {
                              {"abs", "eltwise_abs"},
                              {"elu", "eltwise_elu"},
                              {"exp", "eltwise_exp"},
                              {"log", "eltwise_log"},
                              {"relu", "eltwise_relu"},
                              {"sqrt", "eltwise_sqrt"},
                              {"tanh", "eltwise_tanh"},
                          });

        extend_dnnl_algos("dnnl::reduction",
                          {
                              {"reduce_max", "reduction_max"},
                              {"reduce_mean", "reduction_mean"},
                              {"reduce_min", "reduction_min"},
                              {"reduce_sum", "reduction_sum"},
                          });

        extend_op("softmax", "dnnl::softmax");
        extend_op("logsoftmax", "dnnl::logsoftmax");
        extend_op("convolution", "dnnl::convolution");
        extend_op("deconvolution", "dnnl::deconvolution");
        extend_op("dot", "dnnl::dot");
        extend_op("lrn", "dnnl::lrn");
        extend_op("contiguous", "dnnl::reorder");

        extend_op("im2col", "cpu::im2col", false);
        extend_op("leaky_relu", "cpu::leaky_relu", false);
        extend_op("pad", "cpu::pad", false);
        extend_op("rnn_var_sl_last_output", "cpu::rnn_var_sl_last_output", false);
    }

    void apply()
    {
        init();
        for(auto it : iterator_for(*modl))
        {
            if(it->name() == "pooling")
            {
                apply_pooling(it);
            }
            else if(it->name() == "pow")
            {
                apply_pow(it);
            }
            else if(apply_map.count(it->name()) > 0)
            {
                apply_map.at(it->name())(it);
            }
        }
        for(auto it : iterator_for(*modl))
        {
            if(it->name() == "@literal")
            {
                apply_literal(it);
            }
        }
    }

    instruction_ref apply_literal(instruction_ref ins) const
    {
        return modl->replace_instruction(ins, cpu_literal{ins->get_literal().get_argument()});
    }

    instruction_ref apply_pow(instruction_ref ins)
    {
        auto e = ins->inputs()[1]->eval();
        if (e.empty())
            return ins;
        if (not e.single() and not e.get_shape().scalar())
            return ins;
        float beta = e.at<float>();
        return replace(ins, make_op("dnnl::eltwise", {{"algo", "eltwise_pow"}, {"alpha", 1.0}, {"beta", beta}}), {ins->inputs().front()});
    }

    instruction_ref apply_pooling(instruction_ref ins)
    {
        auto&& op = ins->get_operator();
        auto v    = op.to_value();
        if(has_op("dnnl::pooling") and ins->get_shape().type() == shape::type_t::float_type and
           not v["ceil_mode"].to<bool>())
            return replace(ins, make_op("dnnl::pooling", op.to_value()));
        std::string mode = v["mode"].to<std::string>();
        if(mode == "max")
            return replace(ins, make_op("cpu::pooling_max", v));
        else if(mode == "average")
            return replace(ins, make_op("cpu::pooling_average", v));
        return ins;
    }

    instruction_ref replace(instruction_ref ins, const operation& op)
    {
        return replace(ins, op, ins->inputs());
    }

    instruction_ref replace(instruction_ref ins, const operation& op, std::vector<instruction_ref> inputs)
    {
        inputs.push_back(insert_allocation(ins, ins->get_shape()));
        return modl->replace_instruction(ins, op, inputs);
    }

    instruction_ref insert_allocation(instruction_ref ins, const shape& s)
    {
        auto ins_alias = instruction::get_output_alias(ins);
        if(last->name() == "@return" and prog_output_names.count(ins_alias) > 0)
        {
            return modl->add_parameter(prog_output_names[ins_alias], s);
        }
        else if(ins == last)
        {
            return modl->add_parameter("output", s);
        }

        return modl->insert_instruction(ins, make_op("cpu::allocate", {{"shape", to_value(s)}}));
    }
};

void lowering::apply(module& m) const { cpu_apply{&m}.apply(); }

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
