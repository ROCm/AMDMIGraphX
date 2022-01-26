
#include <migraphx/fpga/lowering.hpp>
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
#include <migraphx/op/loop.hpp>
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
#include <migraphx/register_op.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>
#include <unordered_map>
#include <utility>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace fpga {

struct fpga_op
{
    operation op = op::identity{};
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }
    std::string name() const { return "fpga::op"; }
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
    friend std::ostream& operator<<(std::ostream& os, const fpga_op& x)
    {
        os << "fpga::" << x.op;
        return os;
    }
};
MIGRAPHX_REGISTER_OP(fpga_op)

struct fpga_apply
{
    module* mod;
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
	/*
        apply_map["batch_norm_inference"] =
            extend_op<ref_batch_norm_inference, op::batch_norm_inference>();
        apply_map["convolution"] = extend_op<ref_convolution<op::convolution>, op::convolution>();
        apply_map["dot"]         = extend_op<ref_gemm, op::dot>();
        apply_map["quant_dot"]   = extend_op<ref_quant_gemm, op::quant_dot>();
        apply_map["quant_convolution"] =
            extend_op<ref_convolution<op::quant_convolution>, op::quant_convolution>();
        apply_map["elu"]        = extend_op<ref_unary<elu_op>, op::elu>();
        apply_map["im2col"]     = extend_op<ref_im2col, op::im2col>();
        apply_map["leaky_relu"] = extend_op<ref_unary<leaky_relu_op>, op::leaky_relu>();
        apply_map["logsoftmax"] = extend_op<ref_softmax<op::logsoftmax>, op::logsoftmax>();
        apply_map["lrn"]        = extend_op<ref_lrn, op::lrn>();
        apply_map["pad"]        = extend_op<ref_pad, op::pad>();
        apply_map["softmax"]    = extend_op<ref_softmax<op::softmax>, op::softmax>();
        apply_map["rnn_var_sl_last_output"] =
            extend_op<ref_rnn_var_sl_last_output, op::rnn_var_sl_last_output>();
        */
    }

    void apply()
    {
	// Here is where we need to fuse several FPGA runnable ops together into a single operation.
        
	// Print out all the ops    
	for(auto it : iterator_for(*mod)) {
            std::cout << it->name() << std::endl;
	}
	/*
	init();
        for(auto it : iterator_for(*mod))
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
                apply_fpga_op(it);
            }
        }
        */
    
    }

    void apply_fpga_op(instruction_ref ins) const
    {
        mod->replace_instruction(ins, fpga_op{ins->get_operator()}, ins->inputs());
    }

    template <class T>
    void apply_simple_op(instruction_ref ins)
    {
        mod->replace_instruction(ins, T{}, ins->inputs());
    }

    template <class T, class Op>
    void apply_extend_op(instruction_ref ins)
    {
        auto&& op = any_cast<Op>(ins->get_operator());
        mod->replace_instruction(ins, T{op}, ins->inputs());
    }

};

void lowering::apply(module& m) const { fpga_apply{&m}.apply(); }

} // namespace fpga
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
