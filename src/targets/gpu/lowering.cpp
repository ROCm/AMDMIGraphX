#include <migraphx/gpu/lowering.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/op/abs.hpp>
#include <migraphx/op/batch_norm_inference.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/deconvolution.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/elu.hpp>
#include <migraphx/op/if_op.hpp>
#include <migraphx/op/leaky_relu.hpp>
#include <migraphx/op/lrn.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/op/quant_convolution.hpp>
#include <migraphx/op/quant_dot.hpp>

#include <migraphx/gpu/abs.hpp>
#include <migraphx/gpu/batch_norm_inference.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/convolution.hpp>
#include <migraphx/gpu/deconvolution.hpp>
#include <migraphx/gpu/elu.hpp>
#include <migraphx/gpu/equal.hpp>
#include <migraphx/gpu/gemm.hpp>
#include <migraphx/gpu/greater.hpp>
#include <migraphx/gpu/int8_conv_pack.hpp>
#include <migraphx/gpu/leaky_relu.hpp>
#include <migraphx/gpu/less.hpp>
#include <migraphx/gpu/logical_and.hpp>
#include <migraphx/gpu/logical_or.hpp>
#include <migraphx/gpu/logical_xor.hpp>
#include <migraphx/gpu/lrn.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <migraphx/gpu/quant_convolution.hpp>
#include <migraphx/gpu/rocblas.hpp>
#include <migraphx/gpu/unary_not.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/program.hpp>
#include <utility>
#include <functional>
#include <algorithm>
#include <map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct miopen_apply
{
    module* mod          = nullptr;
    const lowering* pass = nullptr;
    std::unordered_map<std::string, std::function<instruction_ref(instruction_ref)>> apply_map{};
    instruction_ref last{};
    std::unordered_map<instruction_ref, std::string> prog_output_names{};
    bool offload_copy   = false;
    bool int8_x4_format = true;

    context& get_context() const
    {
        assert(pass != nullptr);
        assert(pass->ctx != nullptr);
        return *pass->ctx;
    }

    void check_shape(shape x, instruction_ref i)
    {
        assert(x == i->get_shape());
        (void)x;
        (void)i;
    }

    void create_output_names()
    {
        this->last = instruction::get_output_alias(std::prev(mod->end()));
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
                prog_output_names[ins] = mod->name() + ":#output_" + std::to_string(index++);
            }
        }
    }

    void init()
    {
        assert(mod != nullptr);
        assert(pass != nullptr);

#if ROCBLAS_VERSION_MAJOR >= 2 && ROCBLAS_VERSION_MINOR >= 38
        auto& ctx = get_context();
        rocblas_gemm_flags flag;
        rocblas_query_int8_layout_flag(ctx.get_stream().get_rocblas(), &flag);
        int8_x4_format = (flag == rocblas_gemm_flags_pack_int8x4);
#endif

        offload_copy = (mod->name() == "main") ? pass->offload_copy : false;
        create_output_names();

        add_generic_op("acos");
        add_generic_op("acosh");
        add_generic_op("add");
        add_generic_op("asin");
        add_generic_op("asinh");
        add_generic_op("atan");
        add_generic_op("atanh");
        add_generic_op("ceil");
        add_generic_op("contiguous");
        add_generic_op("cos");
        add_generic_op("cosh");
        add_generic_op("div");
        add_generic_op("equal");
        add_generic_op("erf");
        add_generic_op("exp");
        add_generic_op("floor");
        add_generic_op("greater");
        add_generic_op("less");
        add_generic_op("log");
        add_generic_op("logical_and");
        add_generic_op("logical_or");
        add_generic_op("logical_xor");
        add_generic_op("max");
        add_generic_op("min");
        add_generic_op("mul");
        add_generic_op("not");
        add_generic_op("pow");
        add_generic_op("prelu");
        add_generic_op("recip");
        add_generic_op("relu");
        add_generic_op("round");
        add_generic_op("rsqrt");
        add_generic_op("sigmoid");
        add_generic_op("sign");
        add_generic_op("sin");
        add_generic_op("sinh");
        add_generic_op("sqdiff");
        add_generic_op("sqrt");
        add_generic_op("sub");
        add_generic_op("tan");
        add_generic_op("tanh");

        add_extend_op("abs");
        add_extend_op("argmax");
        add_extend_op("argmin");
        add_extend_op("clip");
        add_extend_op("concat");
        add_extend_op("convert");
        add_extend_op("elu");
        add_extend_op("gather");
        add_extend_op("leaky_relu");
        add_extend_op("logsoftmax");
        add_extend_op("lrn");
        add_extend_op("pad");
        add_extend_op("pooling");
        add_extend_op("prefix_scan_sum");
        add_extend_op("reduce_max");
        add_extend_op("reduce_mean");
        add_extend_op("reduce_min");
        add_extend_op("reduce_prod");
        add_extend_op("reduce_sum");
        add_extend_op("reverse");
        add_extend_op("rnn_var_sl_last_output");
        add_extend_op("rnn_var_sl_shift_output");
        add_extend_op("rnn_var_sl_shift_sequence");
        add_extend_op("scatter");
        add_extend_op("softmax");

        add_gemm_op<op::dot>("dot");
        add_gemm_op<op::quant_dot>("quant_dot");
        add_convolution_op();
        add_deconvolution_op();
        add_quant_convolution_op();
        add_batch_norm_inference_op();
        add_neg_op();
        add_if_op();
    }

    void copy_params()
    {
        if(not offload_copy)
            return;

        for(auto ins : iterator_for(*mod))
        {
            if(ins->name() != "@param")
                continue;

            auto pos = std::next(ins);
            auto a   = insert_allocation(pos, ins->get_shape());
            auto c   = mod->insert_instruction(pos, make_op("hip::copy_to_gpu"), ins, a);
            mod->replace_instruction(ins, c);
        }

        // return instruction
        auto ret = std::prev(mod->end());
        if(ret->name() == "@return")
        {
            const auto& inputs = ret->inputs();

            // each input of ret need to be copied from gpu to host, and replace
            // output with copy output
            for(const auto& in : inputs)
            {
                auto p_output = mod->insert_instruction(ret, make_op("hip::copy_from_gpu"), in);
                instruction::replace_argument(ret, in, p_output);
            }
        }
        // else branch to handle legacy program without the return instruction
        else
        {
            mod->add_instruction(make_op("hip::copy_from_gpu"), ret);
        }
    }

    void apply()
    {
        init();
        for(auto it = mod->begin(); it != mod->end(); it++)
        {
            auto s = it->get_shape();
            if(apply_map.count(it->name()) > 0)
            {
                check_shape(s, apply_map.at(it->name())(it));
            }
        }

        copy_params();
    }

    instruction_ref insert_allocation(instruction_ref ins, const shape& s, std::string tag = "")
    {
        // Instruction's output is an input of the ret instruction
        if(offload_copy)
        {
            auto result = mod->insert_instruction(
                ins, make_op("hip::allocate", {{"shape", to_value(s)}, {"tag", std::move(tag)}}));
            return result;
        }

        auto ins_alias = instruction::get_output_alias(ins);
        if(last->name() == "@return" and tag.empty() and prog_output_names.count(ins_alias) > 0)
        {
            return mod->add_parameter(prog_output_names[ins_alias], s);
        }
        else if(ins == last and tag.empty())
        {
            return mod->add_parameter("output", s);
        }

        return mod->insert_instruction(
            ins, make_op("hip::allocate", {{"shape", to_value(s)}, {"tag", std::move(tag)}}));
    }

    void add_convolution_op()
    {
        apply_map.emplace("convolution", [=](instruction_ref ins) {
            auto&& op = any_cast<op::convolution>(ins->get_operator());

            auto conv = miopen_convolution{op, make_conv(op)};
            auto ws   = conv.find(get_context(), ins->get_shape(), to_shapes(ins->inputs()));

            auto workspace = insert_allocation(ins, ws, "workspace");
            auto output    = insert_allocation(ins, ins->get_shape());

            return mod->replace_instruction(
                ins, conv, ins->inputs().at(0), ins->inputs().at(1), workspace, output);
        });
    }

    void add_deconvolution_op()
    {
        apply_map.emplace("deconvolution", [=](instruction_ref ins) {
            auto&& op = any_cast<op::deconvolution>(ins->get_operator());

            auto conv = miopen_deconvolution{op, make_deconv(op)};
            auto ws   = conv.compile(get_context(), ins->get_shape(), to_shapes(ins->inputs()));

            auto workspace = insert_allocation(ins, ws, "workspace");
            auto output    = insert_allocation(ins, ins->get_shape());

            return mod->replace_instruction(
                ins, conv, ins->inputs().at(0), ins->inputs().at(1), workspace, output);
        });
    }

    template <class Op>
    void add_gemm_op(std::string name)
    {
        apply_map.emplace(name, [=](instruction_ref ins) {
            auto&& op                         = any_cast<Op>(ins->get_operator());
            auto beta                         = op.beta;
            std::vector<instruction_ref> refs = ins->inputs();
            if(refs.size() == 2)
            {
                auto output = insert_allocation(ins, ins->get_shape());
                beta        = 0;
                refs.push_back(output);
            }
            else
            {
                auto c_alias = instruction::get_output_alias(refs.back());
                if(ins == last or refs.back()->outputs().size() > 1 or c_alias->inputs().empty())
                {
                    auto output = insert_allocation(ins, ins->get_shape());
                    auto copy_out =
                        mod->insert_instruction(ins, make_op("hip::copy"), refs.back(), output);
                    refs.back() = copy_out;
                    refs.push_back(copy_out);
                }
                else
                {
                    refs.push_back(refs.back());
                }
            }

            return mod->replace_instruction(
                ins, rocblas_gemm<Op>{Op{op.alpha, beta}, int8_x4_format}, refs);
        });
    }

    void add_quant_convolution_op()
    {
        apply_map.emplace("quant_convolution", [=](instruction_ref ins) {
            auto&& op = any_cast<op::quant_convolution>(ins->get_operator());
            auto conv = miopen_quant_convolution{op, make_conv(op)};
            auto ws   = conv.compile(get_context(), ins->get_shape(), to_shapes(ins->inputs()));

            auto args      = ins->inputs();
            auto workspace = insert_allocation(ins, ws, "workspace");
            auto output    = insert_allocation(ins, ins->get_shape());

            return mod->replace_instruction(ins, conv, args[0], args[1], workspace, output);
        });
    }

    void add_generic_op(const std::string& name) { add_generic_op(name, "gpu::" + name); }

    void add_generic_op(const std::string& op_name, const std::string& gpu_name)
    {
        apply_map.emplace(op_name, [=](instruction_ref ins) {
            auto output                       = insert_allocation(ins, ins->get_shape());
            std::vector<instruction_ref> refs = ins->inputs();
            refs.push_back(output);

            return mod->replace_instruction(ins, make_op(gpu_name), refs);
        });
    }

    void add_extend_op(const std::string& name) { add_extend_op(name, "gpu::" + name); }

    void add_extend_op(const std::string& op_name, const std::string& gpu_name)
    {
        apply_map.emplace(op_name, [=](instruction_ref ins) {
            auto&& op                         = ins->get_operator();
            auto output                       = insert_allocation(ins, ins->get_shape());
            std::vector<instruction_ref> refs = ins->inputs();
            refs.push_back(output);

            return mod->replace_instruction(ins, make_op(gpu_name, op.to_value()), refs);
        });
    }

    void add_batch_norm_inference_op()
    {
        apply_map.emplace("batch_norm_inference", [=](instruction_ref ins) {
            auto&& op       = any_cast<op::batch_norm_inference>(ins->get_operator());
            auto output     = insert_allocation(ins, ins->get_shape());
            shape old_shape = ins->inputs().at(1)->get_shape();
            auto input      = ins->inputs()[0];
            auto input_lens = input->get_shape().lens();
            std::vector<int64_t> rsp_lens(input_lens.size(), 1);
            // for per_activation case, also need to reshape input
            if(op.bn_mode == op::batch_norm_inference::per_activation)
            {
                std::copy(input_lens.begin() + 1, input_lens.end(), rsp_lens.begin() + 1);
            }
            else
            {
                rsp_lens[1] = static_cast<int64_t>(old_shape.elements());
            }

            auto reshape_op = op::reshape{rsp_lens};
            std::vector<instruction_ref> reshapes;
            std::transform(ins->inputs().begin() + 1,
                           ins->inputs().end(),
                           std::back_inserter(reshapes),
                           [&](auto i) { return mod->insert_instruction(ins, reshape_op, i); });

            return mod->replace_instruction(ins,
                                            miopen_batch_norm_inference{op},
                                            input,
                                            reshapes[0],
                                            reshapes[1],
                                            reshapes[2],
                                            reshapes[3],
                                            output);

        });
    }

    // use 0 - input to represent neg
    void add_neg_op()
    {
        apply_map.emplace("neg", [=](instruction_ref ins) {
            auto s = ins->get_shape();
            std::vector<float> zeros(s.elements(), 0.0f);
            auto l0     = mod->add_literal(literal(s, zeros));
            auto output = insert_allocation(ins, s);
            return mod->replace_instruction(
                ins, make_op("gpu::sub"), l0, ins->inputs().front(), output);
        });
    }

    // replace the if operator with gpu_if operator
    void add_if_op()
    {
        apply_map.emplace("if", [=](instruction_ref ins) {
            std::vector<instruction_ref> inputs = ins->inputs();
            auto cpu_cond =
                mod->insert_instruction(ins, make_op("hip::copy_from_gpu"), inputs.front());
            auto sync_cond = mod->insert_instruction(ins, make_op("hip::sync_stream"), cpu_cond);
            inputs.front() = sync_cond;

            std::vector<module_ref> mod_args = ins->module_inputs();
            std::map<std::string, shape> name_shapes;
            for(const auto& smod : mod_args)
            {
                auto ps = smod->get_parameter_shapes();
                name_shapes.insert(ps.begin(), ps.end());
            }

            bool ins_output_allocated = false;
            for(auto& pn : name_shapes)
            {
                const auto& s = pn.second;
                instruction_ref output{};
                if(s == ins->get_shape() and not ins_output_allocated)
                {
                    output               = insert_allocation(ins, s);
                    ins_output_allocated = true;
                }
                else
                {
                    output = mod->insert_instruction(
                        ins, make_op("hip::allocate", {{"shape", to_value(s)}}));
                }
                inputs.push_back(output);
            }

            return mod->replace_instruction(ins, ins->get_operator(), inputs, mod_args);
        });
    }
};

void lowering::apply(module& m) const { miopen_apply{&m, this}.apply(); }
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
