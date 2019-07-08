#include <rocblas.h>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/gpu/device/contiguous.hpp>
#include <migraphx/gpu/device/add.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/gpu/argmax.hpp>
#include <migraphx/gpu/argmin.hpp>
#include <migraphx/gpu/rocblas.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/convolution.hpp>
#include <migraphx/gpu/contiguous.hpp>
#include <migraphx/gpu/relu.hpp>
#include <migraphx/gpu/sigmoid.hpp>
#include <migraphx/gpu/abs.hpp>
#include <migraphx/gpu/leaky_relu.hpp>
#include <migraphx/gpu/elu.hpp>
#include <migraphx/gpu/softmax.hpp>
#include <migraphx/gpu/logsoftmax.hpp>
#include <migraphx/gpu/add.hpp>
#include <migraphx/gpu/sub.hpp>
#include <migraphx/gpu/exp.hpp>
#include <migraphx/gpu/log.hpp>
#include <migraphx/gpu/sin.hpp>
#include <migraphx/gpu/cos.hpp>
#include <migraphx/gpu/tan.hpp>
#include <migraphx/gpu/sinh.hpp>
#include <migraphx/gpu/cosh.hpp>
#include <migraphx/gpu/tanh.hpp>
#include <migraphx/gpu/asin.hpp>
#include <migraphx/gpu/acos.hpp>
#include <migraphx/gpu/atan.hpp>
#include <migraphx/gpu/mul.hpp>
#include <migraphx/gpu/max.hpp>
#include <migraphx/gpu/min.hpp>
#include <migraphx/gpu/batchnorm.hpp>
#include <migraphx/gpu/pooling.hpp>
#include <migraphx/gpu/gemm.hpp>
#include <migraphx/gpu/concat.hpp>
#include <migraphx/gpu/pad.hpp>
#include <migraphx/gpu/gather.hpp>
#include <migraphx/gpu/lrn.hpp>
#include <migraphx/gpu/convert.hpp>
#include <migraphx/gpu/clip.hpp>
#include <migraphx/gpu/reduce_sum.hpp>
#include <migraphx/gpu/pow.hpp>
#include <migraphx/gpu/reduce_mean.hpp>
#include <utility>
#include <functional>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct miopen_apply
{
    program* prog = nullptr;
    context ctx{};
    std::unordered_map<std::string, std::function<instruction_ref(instruction_ref)>> apply_map{};
    instruction_ref last{};

    void check_shape(shape x, instruction_ref i)
    {
        assert(x == i->get_shape());
        (void)x;
        (void)i;
    }

    void init()
    {
        this->last = instruction::get_output_alias(std::prev(prog->end()));
        add_miopen_simple_op<miopen_relu>("relu", make_relu);
        add_miopen_simple_op<miopen_sigmoid>("sigmoid", make_sigmoid);
        add_miopen_simple_op<miopen_abs>("abs", make_abs);
        add_miopen_simple_op<miopen_tanh>("tanh", make_tanh);

        add_miopen_extend_op<miopen_leaky_relu, op::leaky_relu>("leaky_relu", make_leaky_relu);
        add_miopen_extend_op<miopen_elu, op::elu>("elu", make_elu);

        add_generic_op<hip_add>("add");
        add_generic_op<hip_sub>("sub");
        add_generic_op<hip_exp>("exp");
        add_generic_op<hip_log>("log");
        add_generic_op<hip_sin>("sin");
        add_generic_op<hip_cos>("cos");
        add_generic_op<hip_tan>("tan");
        add_generic_op<hip_sinh>("sinh");
        add_generic_op<hip_cosh>("cosh");
        add_generic_op<hip_asin>("asin");
        add_generic_op<hip_acos>("acos");
        add_generic_op<hip_atan>("atan");
        add_generic_op<hip_mul>("mul");
        add_generic_op<hip_max>("max");
        add_generic_op<hip_min>("min");
        add_generic_op<hip_pow>("pow");

        add_extend_op<miopen_gemm, op::dot>("dot");
        add_extend_op<miopen_contiguous, op::contiguous>("contiguous");
        add_extend_op<hip_concat, op::concat>("concat");
        add_extend_op<hip_softmax, op::softmax>("softmax");
        add_extend_op<hip_logsoftmax, op::logsoftmax>("logsoftmax");
        add_extend_op<hip_argmax, op::argmax>("argmax");
        add_extend_op<hip_argmin, op::argmin>("argmin");
        add_extend_op<hip_gather, op::gather>("gather");
        add_extend_op<hip_pad, op::pad>("pad");
        add_extend_op<hip_convert, op::convert>("convert");
        add_extend_op<hip_clip, op::clip>("clip");
        add_extend_op<hip_reduce_sum, op::reduce_sum>("reduce_sum");
        add_extend_op<hip_reduce_mean, op::reduce_mean>("reduce_mean");

        add_lrn_op();
        add_convolution_op();
        add_pooling_op();
        add_batch_norm_inference_op();
    }

    void apply()
    {
        init();
        for(auto it = prog->begin(); it != prog->end(); it++)
        {
            auto s = it->get_shape();
            if(apply_map.count(it->name()) > 0)
            {
                check_shape(s, apply_map.at(it->name())(it));
            }
        }
    }

    instruction_ref insert_allocation(instruction_ref ins, const shape& s, std::string tag = "")
    {
        if(ins == last and tag.empty())
        {
            return prog->add_parameter("output", s);
        }
        else
        {
            auto result = prog->insert_instruction(ins, hip_allocate{s, std::move(tag)});
            return result;
        }
    }

    void add_convolution_op()
    {
        apply_map.emplace("convolution", [=](instruction_ref ins) {
            auto&& op = any_cast<op::convolution>(ins->get_operator());

            auto conv = miopen_convolution{op, make_conv(op)};
            auto ws   = conv.compile(ctx, ins->get_shape(), to_shapes(ins->inputs()));

            auto workspace = insert_allocation(ins, ws, "workspace");
            auto output    = insert_allocation(ins, ins->get_shape());

            return prog->replace_instruction(
                ins, conv, ins->inputs().at(0), ins->inputs().at(1), workspace, output);
        });
    }

    void add_pooling_op()
    {
        apply_map.emplace("pooling", [=](instruction_ref ins) {
            auto&& op   = any_cast<op::pooling>(ins->get_operator());
            auto pd     = make_pooling(op);
            auto output = insert_allocation(ins, ins->get_shape());

            return prog->replace_instruction(
                ins, miopen_pooling{op, std::move(pd)}, ins->inputs().at(0), output);
        });
    }

    void add_lrn_op()
    {
        apply_map.emplace("lrn", [=](instruction_ref ins) {
            auto&& op   = any_cast<op::lrn>(ins->get_operator());
            auto ldesc  = make_lrn(op);
            auto output = insert_allocation(ins, ins->get_shape());
            return prog->replace_instruction(
                ins, miopen_lrn{std::move(ldesc)}, ins->inputs().at(0), output);
        });
    }

    template <class T>
    void add_generic_op(std::string name)
    {
        apply_map.emplace(name, [=](instruction_ref ins) {
            auto output                       = insert_allocation(ins, ins->get_shape());
            std::vector<instruction_ref> refs = ins->inputs();
            refs.push_back(output);

            return prog->replace_instruction(ins, T{}, refs);
        });
    }

    template <class T, class Op>
    void add_extend_op(std::string name)
    {
        apply_map.emplace(name, [=](instruction_ref ins) {
            auto&& op                         = any_cast<Op>(ins->get_operator());
            auto output                       = insert_allocation(ins, ins->get_shape());
            std::vector<instruction_ref> refs = ins->inputs();
            refs.push_back(output);

            return prog->replace_instruction(ins, T{op}, refs);
        });
    }

    template <class T, class Op, class F>
    void add_miopen_extend_op(std::string name, F f)
    {
        apply_map.emplace(name, [=](instruction_ref ins) {
            auto&& op = any_cast<Op>(ins->get_operator());
            auto ad   = f(op.alpha);

            auto output = insert_allocation(ins, ins->get_shape());
            return prog->replace_instruction(ins, T{std::move(ad)}, ins->inputs().at(0), output);
        });
    }

    template <class T, class F>
    void add_miopen_simple_op(std::string name, F f)
    {
        apply_map.emplace(name, [=](instruction_ref ins) {
            auto ad     = f();
            auto output = insert_allocation(ins, ins->get_shape());
            return prog->replace_instruction(ins, T{std::move(ad)}, ins->inputs().at(0), output);
        });
    }

    void add_batch_norm_inference_op()
    {
        apply_map.emplace("batch_norm_inference", [=](instruction_ref ins) {
            auto&& op       = any_cast<op::batch_norm_inference>(ins->get_operator());
            auto output     = insert_allocation(ins, ins->get_shape());
            shape old_shape = ins->inputs().at(1)->get_shape();
            std::vector<int64_t> new_shape{1, static_cast<int64_t>(old_shape.elements()), 1, 1};
            auto reshape_op = op::reshape{new_shape};
            std::vector<instruction_ref> reshapes;
            std::transform(ins->inputs().begin() + 1,
                           ins->inputs().end(),
                           std::back_inserter(reshapes),
                           [&](auto i) { return prog->insert_instruction(ins, reshape_op, i); });
            return prog->replace_instruction(ins,
                                             miopen_batch_norm_inference{op},
                                             ins->inputs().at(0),
                                             reshapes[0],
                                             reshapes[1],
                                             reshapes[2],
                                             reshapes[3],
                                             output);
        });
    }
};

void lowering::apply(program& p) const { miopen_apply{&p, ctx}.apply(); }
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
