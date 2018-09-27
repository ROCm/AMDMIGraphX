#include <rocblas.h>
#include <migraph/gpu/lowering.hpp>
#include <migraph/manage_ptr.hpp>
#include <migraph/instruction.hpp>
#include <migraph/operators.hpp>
#include <migraph/generate.hpp>
#include <migraph/shape_for_each.hpp>
#include <migraph/gpu/miopen.hpp>
#include <migraph/gpu/hip.hpp>
#include <migraph/dfor.hpp>
#include <migraph/gpu/device/contiguous.hpp>
#include <migraph/gpu/device/add.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/gpu/rocblas.hpp>
#include <migraph/gpu/context.hpp>
#include <migraph/gpu/convolution.hpp>
#include <migraph/gpu/pooling.hpp>
#include <migraph/gpu/gemm.hpp>
#include <utility>

namespace migraph {
namespace gpu {

struct miopen_batch_norm_inference
{
    batch_norm_inference op;

    std::string name() const { return "gpu::batch_norm_inference"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(6);
        return op.compute_shape(
            {inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3), inputs.at(4)});
    }

    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        auto x_desc  = make_tensor(args[0].get_shape());
        auto y_desc  = make_tensor(output_shape);
        auto bn_desc = make_tensor(args[3].get_shape());

        float alpha = 1.0, beta = 0.0f;

        miopenBatchNormalizationForwardInference(ctx.handle.get(),
                                                 miopenBatchNormMode_t(op.bn_mode),
                                                 &alpha,
                                                 &beta,
                                                 x_desc.get(),
                                                 args[0].implicit(),
                                                 y_desc.get(),
                                                 args[5].implicit(),
                                                 bn_desc.get(),
                                                 args[1].implicit(),
                                                 args[2].implicit(),
                                                 args[3].implicit(),
                                                 args[4].implicit(),
                                                 op.epsilon);

        return args[5];
    }
};

struct hip_add
{
    std::string name() const { return "gpu::add"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        // check_shapes{inputs, *this}.has(3).standard();
        check_shapes{inputs, *this}.has(3);
        return inputs.at(0);
    }

    argument compute(context&, const shape&, const std::vector<argument>& args) const
    {
        device::add(args[2], args[0], args[1]);
        return args[2];
    }
};

struct miopen_add
{
    std::string name() const { return "gpu::add"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(3).not_broadcasted();
        return inputs.at(0);
    }

    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        float alpha = 1, beta = 0;
        auto a_desc = make_tensor(args[0].get_shape());
        auto b_desc = make_tensor(args[1].get_shape());
        auto c_desc = make_tensor(output_shape);
        miopenOpTensor(ctx.handle.get(),
                       miopenTensorOpAdd,
                       &alpha,
                       a_desc.get(),
                       args[0].implicit(),
                       &alpha,
                       b_desc.get(),
                       args[1].implicit(),
                       &beta,
                       c_desc.get(),
                       args[2].implicit());
        return args[2];
    }
};

struct miopen_contiguous
{
    contiguous op;
    std::string name() const { return "gpu::contiguous"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        return op.compute_shape({inputs.at(0)});
    }
    argument compute(context&, shape output_shape, const std::vector<argument>& args) const
    {
        assert(output_shape == args[1].get_shape());
        assert(output_shape.standard());
        (void)output_shape;
        device::contiguous(args.at(1), args.at(0));
        return args.at(1);
    }
};

struct miopen_relu
{
    shared<activation_descriptor> ad;
    std::string name() const { return "gpu::relu"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(2).not_broadcasted();
        return inputs.at(1);
    }

    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        float alpha = 1, beta = 0;
        auto x_desc = make_tensor(args[0].get_shape());
        auto y_desc = make_tensor(output_shape);
        miopenActivationForward(ctx.handle.get(),
                                ad.get(),
                                &alpha,
                                x_desc.get(),
                                args[0].implicit(),
                                &beta,
                                y_desc.get(),
                                args[1].implicit());

        return args[1];
    }
};

struct miopen_softmax
{
    softmax op;
    std::string name() const { return "gpu::softmax"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(2).standard();
        return op.compute_shape({inputs.at(0)});
    }

    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        float alpha = 1, beta = 0;
        auto x_desc = make_tensor(args[0].get_shape());
        auto y_desc = make_tensor(output_shape);
        miopenSoftmaxForward(ctx.handle.get(),
                             &alpha,
                             x_desc.get(),
                             args[0].implicit(),
                             &beta,
                             y_desc.get(),
                             args[1].implicit());

        return args[1];
    }
};

struct miopen_apply
{
    program* prog = nullptr;
    context ctx{};

    void check_shape(shape x, instruction_ref i)
    {
        assert(x == i->get_shape());
        (void)x;
        (void)i;
    }

    void apply()
    {
        for(auto it = prog->begin(); it != prog->end(); it++)
        {
            auto s = it->get_shape();
            if(it->name() == "convolution")
            {
                check_shape(s, apply_convolution(it));
            }
            else if(it->name() == "activation")
            {
                check_shape(s, apply_activation(it));
            }
            else if(it->name() == "pooling")
            {
                check_shape(s, apply_pooling(it));
            }
            else if(it->name() == "add")
            {
                check_shape(s, apply_add(it));
            }
            else if(it->name() == "gemm")
            {
                check_shape(s, apply_gemm(it));
            }
            else if(it->name() == "contiguous")
            {
                check_shape(s, apply_contiguous(it));
            }
            else if(it->name() == "batch_norm_inference")
            {
                check_shape(s, apply_batch_norm_inference(it));
            }
            else if(it->name() == "softmax")
            {
                check_shape(s, apply_softmax(it));
            }
        }
    }

    instruction_ref insert_allocation(instruction_ref ins, const shape& s, std::string tag = "")
    {
        if(ins == --prog->end() and tag.empty())
        {
            return prog->add_parameter("output", s);
        }
        else
        {
            auto is     = prog->add_outline(s);
            auto result = prog->insert_instruction(ins, hip_allocate{std::move(tag)}, is);
            return result;
        }
    }

    instruction_ref apply_convolution(instruction_ref ins)
    {
        auto&& op = any_cast<convolution>(ins->get_operator());

        auto conv = miopen_convolution{op, make_conv(op)};
        auto ws   = conv.compile(ctx, ins->get_shape(), ins->inputs());

        auto workspace = insert_allocation(ins, ws, "workspace");
        auto output    = insert_allocation(ins, ins->get_shape());

        return prog->replace_instruction(
            ins, conv, ins->inputs().at(0), ins->inputs().at(1), workspace, output);
    }

    instruction_ref apply_pooling(instruction_ref ins)
    {
        auto&& op   = any_cast<pooling>(ins->get_operator());
        auto pd     = make_pooling(op);
        auto output = insert_allocation(ins, ins->get_shape());

        return prog->replace_instruction(
            ins, miopen_pooling{op, std::move(pd)}, ins->inputs().at(0), output);
    }

    instruction_ref apply_activation(instruction_ref ins)
    {
        auto&& op = any_cast<activation>(ins->get_operator());
        auto ad   = make_relu();
        if(op.mode == "relu")
        {
            auto output = insert_allocation(ins, ins->get_shape());
            return prog->replace_instruction(
                ins, miopen_relu{std::move(ad)}, ins->inputs().at(0), output);
        }
        return ins;
    }

    instruction_ref apply_softmax(instruction_ref ins)
    {
        auto&& op   = any_cast<softmax>(ins->get_operator());
        auto output = insert_allocation(ins, ins->get_shape());
        return prog->replace_instruction(ins, miopen_softmax{op}, ins->inputs().at(0), output);
    }

    instruction_ref apply_add(instruction_ref ins)
    {
        auto output = insert_allocation(ins, ins->get_shape());
        return prog->replace_instruction(
            ins, hip_add{}, ins->inputs().at(0), ins->inputs().at(1), output);
    }

    instruction_ref apply_gemm(instruction_ref ins)
    {
        auto&& op   = any_cast<gemm>(ins->get_operator());
        auto output = insert_allocation(ins, ins->get_shape());
        return prog->replace_instruction(
            ins, miopen_gemm{op}, ins->inputs().at(0), ins->inputs().at(1), output);
    }

    instruction_ref apply_contiguous(instruction_ref ins)
    {
        auto&& op   = any_cast<contiguous>(ins->get_operator());
        auto output = insert_allocation(ins, ins->get_shape());
        return prog->replace_instruction(ins, miopen_contiguous{op}, ins->inputs().at(0), output);
    }

    instruction_ref apply_batch_norm_inference(instruction_ref ins)
    {
        auto&& op       = any_cast<batch_norm_inference>(ins->get_operator());
        auto output     = insert_allocation(ins, ins->get_shape());
        shape old_shape = ins->inputs().at(1)->get_shape();
        std::vector<int64_t> new_shape{1, static_cast<int64_t>(old_shape.elements()), 1, 1};
        auto reshape_op = reshape{new_shape};
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
    }
};

void lowering::apply(program& p) const { miopen_apply{&p, ctx}.apply(); }

} // namespace gpu

} // namespace migraph
