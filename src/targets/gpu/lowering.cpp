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
#include <migraphx/gpu/rocblas.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/convolution.hpp>
#include <migraphx/gpu/contiguous.hpp>
#include <migraphx/gpu/relu.hpp>
#include <migraphx/gpu/sigmoid.hpp>
#include <migraphx/gpu/tanh.hpp>
#include <migraphx/gpu/abs.hpp>
#include <migraphx/gpu/leaky_relu.hpp>
#include <migraphx/gpu/elu.hpp>
#include <migraphx/gpu/softmax.hpp>
#include <migraphx/gpu/add.hpp>
#include <migraphx/gpu/sin.hpp>
#include <migraphx/gpu/mul.hpp>
#include <migraphx/gpu/batchnorm.hpp>
#include <migraphx/gpu/pooling.hpp>
#include <migraphx/gpu/gemm.hpp>
#include <migraphx/gpu/concat.hpp>
#include <utility>
#include <functional>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {

struct miopen_apply
{
    program* prog = nullptr;
    context ctx{};
    std::unordered_map<std::string, std::function<instruction_ref(miopen_apply&, instruction_ref)>>
        apply_map{};

    void check_shape(shape x, instruction_ref i)
    {
        assert(x == i->get_shape());
        (void)x;
        (void)i;
    }

    void init()
    {
        apply_map["convolution"]          = &miopen_apply::apply_convolution;
        apply_map["relu"]                 = &miopen_apply::apply_relu;
        apply_map["sigmoid"]              = &miopen_apply::apply_sigmoid;
        apply_map["tanh"]                 = &miopen_apply::apply_tanh;
        apply_map["abs"]                  = &miopen_apply::apply_abs;
        apply_map["leaky_relu"]           = &miopen_apply::apply_leaky_relu;
        apply_map["elu"]                  = &miopen_apply::apply_elu;
        apply_map["pooling"]              = &miopen_apply::apply_pooling;
        apply_map["add"]                  = &miopen_apply::apply_add;
        apply_map["sin"]                  = &miopen_apply::apply_sin;
        apply_map["mul"]                  = &miopen_apply::apply_mul;
        apply_map["dot"]                  = &miopen_apply::apply_dot;
        apply_map["contiguous"]           = &miopen_apply::apply_contiguous;
        apply_map["concat"]               = &miopen_apply::apply_concat;
        apply_map["batch_norm_inference"] = &miopen_apply::apply_batch_norm_inference;
        apply_map["softmax"]              = &miopen_apply::apply_softmax;
    }

    void apply()
    {
        init();
        for(auto it = prog->begin(); it != prog->end(); it++)
        {
            auto s = it->get_shape();
            if(apply_map.count(it->name()) > 0)
            {
                check_shape(s, apply_map.at(it->name())(*this, it));
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
        auto&& op = any_cast<op::convolution>(ins->get_operator());

        auto conv = miopen_convolution{op, make_conv(op)};
        auto ws   = conv.compile(ctx, ins->get_shape(), ins->inputs());

        auto workspace = insert_allocation(ins, ws, "workspace");
        auto output    = insert_allocation(ins, ins->get_shape());

        return prog->replace_instruction(
            ins, conv, ins->inputs().at(0), ins->inputs().at(1), workspace, output);
    }

    instruction_ref apply_pooling(instruction_ref ins)
    {
        auto&& op   = any_cast<op::pooling>(ins->get_operator());
        auto pd     = make_pooling(op);
        auto output = insert_allocation(ins, ins->get_shape());

        return prog->replace_instruction(
            ins, miopen_pooling{op, std::move(pd)}, ins->inputs().at(0), output);
    }

    instruction_ref apply_relu(instruction_ref ins)
    {
        auto ad = make_relu();

        auto output = insert_allocation(ins, ins->get_shape());
        return prog->replace_instruction(
            ins, miopen_relu{std::move(ad)}, ins->inputs().at(0), output);
    }

    instruction_ref apply_sigmoid(instruction_ref ins)
    {
        auto ad = make_sigmoid();

        auto output = insert_allocation(ins, ins->get_shape());
        return prog->replace_instruction(
            ins, miopen_sigmoid{std::move(ad)}, ins->inputs().at(0), output);
    }

    instruction_ref apply_tanh(instruction_ref ins)
    {
        auto ad = make_tanh();

        auto output = insert_allocation(ins, ins->get_shape());
        return prog->replace_instruction(
            ins, miopen_tanh{std::move(ad)}, ins->inputs().at(0), output);
    }

    instruction_ref apply_abs(instruction_ref ins)
    {
        auto ad = make_abs();

        auto output = insert_allocation(ins, ins->get_shape());
        return prog->replace_instruction(
            ins, miopen_abs{std::move(ad)}, ins->inputs().at(0), output);
    }

    instruction_ref apply_leaky_relu(instruction_ref ins)
    {
        auto&& op = any_cast<op::leaky_relu>(ins->get_operator());
        auto ad   = make_leaky_relu(op.alpha);

        auto output = insert_allocation(ins, ins->get_shape());
        return prog->replace_instruction(
            ins, miopen_leaky_relu{std::move(ad)}, ins->inputs().at(0), output);
    }

    instruction_ref apply_elu(instruction_ref ins)
    {
        auto&& op = any_cast<op::leaky_relu>(ins->get_operator());
        auto ad   = make_elu(op.alpha);

        auto output = insert_allocation(ins, ins->get_shape());
        return prog->replace_instruction(
            ins, miopen_elu{std::move(ad)}, ins->inputs().at(0), output);
    }

    instruction_ref apply_softmax(instruction_ref ins)
    {
        auto&& op   = any_cast<op::softmax>(ins->get_operator());
        auto output = insert_allocation(ins, ins->get_shape());
        return prog->replace_instruction(ins, miopen_softmax{op}, ins->inputs().at(0), output);
    }

    instruction_ref apply_add(instruction_ref ins)
    {
        auto output = insert_allocation(ins, ins->get_shape());
        return prog->replace_instruction(
            ins, hip_add{}, ins->inputs().at(0), ins->inputs().at(1), output);
    }

    instruction_ref apply_sin(instruction_ref ins)
    {
        auto output = insert_allocation(ins, ins->get_shape());
        return prog->replace_instruction(ins, hip_sin{}, ins->inputs().at(0), output);
    }

    instruction_ref apply_mul(instruction_ref ins)
    {
        auto output = insert_allocation(ins, ins->get_shape());
        return prog->replace_instruction(
            ins, hip_mul{}, ins->inputs().at(0), ins->inputs().at(1), output);
    }

    instruction_ref apply_dot(instruction_ref ins)
    {
        auto&& op   = any_cast<op::dot>(ins->get_operator());
        auto output = insert_allocation(ins, ins->get_shape());
        return prog->replace_instruction(
            ins, miopen_gemm{op}, ins->inputs().at(0), ins->inputs().at(1), output);
    }

    instruction_ref apply_contiguous(instruction_ref ins)
    {
        auto&& op   = any_cast<op::contiguous>(ins->get_operator());
        auto output = insert_allocation(ins, ins->get_shape());
        return prog->replace_instruction(ins, miopen_contiguous{op}, ins->inputs().at(0), output);
    }

    instruction_ref apply_concat(instruction_ref ins)
    {
        auto&& op                         = any_cast<op::concat>(ins->get_operator());
        auto output                       = insert_allocation(ins, ins->get_shape());
        std::vector<instruction_ref> refs = ins->inputs();
        refs.push_back(output);
        return prog->replace_instruction(ins, hip_concat{op}, refs);
    }

    instruction_ref apply_batch_norm_inference(instruction_ref ins)
    {
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
    }
};

void lowering::apply(program& p) const { miopen_apply{&p, ctx}.apply(); }
} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx
