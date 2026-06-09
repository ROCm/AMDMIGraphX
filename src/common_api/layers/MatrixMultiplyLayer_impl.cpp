// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include <numeric>

#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/builder/insert.hpp>

#include "MatrixMultiplyLayer_impl.hpp"

namespace nvinfer1
{

MatrixMultiplyLayer_impl::MatrixMultiplyLayer_impl() noexcept
    : Layer_impl{LayerType::kMATRIX_MULTIPLY, nullptr}, mOps{MatrixOperation::kNONE, MatrixOperation::kNONE}
{
    IMatrixMultiplyLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));
    IMatrixMultiplyLayer::mImpl  = this;
}

MatrixMultiplyLayer_impl::MatrixMultiplyLayer_impl(ITensor& input0, MatrixOperation op0, ITensor& input1, MatrixOperation op1, const std::shared_ptr<migraphx::program>& program) noexcept
    : Layer_impl{LayerType::kMATRIX_MULTIPLY, program}, mOps{op0, op1}
{
    IMatrixMultiplyLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));
    IMatrixMultiplyLayer::mImpl  = this;

    mInputs.push_back(&static_cast<Tensor_impl&>(input0));
    mInputs.push_back(&static_cast<Tensor_impl&>(input1));
    mOutputs.emplace_back(std::make_unique<Tensor_impl>());
}

MatrixMultiplyLayer_impl::~MatrixMultiplyLayer_impl()
{
    pass_warning("TODO! implement me!", false);
}

// public API
void MatrixMultiplyLayer_impl::setOperation(int32_t index, MatrixOperation op) noexcept
{
    if(index == 0 or index == 1)
        mOps[index] = op;
}

MatrixOperation MatrixMultiplyLayer_impl::getOperation(int32_t index) const noexcept
{
    if(index == 0 or index == 1)
        return mOps[index];
    return MatrixOperation::kNONE;
}

void MatrixMultiplyLayer_impl::build() noexcept
{
    auto args = getInputArguments();
    mInstructions.clear();

    auto* mm = getModule();

    // Apply the per-input MatrixOperation, mirroring TensorRT semantics:
    //   kNONE      : use the operand as-is (last two dims are the matrix).
    //   kTRANSPOSE : swap the last two dims.
    //   kVECTOR    : the operand is a vector; the first input [..,K] is treated
    //                as [..,1,K] (row vector) and the second as [..,K,1]
    //                (column vector). The introduced unit dim is removed from
    //                the result afterwards.
    auto apply_op = [&](migraphx::instruction_ref arg, MatrixOperation op, int input_index) {
        const auto rank = static_cast<int64_t>(arg->get_shape().ndim());
        switch(op)
        {
        case MatrixOperation::kTRANSPOSE:
        {
            std::vector<int64_t> perm(rank);
            std::iota(perm.begin(), perm.end(), 0);
            if(rank >= 2)
                std::swap(perm[rank - 1], perm[rank - 2]);
            return mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), arg);
        }
        case MatrixOperation::kVECTOR:
        {
            const int64_t axis = (input_index == 0) ? rank - 1 : rank;
            return mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {axis}}}), arg);
        }
        case MatrixOperation::kNONE:
            return arg;
        }
    };

    auto a = apply_op(args[0], mOps[0], 0);
    auto b = apply_op(args[1], mOps[1], 1);

    mInstructions.push_back(a);
    mInstructions.push_back(b);

    auto result = migraphx::op::builder::add("dot", *mm, {a, b}).at(0);

    // Drop the unit dimensions introduced by kVECTOR operands.
    if(mOps[0] == MatrixOperation::kVECTOR)
    {
        const int64_t rank = static_cast<int64_t>(result->get_shape().ndim());
        result = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {rank - 2}}}), result);
    }
    if(mOps[1] == MatrixOperation::kVECTOR)
    {
        const int64_t rank = static_cast<int64_t>(result->get_shape().ndim());
        result = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {rank - 1}}}), result);
    }

    mInstructions.push_back(result);
    mOutputs[0]->setInstruction(result);
}

}   // namespace nvinfer1
