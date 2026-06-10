#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/shape.hpp>

#include "TopKLayer_impl.hpp"

namespace nvinfer1
{

namespace
{
    // TensorRT's reduceAxes is a bitmask that must select exactly one dimension.
    // Return the index of the (lowest) set bit, i.e. the axis to reduce over.
    int reduce_axis_from_mask(uint32_t reduceAxes) noexcept
    {
        for(int i = 0; i < 32; ++i)
        {
            if(reduceAxes & (1u << i))
                return i;
        }
        return 0;
    }
}  // namespace

TopKLayer_impl::TopKLayer_impl() noexcept
    : Layer_impl{LayerType::kTOPK, nullptr}
{
    ITopKLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));
    ITopKLayer::mImpl  = this;
}

TopKLayer_impl::TopKLayer_impl(ITensor& input, TopKOperation op, int32_t k, uint32_t reduceAxes, const std::shared_ptr<migraphx::program>& program) noexcept
    : Layer_impl{LayerType::kTOPK, program}, mOp{op}, mK{k}, mReduceAxes{reduceAxes}
{
    ITopKLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));
    ITopKLayer::mImpl  = this;

    mInputs.push_back(&static_cast<Tensor_impl&>(input));
    // ITopKLayer has two outputs: output(0) = values, output(1) = indices.
    mOutputs.emplace_back(std::make_unique<Tensor_impl>());
    mOutputs.emplace_back(std::make_unique<Tensor_impl>());
}

TopKLayer_impl::~TopKLayer_impl() = default;

void TopKLayer_impl::setOperation(TopKOperation op) noexcept { mOp = op; }
TopKOperation TopKLayer_impl::getOperation() const noexcept { return mOp; }

void TopKLayer_impl::setK(int32_t k) noexcept { mK = k; }
int32_t TopKLayer_impl::getK() const noexcept { return mK; }

void TopKLayer_impl::setReduceAxes(uint32_t reduceAxes) noexcept { mReduceAxes = reduceAxes; }
uint32_t TopKLayer_impl::getReduceAxes() const noexcept { return mReduceAxes; }

void TopKLayer_impl::build() noexcept
{
    auto args = getInputArguments();

    mInstructions.clear();
    mInstructions.push_back(args[0]);

    auto* mm = getModule();

    const int64_t axis    = reduce_axis_from_mask(mReduceAxes);
    const bool largest    = (mOp == TopKOperation::kMAX);

    auto topk = mm->add_instruction(
        migraphx::make_op("topk", {{"k", mK}, {"axis", axis}, {"largest", largest}}), args[0]);
    mInstructions.push_back(topk);

    auto values = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), topk);
    mInstructions.push_back(values);

    // topk emits int64 indices; TensorRT's ITopKLayer exposes int32 indices.
    auto indices = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), topk);
    indices = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), indices);
    mInstructions.push_back(indices);

    mOutputs[0]->setInstruction(values);
    mOutputs[1]->setInstruction(indices);
}

}  // namespace nvinfer1
