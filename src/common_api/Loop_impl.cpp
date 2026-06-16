#include <unordered_set>

#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>

#include "Loop_impl.hpp"
#include "NetworkDefinition_impl.hpp"
#include "Tensor_impl.hpp"

#include "layers/TripLimitLayer_impl.hpp"
#include "layers/RecurrenceLayer_impl.hpp"
#include "layers/IteratorLayer_impl.hpp"
#include "layers/LoopOutputLayer_impl.hpp"

namespace nvinfer1
{

Loop_impl::Loop_impl(const std::shared_ptr<migraphx::program>& program,
                     NvNetworkDefinition_impl* network,
                     int index)
    : mProgram{program}, mNetwork{network}, mIndex{index}, mName{"loop_" + std::to_string(index)}
{
    mImpl = this;
}

Loop_impl::~Loop_impl() = default;

// public API
IRecurrenceLayer* Loop_impl::addRecurrence(ITensor& initialValue) noexcept
{
    auto layer = std::make_unique<RecurrenceLayer_impl>(initialValue, mProgram, this);
    auto* ptr  = layer.get();
    mRecurrences.push_back(ptr);
    mOwned.push_back(std::move(layer));
    return ptr;
}

ITripLimitLayer* Loop_impl::addTripLimit(ITensor& tensor, TripLimit limit) noexcept
{
    auto layer = std::make_unique<TripLimitLayer_impl>(tensor, limit, mProgram, this);
    auto* ptr  = layer.get();
    mTripLimits.push_back(ptr);
    mOwned.push_back(std::move(layer));
    return ptr;
}

IIteratorLayer* Loop_impl::addIterator(ITensor& tensor, int32_t axis /*= 0*/, bool reverse /*= false*/) noexcept
{
    auto layer = std::make_unique<IteratorLayer_impl>(tensor, axis, reverse, mProgram, this);
    auto* ptr  = layer.get();
    mIterators.push_back(ptr);
    mOwned.push_back(std::move(layer));
    return ptr;
}

ILoopOutputLayer* Loop_impl::addLoopOutput(ITensor& tensor, LoopOutput outputKind, int32_t axis /*= 0*/) noexcept
{
    auto layer = std::make_unique<LoopOutputLayer_impl>(tensor, outputKind, axis, mProgram, this);
    auto* ptr  = layer.get();
    mLoopOutputs.push_back(ptr);
    mOwned.push_back(std::move(layer));
    return ptr;
}

void Loop_impl::setName(char const* name) noexcept
{
    if(name != nullptr)
        mName = name;
}

char const* Loop_impl::getName() const noexcept
{
    return mName.c_str();
}

void Loop_impl::assignModules() noexcept
{
    // Create the body submodule and decide which regular layers belong to it.
    // This is purely pointer bookkeeping (no instruction is dereferenced), so it
    // is safe to run up front, before any layer or loop boundary input has been
    // built. Doing it early lets the orchestrator gate body-layer builds on this
    // loop's preBuild() regardless of build order.
    if(mBody != nullptr)
        return;
    mBody = mProgram->create_module("loop_" + std::to_string(mIndex) + "_body");
    markBodyLayers();
}

void Loop_impl::preBuild() noexcept
{
    if(mPreBuilt)
        return;
    // assignModules() is normally called first by the orchestrator, but guard
    // against direct callers so mBody is always valid here.
    if(mBody == nullptr)
        assignModules();

    // Submodule parameter order must be: iteration index, keep-going condition,
    // then one parameter per loop-carried dependency (recurrence). run_loop()
    // matches parameters to inputs in this order.
    mIterParam = mBody->add_parameter("iteration_num", migraphx::shape{migraphx::shape::int64_type});
    mCondParam = mBody->add_parameter("keep_going", migraphx::shape{migraphx::shape::bool_type});

    for(std::size_t i = 0; i < mRecurrences.size(); ++i)
    {
        auto* rec       = mRecurrences[i];
        auto init_shape = rec->inputTensors().at(0)->getInstruction()->get_shape();
        auto param      = mBody->add_parameter("recurrence_" + std::to_string(i), init_shape);
        // The recurrence's output, as seen inside the body, is this parameter.
        rec->outputTensor(0)->setInstruction(param);
    }

    // Each iterator exposes gather(tensor, iteration_index, axis) inside the body.
    // Only emit the gather if the iterator output is actually consumed: an unused
    // gather would reference a main-module tensor from inside the body, which the
    // backend would later have to lift into an extra body parameter, perturbing the
    // positional parameter order that run_loop relies on.
    for(auto* it : mIterators)
    {
        if(not isOutputConsumed(it->outputTensor(0)))
            continue;
        try
        {
            auto data_ins = it->inputTensors().at(0)->getInstruction();
            auto gathered = mBody->add_instruction(
                migraphx::make_op("gather", {{"axis", it->getAxis()}}), {data_ins, mIterParam});
            it->outputTensor(0)->setInstruction(gathered);
        }
        catch(...)
        {
            // Leave the iterator output unbound; it is only an error if it is
            // actually consumed by the body or a loop output.
        }
    }

    mPreBuilt = true;
}

void Loop_impl::markBodyLayers() noexcept
{
    // A regular network layer belongs to the loop body iff it (transitively)
    // consumes a value produced inside the loop (a recurrence or iterator
    // output, or another body layer's output). Loop-invariant layers (e.g.
    // constants) stay in the main module and are captured by reference.
    std::unordered_set<const Tensor_impl*> body_values;
    for(auto* rec : mRecurrences)
        body_values.insert(rec->outputTensor(0));
    for(auto* it : mIterators)
        body_values.insert(it->outputTensor(0));

    auto& layers = mNetwork->getLayers();

    std::unordered_set<Layer_impl*> body_layers;
    for(auto& layer : layers)
    {
        bool is_body = false;
        for(auto* input : layer->inputTensors())
        {
            if(body_values.count(input) != 0)
            {
                is_body = true;
                break;
            }
        }
        if(is_body)
        {
            layer->setModule(mBody);
            body_layers.insert(layer.get());
            for(int32_t i = 0; i < layer->getNbOutputs(); ++i)
                body_values.insert(layer->outputTensor(i));
        }
    }

    // Pull loop-invariant producers (e.g. constants) that are consumed only by
    // body layers into the submodule too. This keeps the body self-contained so
    // pointwise fusion does not have to reach across module boundaries.
    auto consumed_only_by_body = [&](Layer_impl* producer) {
        bool consumed_by_body = false;
        for(auto& layer : layers)
        {
            for(auto* input : layer->inputTensors())
            {
                bool produces = false;
                for(int32_t i = 0; i < producer->getNbOutputs(); ++i)
                    if(producer->outputTensor(i) == input)
                        produces = true;
                if(not produces)
                    continue;
                if(body_layers.count(layer.get()) != 0)
                    consumed_by_body = true;
                else
                    return false; // consumed by a non-body layer -> must stay in main
            }
        }
        return consumed_by_body;
    };

    bool changed = true;
    while(changed)
    {
        changed = false;
        for(auto& layer : layers)
        {
            if(body_layers.count(layer.get()) != 0)
                continue;
            if(consumed_only_by_body(layer.get()))
            {
                layer->setModule(mBody);
                body_layers.insert(layer.get());
                for(int32_t i = 0; i < layer->getNbOutputs(); ++i)
                    body_values.insert(layer->outputTensor(i));
                changed = true;
            }
        }
    }
}

bool Loop_impl::isOutputConsumed(const Tensor_impl* tensor) const noexcept
{
    if(tensor == nullptr)
        return false;
    for(auto& layer : mNetwork->getLayers())
    {
        for(auto* input : layer->inputTensors())
            if(input == tensor)
                return true;
    }
    for(auto* lo : mLoopOutputs)
    {
        for(auto* input : lo->inputTensors())
            if(input == tensor)
                return true;
    }
    return false;
}

void Loop_impl::finalize() noexcept
{
    if(mFinalized)
        return;
    if(mBody == nullptr)
        return;
    mFinalized = true;

    auto* mm           = mProgram->get_main_module();
    const int64_t trip = readTripCount();
    const std::size_t dep_num = mRecurrences.size();

    // Body return: keep-going condition, recurrence back-edges, then scan outputs.
    std::vector<migraphx::instruction_ref> returns;
    returns.reserve(1 + dep_num + mLoopOutputs.size());
    returns.push_back(mCondParam);
    for(auto* rec : mRecurrences)
    {
        const auto& inputs = rec->inputTensors();
        if(inputs.size() > 1 and inputs.at(1) != nullptr)
            returns.push_back(inputs.at(1)->getInstruction());
        else
            returns.push_back(rec->outputTensor(0)->getInstruction());
    }

    std::vector<int64_t> scan_dirs;
    for(auto* lo : mLoopOutputs)
    {
        const auto kind = lo->getLoopOutput();
        if(kind == LoopOutput::kLAST_VALUE)
            continue;
        returns.push_back(lo->inputTensors().at(0)->getInstruction());
        scan_dirs.push_back(kind == LoopOutput::kREVERSE ? 1 : 0);
    }

    mBody->add_return(returns);

    // Loop inputs: iteration count, initial condition, recurrence initial values.
    auto iter_lit =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type}, {trip}});
    auto cond_lit =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::bool_type}, {true}});
    std::vector<migraphx::instruction_ref> loop_inputs{iter_lit, cond_lit};
    for(auto* rec : mRecurrences)
        loop_inputs.push_back(rec->inputTensors().at(0)->getInstruction());

    auto op = scan_dirs.empty()
                  ? migraphx::make_op("loop", {{"max_iterations", trip}})
                  : migraphx::make_op(
                        "loop", {{"max_iterations", trip}, {"scan_output_directions", scan_dirs}});
    auto loop_ins = mm->add_instruction(op, loop_inputs, {mBody});

    // Bind each loop output to its element of the result tuple, which is laid
    // out as [recurrence finals..., scan outputs...].
    std::size_t scan_idx = 0;
    for(auto* lo : mLoopOutputs)
    {
        int index = 0;
        if(lo->getLoopOutput() == LoopOutput::kLAST_VALUE)
        {
            index = recurrenceIndex(lo->inputTensors().at(0));
        }
        else
        {
            index = static_cast<int>(dep_num + scan_idx);
            ++scan_idx;
        }
        auto result =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", index}}), loop_ins);
        lo->outputTensor(0)->setInstruction(result);
    }
}

namespace
{
    bool input_bound(const Layer_impl* layer, std::size_t index) noexcept
    {
        const auto& inputs = layer->inputTensors();
        if(index >= inputs.size())
            return false;
        const auto* t = inputs[index];
        return t != nullptr and t->isBound();
    }
} // namespace

bool Loop_impl::preBuildReady() const noexcept
{
    // preBuild() reads the shape of each recurrence initial value, gathers from
    // each iterator's source tensor and (via the loop instruction) the trip
    // count. All of those producers must already be built.
    for(auto* rec : mRecurrences)
        if(not input_bound(rec, 0))
            return false;
    for(auto* it : mIterators)
        if(not input_bound(it, 0))
            return false;
    for(auto* tl : mTripLimits)
        if(not input_bound(tl, 0))
            return false;
    return true;
}

bool Loop_impl::finalizeReady() const noexcept
{
    // The body return wires up the recurrence back-edges and the loop-output
    // sources, so those must be bound before finalize() can emit the return.
    for(auto* rec : mRecurrences)
    {
        const auto& inputs = rec->inputTensors();
        // A recurrence with a back-edge (input 1) must have it built; otherwise
        // the body simply returns the recurrence parameter, which preBuild
        // already bound.
        if(inputs.size() > 1 and inputs.at(1) != nullptr and not inputs.at(1)->isBound())
            return false;
    }
    for(auto* lo : mLoopOutputs)
        if(not input_bound(lo, 0))
            return false;
    return true;
}

namespace
{
    // Try to fold a tensor to a single int64 at build time. Returns false when
    // the producer is not a compile-time constant (e.g. a runtime input).
    bool eval_scalar(const Tensor_impl* tensor, int64_t& out) noexcept
    {
        if(tensor == nullptr or not tensor->isBound())
            return false;
        try
        {
            auto arg = tensor->getInstruction()->eval();
            if(arg.empty())
                return false;
            arg.visit([&](auto view) { out = static_cast<int64_t>(view.front()); });
            return true;
        }
        catch(...)
        {
            return false;
        }
    }
} // namespace

int64_t Loop_impl::readTripCount() const noexcept
{
    // migraphx's loop needs a compile-time upper bound (max_iterations). The
    // trip-limit tensor is often a runtime input (e.g. char-rnn's sequenceSize),
    // which cannot be folded. TensorRT supplies the static bound separately via
    // ILoopOutputLayer::setInput(1, maxSequenceSize) on a kCONCATENATE output, so
    // fall back to that constant when the trip limit itself is not foldable.
    int64_t value = 0;
    if(not mTripLimits.empty() and eval_scalar(mTripLimits.front()->inputTensors().at(0), value))
        return value;

    for(auto* lo : mLoopOutputs)
    {
        if(lo->getLoopOutput() == LoopOutput::kLAST_VALUE)
            continue;
        const auto& inputs = lo->inputTensors();
        if(inputs.size() > 1 and eval_scalar(inputs.at(1), value))
            return value;
    }
    return 0;
}

int Loop_impl::recurrenceIndex(const Tensor_impl* tensor) const noexcept
{
    for(std::size_t i = 0; i < mRecurrences.size(); ++i)
    {
        if(mRecurrences[i]->outputTensor(0) == tensor)
            return static_cast<int>(i);
    }
    return 0;
}

} // namespace nvinfer1
