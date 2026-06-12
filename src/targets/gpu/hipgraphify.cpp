/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/gpu/hipgraphify.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Ops that synchronize with the host and therefore cannot be recorded into a HIP
// graph. They become partition boundaries.
static bool is_unsupported(const std::string& name)
{
    static const std::unordered_set<std::string> unsupported = {
        "hip::copy_from_gpu", "hip::copy_to_gpu", "hip::sync_stream"};
    return contains(unsupported, name);
}

static bool is_capturable(instruction_ref ins)
{
    const auto& op = ins->get_operator();
    // Builtins (@param/@return/@literal/...) are not real device work.
    if(starts_with(op.name(), "@"))
        return false;
    if(is_unsupported(op.name()))
        return false;
    // A context-free op that consumes inputs but does not alias them computes a
    // fresh result on the host (e.g. a ref fallback). It would not re-execute on
    // graph replay, so it cannot be captured. Two cases are deliberately kept
    // capturable: pure view ops that alias an input (load, reshape, slice,
    // get_tuple_elem) issue no work, and input-less constant sources
    // (gpu::literal) only hand back a preallocated buffer.
    if(not ins->inputs().empty() and op.is_context_free() and
       op.output_alias(to_shapes(ins->inputs())).empty())
        return false;
    return true;
}

static bool is_allocation(instruction_ref ins) { return ins->name() == "hip::allocate"; }

static void
graphify_run(module_pass_manager& mpm, const std::vector<instruction_ref>& run, std::size_t n)
{
    module& m = mpm.get_module();
    std::unordered_set<instruction_ref> run_set(run.begin(), run.end());
    auto used_outside = [&](instruction_ref ins) {
        return std::any_of(ins->outputs().begin(), ins->outputs().end(), [&](instruction_ref out) {
            return not contains(run_set, out);
        });
    };

    // Instructions kept in the parent module instead of being captured. Any
    // source with no inputs (gpu::literal constants, hip::allocate buffers) that
    // is used outside the run stays in the parent: a constant should not become a
    // passthrough output, and an allocation used outside is a buffer for an op
    // outside the captured region, not a captured value. If also used inside it
    // becomes a graph input.
    std::unordered_set<instruction_ref> keep;
    for(auto ins : run)
    {
        if(ins->inputs().empty() and used_outside(ins))
            keep.insert(ins);
    }

    // Captured outputs: run instructions consumed outside the run. Allocations
    // are buffers rather than computed outputs and are never returned.
    std::vector<instruction_ref> outputs;
    std::copy_if(run.begin(), run.end(), std::back_inserter(outputs), [&](instruction_ref ins) {
        return used_outside(ins) and not is_allocation(ins) and not contains(keep, ins);
    });
    // A run with no external outputs is pure dead code; leave it for dce.
    if(outputs.empty())
        return;

    // A module @return cannot return a value backed by internal scratch (the
    // load that memory_coloring produces has borrow lifetime). When an output is
    // produced by a kernel that writes into an allocation, keep that allocation
    // in the parent and let the submodule write into it through a parameter
    // (global lifetime). Intermediate (non-output) allocations stay in the
    // submodule so memory_coloring can still reuse their memory. Either every
    // output is allocation-backed (lowered gpu kernels) or none are (view-like
    // ops with no scratch); a mix is left uncaptured.
    std::unordered_map<instruction_ref, instruction_ref> output_buffer;
    for(auto out : outputs)
    {
        // Follow the whole alias chain (the output may be a view such as
        // multibroadcast/get_tuple_elem) down to the buffer it ultimately writes
        // into.
        auto roots = instruction::get_output_alias(out, false);
        if(roots.size() == 1 and is_allocation(roots.front()))
        {
            output_buffer[out] = roots.front();
            if(contains(run_set, roots.front()))
                keep.insert(roots.front());
        }
    }
    if(not output_buffer.empty() and output_buffer.size() != outputs.size())
        return;

    auto* sub = mpm.create_module(m.name() + ":hipgraph" + std::to_string(n));
    std::unordered_map<instruction_ref, instruction_ref> map_ins;
    std::vector<instruction_ref> fused;
    std::copy_if(run.begin(), run.end(), std::back_inserter(fused), [&](instruction_ref ins) {
        return not contains(keep, ins);
    });
    // Preserve the exact input layouts: the captured kernels are already compiled
    // for specific (possibly non-standard) strides, so parameters must keep them
    // rather than being standardized (fuse's default).
    sub->fuse(fused, &map_ins, nullptr, [](const shape& s) { return s; });
    std::vector<instruction_ref> sub_outputs;
    std::transform(outputs.begin(),
                   outputs.end(),
                   std::back_inserter(sub_outputs),
                   [&](instruction_ref ins) { return map_ins.at(ins); });
    sub->add_return(sub_outputs);

    auto inputs = find_inputs(map_ins, &m, sub);

    // The captured outputs alias the kept output buffers, which are inputs to the
    // hip::graph op.
    std::vector<std::size_t> aliases;
    if(not output_buffer.empty())
    {
        std::transform(
            outputs.begin(), outputs.end(), std::back_inserter(aliases), [&](instruction_ref out) {
                auto buf = output_buffer.at(out);
                auto it  = std::find(inputs.begin(), inputs.end(), buf);
                assert(it != inputs.end());
                return static_cast<std::size_t>(std::distance(inputs.begin(), it));
            });
    }

    // Insert the hip::graph op right after the run. A single output is produced
    // directly; multiple outputs come back as a tuple unpacked with
    // get_tuple_elem.
    auto pos = std::next(run.back());
    auto g =
        m.insert_instruction(pos, make_op("hip::graph", {{"aliases", aliases}}), inputs, {sub});
    for(std::size_t i = 0; i < outputs.size(); ++i)
    {
        auto replacement = g;
        if(outputs.size() > 1)
            replacement = m.insert_instruction(pos, make_op("get_tuple_elem", {{"index", i}}), g);
        // Only redirect uses outside the run: a run output may also feed other
        // instructions inside the run, and those must keep using the original
        // (in-submodule) value to avoid a use-before-def in the parent module.
        auto consumers = outputs[i]->outputs();
        for(auto consumer : consumers)
        {
            if(not contains(run_set, consumer))
                instruction::replace_argument(consumer, outputs[i], replacement);
        }
    }
}

void hipgraphify::apply(module_pass_manager& mpm) const
{
    module& m = mpm.get_module();
    // Only partition the root module; loop/if bodies and fused submodules are
    // left untouched.
    if(&m != mpm.get_root_module())
        return;

    // Collect the qualifying maximal runs of capturable instructions (read-only),
    // then rewrite the module afterwards so iteration is not invalidated.
    std::vector<std::vector<instruction_ref>> runs;
    auto range = iterator_for(m);
    group_find(range.begin(), range.end(), is_capturable, [&](auto start, auto last) {
        if(static_cast<std::size_t>(std::distance(start, last)) >= min_partition_size)
            runs.emplace_back(start, last);
    });

    std::size_t n = 0;
    for(const auto& run : runs)
        graphify_run(mpm, run, n++);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
