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
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/module.hpp>
#include <migraphx/module_ref.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/register_op.hpp>
#include <hip/hip_runtime_api.h>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using hip_graph_ptr      = MIGRAPHX_MANAGE_PTR(hipGraph_t, hipGraphDestroy);
using hip_graph_exec_ptr = MIGRAPHX_MANAGE_PTR(hipGraphExec_t, hipGraphExecDestroy);

static void check_hip(hipError_t status, const char* what)
{
    if(status != hipSuccess)
        MIGRAPHX_THROW("hip::graph: " + std::string(what) + " failed: " + hip_error(status));
}

// A captured submodule with a single output yields that output directly; with
// multiple outputs it yields a tuple, matching compute_shape().
static argument pack_outputs(const std::vector<argument>& outputs)
{
    if(outputs.size() == 1)
        return outputs.front();
    return argument(outputs);
}

// Records the work of a submodule into a HIP graph the first time it is run and
// then replays the instantiated graph on every subsequent run. This amortizes
// the per-launch CPU overhead of issuing many kernels/library calls. Construct
// via make_op("hip::graph").
struct hip_graph
{
    // Runtime-only state, owned through a shared_ptr so the const compute() can
    // mutate the captured graph through the pointer.
    struct graph_state
    {
        hip_graph_ptr graph     = nullptr;
        hip_graph_exec_ptr exec = nullptr;
        bool captured           = false;
        std::vector<argument> outputs{};
    };

    // Created in finalize() rather than at construction: operation handles are
    // copy-on-write, so a construction-time state would be shared by every copy
    // of the operator made during compilation. finalize() runs once per
    // instruction (after the handle is cloned), giving each its own state.
    std::shared_ptr<graph_state> state{};

    // Indices of the inputs that the captured outputs are written into (and so
    // alias). The submodule writes each output into one of these passed-in
    // buffers, so they have global lifetime and can be returned safely.
    std::vector<std::size_t> aliases{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        // The captured graph is runtime-only state and is excluded.
        return pack(f(self.aliases, "aliases"));
    }

    std::string name() const { return "hip::graph"; }

    std::vector<std::size_t> output_alias(const std::vector<shape>&) const { return aliases; }

    shape compute_shape(std::vector<shape>, std::vector<module_ref> mods) const
    {
        if(mods.size() != 1)
            MIGRAPHX_THROW("hip::graph: expected exactly one submodule");
        auto out_shapes = mods.front()->get_output_shapes();
        if(out_shapes.size() == 1)
            return out_shapes.front();
        return shape(out_shapes);
    }

    void finalize(context&, const shape&, const std::vector<shape>&)
    {
        state = std::make_shared<graph_state>();
    }

    argument compute(context& ctx,
                     const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& mods,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        module_ref sub   = mods.front();
        auto param_names = sub->get_parameter_names();
        assert(param_names.size() == args.size());
        std::unordered_map<std::string, argument> params;
        for_each(param_names.begin(),
                 param_names.end(),
                 args.begin(),
                 [&](const std::string& pname, const argument& a) { params[pname] = a; });

        hipStream_t stream = ctx.get_stream().get();
        // The legacy/null stream cannot be captured; fall back to a normal run.
        if(stream == nullptr)
            return pack_outputs(run(sub, params));

        if(not state->captured)
        {
            check_hip(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
                      "hipStreamBeginCapture");
            // Recording does not execute the submodule: the kernels are added to
            // the graph and the returned arguments are views into stable
            // (preallocated) buffers that get filled when the instantiated graph
            // is launched below.
            state->outputs   = run(sub, params);
            hipGraph_t graph = nullptr;
            check_hip(hipStreamEndCapture(stream, &graph), "hipStreamEndCapture");
            state->graph        = hip_graph_ptr{graph};
            hipGraphExec_t exec = nullptr;
            check_hip(hipGraphInstantiate(&exec, state->graph.get(), nullptr, nullptr, 0),
                      "hipGraphInstantiate");
            state->exec     = hip_graph_exec_ptr{exec};
            state->captured = true;
        }

        check_hip(hipGraphLaunch(state->exec.get(), stream), "hipGraphLaunch");
        return pack_outputs(state->outputs);
    }
};

MIGRAPHX_REGISTER_OP(hip_graph)

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
