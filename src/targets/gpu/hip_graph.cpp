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
#include <migraphx/gpu/kernel.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/module.hpp>
#include <migraphx/module_ref.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/output_iterator.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/transform_view.hpp>
#include <migraphx/register_op.hpp>
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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

// The data pointer of each leaf argument (the buffers the kernels read/write).
static std::vector<const void*> leaf_ptrs(const std::vector<argument>& leaves)
{
    std::vector<const void*> ptrs;
    std::transform(leaves.begin(), leaves.end(), std::back_inserter(ptrs), [](const argument& a) {
        return a.data();
    });
    return ptrs;
}

// True when `consumer` passes the buffer of its input `input` straight through
// (a view op such as reshape/slice/load), so the parameter's address continues
// to flow to the consumer's own outputs.
static bool aliases_input(instruction_ref consumer, instruction_ref input)
{
    return contains(instruction::get_output_alias(consumer, true), input);
}

// Bind the submodule's parameters (in order) to the op's input arguments.
static std::unordered_map<std::string, argument> create_params(const_module_ref sub,
                                                               const std::vector<argument>& args)
{
    auto param_names = sub->get_parameter_names();
    assert(param_names.size() == args.size());
    std::unordered_map<std::string, argument> params;
    for_each(param_names.begin(),
             param_names.end(),
             args.begin(),
             [&](const std::string& pname, const argument& a) { params[pname] = a; });
    return params;
}

// Thin RAII wrapper over the HIP graph C API: capture a stream into a graph,
// enumerate/patch its nodes, instantiate an executable graph, and launch/update
// it. Graph and exec handles are shared so value-semantic copies share one graph.
struct hip_graph
{
    template <class F>
    static hip_graph capture(hipStream_t stream, F f)
    {
        check_hip(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
                  "hipStreamBeginCapture");
        f();
        hipGraph_t g = nullptr;
        check_hip(hipStreamEndCapture(stream, &g), "hipStreamEndCapture");
        return hip_graph{share(hip_graph_ptr{g})};
    }

    struct node
    {
        hipGraphNodeType type() const
        {
            hipGraphNodeType t{};
            check_hip(hipGraphNodeGetType(ptr, &t), "hipGraphNodeGetType");
            return t;
        }
        hipKernelNodeParams get_kernel_node_params() const
        {
            hipKernelNodeParams params{};
            check_hip(hipGraphKernelNodeGetParams(ptr, &params), "hipGraphKernelNodeGetParams");
            return params;
        }
        void set_kernel_node_params(hipKernelNodeParams params) const
        {
            check_hip(hipGraphKernelNodeSetParams(ptr, &params), "hipGraphKernelNodeSetParams");
        }
        hipGraphNode_t ptr = nullptr;
    };

    std::vector<node> nodes() const
    {
        std::size_t num = 0;
        check_hip(hipGraphGetNodes(ptr.get(), nullptr, &num), "hipGraphGetNodes");
        std::vector<hipGraphNode_t> handles(num);
        if(num > 0)
            check_hip(hipGraphGetNodes(ptr.get(), handles.data(), &num), "hipGraphGetNodes");
        std::vector<node> result(num);
        std::transform(handles.begin(), handles.end(), result.begin(), [](hipGraphNode_t h) {
            return node{h};
        });
        return result;
    }

    struct exec
    {
        shared<hip_graph_exec_ptr> ptr = nullptr;
        void launch(hipStream_t stream) const
        {
            check_hip(hipGraphLaunch(ptr.get(), stream), "hipGraphLaunch");
        }
        // Re-sync the executable graph after `g`'s nodes changed; returns false if
        // the topology cannot be updated in place and the exec must be rebuilt.
        bool update(hip_graph& g)
        {
            hipGraphNode_t error_node       = nullptr;
            hipGraphExecUpdateResult result = hipGraphExecUpdateError;
            auto status = hipGraphExecUpdate(ptr.get(), g.ptr.get(), &error_node, &result);
            return status == hipSuccess and result == hipGraphExecUpdateSuccess;
        }
    };

    exec instantiate() const
    {
        hipGraphExec_t e = nullptr;
        check_hip(hipGraphInstantiate(&e, ptr.get(), nullptr, nullptr, 0), "hipGraphInstantiate");
        return exec{share(hip_graph_exec_ptr{e})};
    }

    shared<hip_graph_ptr> ptr = nullptr;
};

// One word in a kernel node's packed argument buffer that holds a graph-input
// pointer: its byte offset in the buffer, which flattened graph-input leaf it
// points into, and the offset within that leaf (nonzero for a viewed/sliced
// input). A rebind rewrites exactly these words.
struct graph_slot_patch
{
    std::size_t offset;
    std::size_t leaf;
    std::size_t ptr_offset;
};

// A captured kernel node and the buffer slots holding a movable-parameter pointer
// to rewrite when that parameter moves.
struct graph_node_patch
{
    hip_graph::node node{};
    std::vector<graph_slot_patch> slots{};

    // Record a slot at byte offset `off` when the pointer `p` falls inside one of
    // the movable input leaves, viewing each leaf as its [begin, end) byte range.
    void record_slot(std::size_t off, const char* p, const std::vector<argument>& leaves)
    {
        auto bounds = views::transform(leaves, [](const argument& leaf) {
            auto base = reinterpret_cast<std::uintptr_t>(leaf.data());
            return std::make_pair(base, base + leaf.get_shape().bytes());
        });
        auto p_addr = reinterpret_cast<std::uintptr_t>(p);
        auto it     = std::find_if(bounds.begin(), bounds.end(), [&](const auto& bound) {
            return p_addr >= bound.first and p_addr < bound.second;
        });
        if(it == bounds.end())
            return;
        slots.push_back({off,
                         static_cast<std::size_t>(std::distance(bounds.begin(), it)),
                         static_cast<std::size_t>(p_addr - (*it).first)});
    };

    // Records the work of a submodule into a HIP graph the first time it is run and
    // then replays the instantiated graph on every subsequent run. This amortizes
    // the per-launch CPU overhead of issuing many kernels/library calls. Construct
    // via make_op("hip::graph").
    struct hip_graph_op
    {
        struct graph_state
        {
            hip_graph graph{};
            hip_graph::exec exec{};
            bool captured = false;
            std::vector<argument> outputs{};
            // The packed return value (single output or a tuple), cached so the replay
            // path does not rebuild it each eval. Refreshed whenever `outputs` is.
            argument result{};
            // Addresses of the movable (program-parameter) leaves currently bound in
            // the captured graph, used to detect when a parameter buffer has moved.
            // Empty when the op has no movable inputs, so nothing is ever re-bound.
            std::vector<const void*> applied_ptrs{};
            // True when every movable parameter is consumed only by code-object
            // kernels we can patch; `patches` then lists, per such node, the slots to
            // rewrite. False (a parameter reaches a library kernel) -> re-record.
            bool patchable = false;
            std::vector<graph_node_patch> patches{};

            // Capture `f`'s device work into a fresh graph and cache the packed result.
            // `f` returns the submodule outputs; capturing only records the launches,
            // so the buffers are filled when the instantiated graph is later launched.
            template <class F>
            void record(hipStream_t stream, F f)
            {
                graph  = hip_graph::capture(stream, [&] { outputs = f(); });
                result = pack_outputs(outputs);
            }
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

        // Indices of the inputs whose buffer the caller can rebind between runs (the
        // program parameters). Every other input is a fixed allocation/constant.
        // hipgraphify fills this in; when it is empty the captured graph is bound to
        // stable addresses and is replayed without ever inspecting its nodes.
        std::vector<std::size_t> replace_inputs{};

        template <class Self, class F>
        static auto reflect(Self& self, F f)
        {
            // The captured graph is runtime-only state and is excluded.
            return pack(f(self.aliases, "aliases"), f(self.replace_inputs, "replace_inputs"));
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

        // The movable (program-parameter) inputs flattened to their leaf arguments, in
        // replace_inputs order. A tuple has no single data pointer, so the rebind logic
        // tracks its leaves; these are the only addresses that can change between runs.
        std::vector<argument> movable_leaves(const std::vector<argument>& args) const
        {
            std::vector<argument> leaves;
            std::transform(replace_inputs.begin(),
                           replace_inputs.end(),
                           join_back_inserter(leaves),
                           [&](std::size_t idx) { return flatten({args[idx]}); });
            return leaves;
        }

        // Walk forward from a movable parameter through alias (view) ops, adding each
        // code-object kernel that consumes it -- keyed by function handle -- to `out`.
        // Returns false if the parameter reaches a non-code-object kernel consumer (a
        // library gemm/conv), whose argument buffer we cannot interpret and so must
        // re-record rather than patch.
        static bool collect_param_code_objects(
            instruction_ref param,
            std::unordered_map<void*, const std::map<std::size_t, kernel_argument_value>*>& out)
        {
            bool patchable = true;
            std::unordered_set<instruction_ref> visited;
            fix([&](auto self, instruction_ref ins) {
                if(not visited.insert(ins).second)
                    return;
                for(auto consumer : ins->outputs())
                {
                    if(consumer->name() == "gpu::code_object")
                    {
                        const auto& cop = any_cast<code_object_op>(consumer->get_operator());
                        out.emplace(cop.k.get_function(), &cop.kernel_args);
                    }
                    else if(aliases_input(consumer, ins))
                    {
                        // A view passes the buffer through; keep following it.
                        self(consumer);
                    }
                    else if(not starts_with(consumer->name(), "@"))
                    {
                        // A non-code-object, non-view consumer (a library kernel) we
                        // cannot interpret; a builtin (e.g. @return) is skipped.
                        patchable = false;
                    }
                }
            })(param);
            return patchable;
        }

        // Build the per-node patch plan. Only the code-object kernels that actually
        // consume a movable parameter are considered (found by walking forward from
        // each parameter); a node is matched to one by function handle and its
        // kernel_args layout used to inspect only its pointer slots, skipping the
        // inlined scalars (so a scalar can never be mistaken for a moved pointer).
        // Nodes that consume no movable parameter are left alone. Returns false (re-
        // bind by re-recording) when a parameter is consumed by a kernel we cannot
        // patch -- a library gemm/conv (see collect_param_code_objects).
        bool build_patch_plan(const std::vector<argument>& args, const_module_ref sub) const
        {
            auto leaves = movable_leaves(args);

            // The code-object kernels consuming a movable parameter, keyed by their
            // function handle so a captured node can be matched back to one.
            std::unordered_map<void*, const std::map<std::size_t, kernel_argument_value>*>
                code_object_args;
            auto param_names = sub->get_parameter_names();
            for(auto idx : replace_inputs)
            {
                if(not collect_param_code_objects(sub->get_parameter(param_names.at(idx)),
                                                  code_object_args))
                    return false;
            }

            std::vector<graph_node_patch> patches;
            for(const auto& node : state->graph.nodes())
            {
                // Only kernel nodes carry packed arguments; others hold no parameter.
                if(node.type() != hipGraphNodeTypeKernel)
                    continue;
                auto params = node.get_kernel_node_params();
                auto cobj   = code_object_args.find(params.func);
                if(cobj == code_object_args.end())
                    continue; // does not consume a movable parameter

                graph_node_patch np;
                np.node = node;
                for(const auto& [off, p] : unpack_kernel_config(params.extra, *cobj->second))
                    np.record_slot(off, p, leaves);
                if(not np.slots.empty())
                    patches.push_back(std::move(np));
            }

            state->patches = std::move(patches);
            return true;
        }

        // Apply the prebuilt plan to the captured graph: for each recorded node, copy
        // its argument buffer, overwrite only the movable-parameter slots with the
        // current parameter address (plus the captured within-buffer offset), and
        // write it back to the node. All other words are left untouched. The caller
        // re-syncs the executable graph afterwards.
        void patch_kernel_nodes(const std::vector<const void*>& current_ptrs) const
        {
            for(const auto& np : state->patches)
            {
                auto params = np.node.get_kernel_node_params();
                // build_patch_plan already verified every node parses.
                auto buf = unpack_kernel_config(params.extra);
                assert(not buf.empty());
                for(const auto& slot : np.slots)
                {
                    assert(slot.leaf < current_ptrs.size());
                    assert(slot.offset + sizeof(char*) <= buf.size());
                    const char* p =
                        static_cast<const char*>(current_ptrs[slot.leaf]) + slot.ptr_offset;
                    const auto* bytes = reinterpret_cast<const char*>(&p);
                    std::copy(bytes, bytes + sizeof(char*), buf.data() + slot.offset);
                }
                std::size_t bytes   = buf.size();
                auto extra          = pack_kernel_config(buf.data(), &bytes);
                params.extra        = extra.data();
                params.kernelParams = nullptr;
                np.node.set_kernel_node_params(params);
            }
        }

        argument
        compute(context& ctx,
                const shape&,
                const std::vector<argument>& args,
                const std::vector<module_ref>& mods,
                const std::function<std::vector<argument>(
                    module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
        {
            assert(mods.size() == 1);
            module_ref sub = mods.front();

            hipStream_t stream = ctx.get_stream().get();
            // The legacy/null stream cannot be captured; fall back to a normal run.
            if(stream == nullptr)
                return pack_outputs(run(sub, create_params(sub, args)));

            // Capturing only records the kernel launches into the graph (it does not
            // execute them); the returned arguments are views into stable buffers that
            // get filled when the instantiated graph is launched below.
            if(not state->captured)
            {
                state->record(stream, [&] { return run(sub, create_params(sub, args)); });
                state->exec = state->graph.instantiate();
                // Only inspect the captured nodes when a parameter can actually move;
                // with no movable inputs the graph stays bound to stable buffers.
                if(not replace_inputs.empty())
                {
                    state->patchable    = build_patch_plan(args, sub);
                    state->applied_ptrs = leaf_ptrs(movable_leaves(args));
                }
                state->captured = true;
            }
            else if(not replace_inputs.empty())
            {
                // Re-bind only when a parameter buffer has moved since the last run.
                auto current_ptrs = leaf_ptrs(movable_leaves(args));
                if(current_ptrs != state->applied_ptrs)
                {
                    // Re-bind the moved parameter by patching the captured graph's
                    // kernel nodes in place; for a non-patchable graph re-record it.
                    // Either way re-sync the executable graph, re-instantiating if it
                    // cannot be updated in place.
                    if(state->patchable)
                        patch_kernel_nodes(current_ptrs);
                    else
                        state->record(stream, [&] { return run(sub, create_params(sub, args)); });
                    if(not state->exec.update(state->graph))
                        state->exec = state->graph.instantiate();
                    state->applied_ptrs = std::move(current_ptrs);
                }
            }

            state->exec.launch(stream);
            return state->result;
        }
    };

    MIGRAPHX_REGISTER_OP(hip_graph_op)

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
