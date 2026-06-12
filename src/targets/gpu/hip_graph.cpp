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
#include <migraphx/iterator_for.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/register_op.hpp>
#include <hip/hip_runtime_api.h>
#include <cassert>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
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

// Append the leaf buffer pointer (and, if requested, its size) of each argument,
// recursing into tuples. Tuple arguments have no single data pointer, and the
// kernels consume their leaves, so the rebind logic tracks leaves.
static void
leaf_buffers(const argument& a, std::vector<const void*>& ptrs, std::vector<std::size_t>* sizes)
{
    if(a.get_shape().type() == shape::tuple_type)
    {
        for(const auto& sub : a.get_sub_objects())
            leaf_buffers(sub, ptrs, sizes);
    }
    else
    {
        ptrs.push_back(a.data());
        if(sizes != nullptr)
            sizes->push_back(a.get_shape().bytes());
    }
}

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
    hipGraphNode_t node = nullptr;
    std::vector<graph_slot_patch> slots{};
};

// Records the work of a submodule into a HIP graph the first time it is run and
// then replays the instantiated graph on every subsequent run. This amortizes
// the per-launch CPU overhead of issuing many kernels/library calls. Construct
// via make_op("hip::graph").
struct hip_graph
{
    struct graph_state
    {
        hip_graph_ptr graph     = nullptr;
        hip_graph_exec_ptr exec = nullptr;
        bool captured           = false;
        std::vector<argument> outputs{};
        // Addresses of the movable (program-parameter) leaves currently programmed
        // into `exec`, used to detect when a parameter buffer has moved. Empty when
        // the op has no movable inputs, in which case nothing is ever re-bound.
        std::vector<const void*> applied_ptrs{};
        // Reused buffer holding the current run's movable-leaf addresses so the
        // per-eval move check does not allocate.
        std::vector<const void*> scratch_ptrs{};
        // True when every captured node holding a movable-parameter pointer is an
        // extra-packed kernel node we can rewrite; `patches` then lists, per such
        // node, the slots to rewrite. False -> re-bind by re-recording.
        bool patchable = false;
        std::vector<graph_node_patch> patches{};
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

    // Flatten the movable (program-parameter) inputs to their leaf buffer
    // addresses (and, if requested, sizes), in replace_inputs order. These are the
    // only addresses that can change between runs.
    void movable_leaves(const std::vector<argument>& args,
                        std::vector<const void*>& ptrs,
                        std::vector<std::size_t>* sizes) const
    {
        for(auto idx : replace_inputs)
            leaf_buffers(args[idx], ptrs, sizes);
    }

    // Build the per-node patch plan: for every captured node, record the buffer
    // slots that hold a movable-parameter address. Each kernel node is correlated
    // back to its gpu::code_object instruction by function handle so the node's
    // kernel_args layout is known -- only the pointer slots are inspected, and the
    // inlined scalar arguments are skipped (so a scalar cannot be mistaken for a
    // moved pointer). A library kernel (rocBLAS/hipBLASLt/MIOpen) has no entry and
    // falls back to scanning every word; if it does not consume a movable
    // parameter none of its words match and it gets no slots. Returns false (re-
    // bind by re-recording) for any node that is not an extra-packed kernel node,
    // since a movable parameter hidden in a buffer we cannot rewrite (a
    // kernelParams launch or a memcpy/memset node) could not be patched.
    bool build_patch_plan(const std::vector<argument>& args, module_ref sub) const
    {
        std::size_t num = 0;
        if(hipGraphGetNodes(state->graph.get(), nullptr, &num) != hipSuccess)
            return false;
        std::vector<hipGraphNode_t> nodes(num);
        if(num > 0 and hipGraphGetNodes(state->graph.get(), nodes.data(), &num) != hipSuccess)
            return false;

        std::vector<const void*> in_ptrs;
        std::vector<std::size_t> in_sizes;
        movable_leaves(args, in_ptrs, &in_sizes);
        assert(in_ptrs.size() == in_sizes.size());

        // Map each code-object kernel's function handle to its argument layout.
        std::unordered_map<void*, const std::map<std::size_t, kernel_argument_value>*>
            code_object_args;
        for(auto ins : iterator_for(*sub))
        {
            if(ins->name() != "gpu::code_object")
                continue;
            const auto& cop = any_cast<code_object_op>(ins->get_operator());
            code_object_args.emplace(static_cast<void*>(cop.k.get_function()), &cop.kernel_args);
        }

        // Record the slot if the pointer at `off` lands inside a movable input.
        auto record_slot = [&](graph_node_patch& np, std::size_t off, char* p) {
            for(std::size_t i = 0; i < in_ptrs.size(); ++i)
            {
                const auto* base = static_cast<const char*>(in_ptrs[i]);
                if(p >= base and p < base + in_sizes[i])
                {
                    np.slots.push_back({off, i, static_cast<std::size_t>(p - base)});
                    break;
                }
            }
        };

        std::vector<graph_node_patch> patches;
        for(auto node : nodes)
        {
            hipGraphNodeType type{};
            if(hipGraphNodeGetType(node, &type) != hipSuccess)
                return false;
            // Empty nodes are capture bookkeeping; they carry no arguments.
            if(type == hipGraphNodeTypeEmpty)
                continue;
            if(type != hipGraphNodeTypeKernel)
                return false;
            hipKernelNodeParams params{};
            if(hipGraphKernelNodeGetParams(node, &params) != hipSuccess)
                return false;
            if(params.kernelParams != nullptr)
                return false;

            graph_node_patch np;
            np.node   = node;
            auto cobj = code_object_args.find(params.func);
            if(cobj != code_object_args.end())
            {
                // Known code-object kernel: only its pointer slots can move.
                for(const auto& [off, p] : unpack_kernel_config(params.extra, *cobj->second))
                    record_slot(np, off, p);
            }
            else
            {
                // Library kernel: no layout, so treat every aligned word as a
                // candidate pointer.
                auto buf = unpack_kernel_config(params.extra);
                if(buf.empty())
                    return false;
                for(std::size_t off = 0; off + sizeof(char*) <= buf.size(); off += sizeof(char*))
                {
                    char* p = nullptr;
                    std::memcpy(&p, buf.data() + off, sizeof(char*));
                    record_slot(np, off, p);
                }
            }
            if(not np.slots.empty())
                patches.push_back(std::move(np));
        }

        state->patches = std::move(patches);
        return true;
    }

    // Apply the prebuilt plan: for each recorded node, copy its captured argument
    // buffer, overwrite only the movable-parameter slots with the current
    // parameter address (plus the captured within-buffer offset), and program it
    // into the executable graph. All other words are left untouched.
    void patch_kernel_nodes(const std::vector<const void*>& current_ptrs) const
    {
        for(const auto& np : state->patches)
        {
            hipKernelNodeParams params{};
            check_hip(hipGraphKernelNodeGetParams(np.node, &params), "hipGraphKernelNodeGetParams");
            // build_patch_plan already verified every node parses.
            auto buf = unpack_kernel_config(params.extra);
            assert(not buf.empty());
            for(const auto& slot : np.slots)
            {
                assert(slot.leaf < current_ptrs.size());
                assert(slot.offset + sizeof(char*) <= buf.size());
                const char* p = static_cast<const char*>(current_ptrs[slot.leaf]) + slot.ptr_offset;
                std::memcpy(buf.data() + slot.offset, &p, sizeof(char*));
            }
            std::size_t bytes   = buf.size();
            auto extra          = pack_kernel_config(buf.data(), &bytes);
            params.extra        = extra.data();
            params.kernelParams = nullptr;
            check_hip(hipGraphExecKernelNodeSetParams(state->exec.get(), np.node, &params),
                      "hipGraphExecKernelNodeSetParams");
        }
    }

    argument compute(context& ctx,
                     const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& mods,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        assert(mods.size() == 1);
        module_ref sub = mods.front();

        // Build the parameter map and run the submodule. Only needed when
        // (re-)recording or on the null-stream fallback -- never on the
        // steady-state replay path -- so it stays off the hot path.
        auto run_submodule = [&] {
            auto param_names = sub->get_parameter_names();
            assert(param_names.size() == args.size());
            std::unordered_map<std::string, argument> params;
            for_each(param_names.begin(),
                     param_names.end(),
                     args.begin(),
                     [&](const std::string& pname, const argument& a) { params[pname] = a; });
            return run(sub, params);
        };

        hipStream_t stream = ctx.get_stream().get();
        // The legacy/null stream cannot be captured; fall back to a normal run.
        if(stream == nullptr)
            return pack_outputs(run_submodule());

        auto instantiate = [&] {
            hipGraphExec_t exec = nullptr;
            check_hip(hipGraphInstantiate(&exec, state->graph.get(), nullptr, nullptr, 0),
                      "hipGraphInstantiate");
            state->exec = hip_graph_exec_ptr{exec};
        };
        // Record the submodule's kernels into a fresh graph using the current
        // buffers. Capturing does not execute the kernels: they are added to the
        // graph and the returned arguments are views into stable buffers that get
        // filled when the instantiated graph is launched below.
        auto record = [&] {
            check_hip(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
                      "hipStreamBeginCapture");
            state->outputs   = run_submodule();
            hipGraph_t graph = nullptr;
            check_hip(hipStreamEndCapture(stream, &graph), "hipStreamEndCapture");
            state->graph = hip_graph_ptr{graph};
        };
        if(not state->captured)
        {
            record();
            instantiate();
            // Only inspect the captured nodes when a parameter can actually move;
            // with no movable inputs the graph stays bound to stable buffers.
            if(not replace_inputs.empty())
            {
                state->patchable = build_patch_plan(args, sub);
                movable_leaves(args, state->applied_ptrs, nullptr);
            }
            state->captured = true;
        }
        else if(not replace_inputs.empty())
        {
            // Re-bind only when a parameter buffer has moved since the last run.
            // The current addresses go in a reused scratch buffer to keep the
            // steady-state replay path allocation-free.
            auto& current_ptrs = state->scratch_ptrs;
            current_ptrs.clear();
            movable_leaves(args, current_ptrs, nullptr);
            if(current_ptrs != state->applied_ptrs)
            {
                // Patch the new addresses into the executable graph when every node
                // holding one is rewritable; otherwise re-record and let
                // hipGraphExecUpdate patch it (re-instantiating if the topology
                // cannot be updated).
                if(state->patchable)
                {
                    patch_kernel_nodes(current_ptrs);
                }
                else
                {
                    record();
                    hipGraphNode_t error_node       = nullptr;
                    hipGraphExecUpdateResult result = hipGraphExecUpdateError;
                    auto status                     = hipGraphExecUpdate(
                        state->exec.get(), state->graph.get(), &error_node, &result);
                    if(status != hipSuccess or result != hipGraphExecUpdateSuccess)
                        instantiate();
                }
                state->applied_ptrs = current_ptrs;
            }
        }

        check_hip(hipGraphLaunch(state->exec.get(), stream), "hipGraphLaunch");
        return pack_outputs(state->outputs);
    }
};

MIGRAPHX_REGISTER_OP(hip_graph)

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
