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

static std::vector<const void*> leaf_ptrs(const std::vector<argument>& args)
{
    std::vector<const void*> ptrs;
    for(const auto& a : args)
        leaf_buffers(a, ptrs, nullptr);
    return ptrs;
}

// MIGraphX launches its code-object kernels with hipExtModuleLaunchKernel, which
// passes the arguments as a packed buffer tagged with HIP_LAUNCH_PARAM_BUFFER_*
// sentinels (kernel.cpp). Recover the buffer and its size from a captured node's
// `extra` array; returns false for any other argument-passing scheme.
static bool parse_packed_args(void** extra, char*& buffer, std::size_t& size)
{
    if(extra == nullptr)
        return false;
    buffer        = nullptr;
    bool has_size = false;
    // The config is a sequence of (tag, value) pairs terminated by
    // HIP_LAUNCH_PARAM_END; the fixed cap guards against an unterminated array.
    constexpr std::size_t max_tokens = 16;
    for(std::size_t i = 0; i < max_tokens and extra[i] != HIP_LAUNCH_PARAM_END; i += 2)
    {
        if(extra[i] == HIP_LAUNCH_PARAM_BUFFER_POINTER)
            buffer = static_cast<char*>(extra[i + 1]);
        else if(extra[i] == HIP_LAUNCH_PARAM_BUFFER_SIZE)
        {
            size     = *static_cast<std::size_t*>(extra[i + 1]);
            has_size = true;
        }
        else
            return false;
    }
    return buffer != nullptr and has_size;
}

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
        // Leaf buffer addresses (tuples flattened) baked into `graph`, with their
        // sizes, used to remap a moved input; and the leaf addresses currently
        // programmed into `exec`, used to detect a move.
        std::vector<const void*> recorded_ptrs{};
        std::vector<std::size_t> recorded_sizes{};
        std::vector<const void*> applied_ptrs{};
        // The captured graph is entirely kernel nodes whose arguments are packed
        // buffers, so a moved input can be patched into `exec` directly instead
        // of re-recording the graph.
        bool patchable = false;
        std::vector<hipGraphNode_t> kernel_nodes{};
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

    // True when every node of the captured graph is a kernel node whose arguments
    // are an extra-packed buffer we can rewrite (how MIGraphX launches its
    // code-object kernels, see kernel.cpp). Library kernels (rocBLAS/hipBLASLt/
    // MIOpen) are typically captured the same way and so are covered too; it
    // returns false for anything that isn't an extra-packed kernel node -- a
    // non-kernel node (memcpy/memset) or a kernelParams-style launch -- which
    // falls back to re-recording. Caches the node handles for later patching.
    bool collect_kernel_nodes() const
    {
        std::size_t num = 0;
        if(hipGraphGetNodes(state->graph.get(), nullptr, &num) != hipSuccess)
            return false;
        std::vector<hipGraphNode_t> nodes(num);
        if(num > 0 and hipGraphGetNodes(state->graph.get(), nodes.data(), &num) != hipSuccess)
            return false;
        for(auto node : nodes)
        {
            hipGraphNodeType type{};
            if(hipGraphNodeGetType(node, &type) != hipSuccess or type != hipGraphNodeTypeKernel)
                return false;
            hipKernelNodeParams params{};
            if(hipGraphKernelNodeGetParams(node, &params) != hipSuccess)
                return false;
            char* buffer      = nullptr;
            std::size_t bytes = 0;
            if(params.kernelParams != nullptr or not parse_packed_args(params.extra, buffer, bytes))
                return false;
        }
        state->kernel_nodes = std::move(nodes);
        return true;
    }

    // Rewrite each kernel node's argument buffer, remapping any pointer that falls
    // within an input leaf buffer's old range to the new leaf buffer (preserving
    // offsets), and program it into the executable graph.
    void patch_kernel_nodes(const std::vector<const void*>& current_ptrs) const
    {
        assert(state->recorded_ptrs.size() == current_ptrs.size() and
               state->recorded_sizes.size() == current_ptrs.size());
        std::vector<char> patched; // reused across nodes
        for(auto node : state->kernel_nodes)
        {
            hipKernelNodeParams params{};
            check_hip(hipGraphKernelNodeGetParams(node, &params), "hipGraphKernelNodeGetParams");
            char* buffer      = nullptr;
            std::size_t bytes = 0;
            // collect_kernel_nodes already verified every cached node parses.
            bool ok = parse_packed_args(params.extra, buffer, bytes);
            assert(ok and buffer != nullptr);
            (void)ok;
            patched.assign(buffer, buffer + bytes);
            for(std::size_t off = 0; off + sizeof(char*) <= bytes; off += sizeof(char*))
            {
                char* p = nullptr;
                std::memcpy(&p, patched.data() + off, sizeof(char*));
                for(std::size_t i = 0; i < state->recorded_ptrs.size(); ++i)
                {
                    const auto* base = static_cast<const char*>(state->recorded_ptrs[i]);
                    if(p >= base and p < base + state->recorded_sizes[i])
                    {
                        const char* np = static_cast<const char*>(current_ptrs[i]) + (p - base);
                        std::memcpy(patched.data() + off, &np, sizeof(char*));
                        break;
                    }
                }
            }
            void* extra[]       = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                   patched.data(),
                                   HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                   &bytes,
                                   HIP_LAUNCH_PARAM_END};
            params.extra        = extra;
            params.kernelParams = nullptr;
            check_hip(hipGraphExecKernelNodeSetParams(state->exec.get(), node, &params),
                      "hipGraphExecKernelNodeSetParams");
        }
    }

    // Snapshot the leaf buffers baked into the freshly recorded `graph`.
    void record_buffers(const std::vector<argument>& args) const
    {
        state->recorded_ptrs.clear();
        state->recorded_sizes.clear();
        for(const auto& a : args)
            leaf_buffers(a, state->recorded_ptrs, &state->recorded_sizes);
    }

    argument compute(context& ctx,
                     const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& mods,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
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
        // The leaf buffers the captured graph is bound to. If any of them moved,
        // the executable graph must be re-bound.
        auto current_ptrs = leaf_ptrs(args);

        if(not state->captured)
        {
            record();
            instantiate();
            record_buffers(args);
            state->applied_ptrs = current_ptrs;
            state->patchable    = collect_kernel_nodes();
            state->captured     = true;
        }
        else if(current_ptrs != state->applied_ptrs)
        {
            // Patch the new addresses into the executable graph node-by-node when
            // possible; otherwise re-record and let hipGraphExecUpdate patch it
            // (re-instantiating if the topology cannot be updated).
            if(state->patchable)
            {
                patch_kernel_nodes(current_ptrs);
            }
            else
            {
                record();
                hipGraphNode_t error_node       = nullptr;
                hipGraphExecUpdateResult result = hipGraphExecUpdateError;
                auto status =
                    hipGraphExecUpdate(state->exec.get(), state->graph.get(), &error_node, &result);
                if(status != hipSuccess or result != hipGraphExecUpdateSuccess)
                    instantiate();
                record_buffers(args);
            }
            state->applied_ptrs = std::move(current_ptrs);
        }

        check_hip(hipGraphLaunch(state->exec.get(), stream), "hipGraphLaunch");
        return pack_outputs(state->outputs);
    }
};

MIGRAPHX_REGISTER_OP(hip_graph)

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
