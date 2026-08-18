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
#include <migraphx/env.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/module.hpp>
#include <migraphx/module_ref.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/program.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/register_op.hpp>
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
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
    return {outputs};
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

// The [begin, end) byte range of a buffer.
using address_range = std::pair<std::uintptr_t, std::uintptr_t>;

// The byte range of each leaf buffer, in leaf order.
static std::vector<address_range> leaf_bounds(const std::vector<argument>& leaves)
{
    std::vector<address_range> bounds;
    std::transform(
        leaves.begin(), leaves.end(), std::back_inserter(bounds), [](const argument& leaf) {
            auto base = reinterpret_cast<std::uintptr_t>(leaf.data());
            return std::make_pair(base, base + leaf.get_shape().bytes());
        });
    return bounds;
}

// The leaf whose byte range contains `p`, as (leaf index, byte offset within
// the leaf); nullopt when the address is not in any leaf.
static optional<std::pair<std::size_t, std::size_t>>
locate_leaf(const char* p, const std::vector<address_range>& bounds)
{
    auto addr = reinterpret_cast<std::uintptr_t>(p);
    auto it   = std::find_if(bounds.begin(), bounds.end(), [&](const address_range& bound) {
        return addr >= bound.first and addr < bound.second;
    });
    if(it == bounds.end())
        return nullopt;
    return std::make_pair(static_cast<std::size_t>(std::distance(bounds.begin(), it)),
                          static_cast<std::size_t>(addr - it->first));
}

// True when `consumer` passes the buffer of its input `input` straight through
// (a view op such as reshape/slice/load) to its own outputs.
static bool aliases_input(instruction_ref consumer, instruction_ref input)
{
    return contains(instruction::get_output_alias(consumer, true), input);
}

// The submodule's parameter names in input-binding order. Sorted by name
// because get_parameter_names() order is not serialized (it would not survive
// save/load), while sorted names match find_inputs' name-ordered inputs.
static std::vector<std::string> input_param_names(const_module_ref sub)
{
    auto param_names = sub->get_parameter_names();
    std::sort(param_names.begin(), param_names.end());
    return param_names;
}

// Bind the submodule's parameters (in input order) to the op's input arguments.
static std::unordered_map<std::string, argument> create_params(const_module_ref sub,
                                                               const std::vector<argument>& args)
{
    auto param_names = input_param_names(sub);
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
        hipGraph_t g = nullptr;
        try
        {
            f();
        }
        catch(...)
        {
            // End the capture so the stream stays usable; discard the partial graph.
            if(hipStreamEndCapture(stream, &g) == hipSuccess)
                hip_graph_ptr partial{g};
            throw;
        }
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
        // The second query may report fewer nodes than the first.
        handles.resize(num);
        std::vector<node> result(handles.size());
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
        // the exec cannot be updated in place and must be re-instantiated.
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

// One pointer word in a kernel node's packed argument buffer: where it sits,
// which flattened movable-input leaf it points into, and where within that
// leaf (nonzero for a viewed/sliced input). A rebind rewrites exactly these.
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
    // Launch params and argument buffer fetched once at plan build; a rebind
    // rewrites only the buffer's pointer slots. Kept alive here because HIP does
    // not document copying `extra` at SetParams time.
    hipKernelNodeParams params{};
    std::vector<char> buffer{};
    std::size_t buffer_size = 0;
    std::array<void*, 5> config{};
};

// Records the work of a submodule into a HIP graph the first time it is run and
// replays the instantiated graph on every subsequent run, amortizing the
// per-launch CPU overhead of issuing many kernels/library calls.
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
        // Addresses of the movable leaves currently bound in the captured
        // graph, used to detect when a parameter buffer has moved.
        std::vector<const void*> applied_ptrs{};
        // True when every movable parameter is consumed only by code-object
        // kernels we can patch; `patches` then lists, per such node, the slots to
        // rewrite. False (a parameter reaches a library kernel) -> re-record.
        bool patchable = false;
        std::vector<graph_node_patch> patches{};
        // A captured output whose data lies inside a movable leaf. Patching
        // rewrites only kernel arguments, so when a leaf moves the cached
        // outputs/result must be rebased onto the new address too.
        struct output_rebind
        {
            std::size_t output;
            std::size_t leaf;
            std::size_t offset;
        };
        std::vector<output_rebind> output_rebinds{};
        // Serializes evals: the state is mutated on capture/rebind and may be
        // shared by copies of a compiled program (see hip_graph_op::state).
        std::mutex mutex{};

        // Capture `f`'s device work into a fresh graph and cache the packed result.
        // `f` returns the submodule outputs; capturing only records the launches,
        // so the buffers are filled when the instantiated graph is later launched.
        template <class F>
        void record(hipStream_t stream, F f)
        {
            graph  = hip_graph::capture(stream, [&] { outputs = f(); });
            result = pack_outputs(outputs);
        }

        // Rebase the cached outputs (and the packed result) onto the current
        // movable-leaf buffers after the kernel nodes were patched to use them.
        void rebind_outputs(const std::vector<argument>& leaves)
        {
            for(const auto& r : output_rebinds)
            {
                assert(r.output < outputs.size());
                assert(r.leaf < leaves.size());
                assert(r.offset + outputs[r.output].get_shape().bytes() <=
                       leaves[r.leaf].get_shape().bytes());
                outputs[r.output] = {outputs[r.output].get_shape(),
                                     leaves[r.leaf].data() + r.offset};
            }
            if(not output_rebinds.empty())
                result = pack_outputs(outputs);
        }
    };

    // Created in finalize(), which runs once per instruction, so each gets its
    // own state (a construction-time state would be shared by every copy-on-write
    // copy of the operator made during compilation). Copies of an already-compiled
    // program do not re-run finalize and so share this state; graph_state's mutex
    // keeps that sharing safe.
    std::shared_ptr<graph_state> state{};

    // Indices of the inputs the captured outputs are written into (and so
    // alias); these passed-in buffers have global lifetime, so the outputs can
    // be returned safely.
    std::vector<std::size_t> aliases{};

    // Indices of the inputs whose buffer can move between runs (the program
    // parameters); every other input is a fixed allocation/constant. When empty
    // the graph is bound to stable addresses and replayed without inspection.
    std::vector<std::size_t> replace_inputs{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        // The captured graph is runtime-only state and is excluded.
        return pack(f(self.aliases, "aliases"), f(self.replace_inputs, "replace_inputs"));
    }

    std::string name() const { return "hip::graph"; }

    std::vector<std::size_t> output_alias(const std::vector<shape>&) const { return aliases; }

    shape compute_shape(const std::vector<shape>&, std::vector<module_ref> mods) const
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

    // The movable inputs flattened to their leaf arguments, in replace_inputs
    // order (a tuple has no single data pointer, so rebinding tracks its leaves).
    std::vector<argument> movable_leaves(const std::vector<argument>& args) const
    {
        std::vector<argument> selected;
        selected.reserve(replace_inputs.size());
        std::transform(replace_inputs.begin(),
                       replace_inputs.end(),
                       std::back_inserter(selected),
                       [&](std::size_t idx) {
                           assert(idx < args.size());
                           return args[idx];
                       });
        return flatten(selected);
    }

    // Walk forward from a movable parameter through alias (view) ops, adding
    // each code-object kernel that consumes it to `out`, keyed by function
    // handle. Returns false if the parameter reaches a consumer whose argument
    // buffer we cannot interpret (a library gemm/conv), forcing a re-record.
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
                    // An in-place kernel (output aliases the input) passes the
                    // buffer to its readers; follow so they are patched too.
                    if(aliases_input(consumer, ins))
                        self(consumer);
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

    // Build the per-node patch plan: match captured kernel nodes to the
    // code-object kernels consuming a movable parameter (by function handle) and
    // record their pointer slots, using the kernel_args layout to skip inlined
    // scalars. Returns false when the graph must be re-recorded instead.
    bool build_patch_plan(const std::vector<address_range>& bounds, const_module_ref sub) const
    {
        // Address-range lookup (locate_leaf) is only unambiguous when the leaf
        // ranges are disjoint; overlapping buffers would attribute one leaf's
        // slots to another and later patch them with the wrong address.
        auto sorted_bounds = bounds;
        std::sort(sorted_bounds.begin(), sorted_bounds.end());
        if(std::adjacent_find(
               sorted_bounds.begin(), sorted_bounds.end(), [](const auto& b1, const auto& b2) {
                   return b2.first < b1.second;
               }) != sorted_bounds.end())
            return false;

        // The code-object kernels consuming a movable parameter, keyed by their
        // function handle so a captured node can be matched back to one.
        std::unordered_map<void*, const std::map<std::size_t, kernel_argument_value>*>
            code_object_args;
        auto param_names = input_param_names(sub);
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
            // A buffer that cannot be parsed back cannot be patched; re-record
            // rather than leave it bound to the captured addresses.
            auto buffer = unpack_kernel_config(params.extra);
            if(buffer.empty())
                return false;

            graph_node_patch np;
            np.node = node;
            for(const auto& [off, p] : unpack_pointer_args(buffer, *cobj->second))
            {
                if(auto loc = locate_leaf(p, bounds))
                    np.slots.push_back({off, loc->first, loc->second});
            }
            if(np.slots.empty())
                continue;
            np.params = params;
            np.buffer = std::move(buffer);
            patches.push_back(std::move(np));
        }

        state->patches = std::move(patches);
        return true;
    }

    // Overwrite the recorded pointer slots with the current leaf addresses and
    // hand the buffers back to the nodes; the caller re-syncs the executable graph.
    void patch_kernel_nodes(const std::vector<argument>& leaves) const
    {
        for(auto& np : state->patches)
        {
            for(const auto& slot : np.slots)
            {
                assert(slot.leaf < leaves.size());
                // The re-bound leaf is assumed to keep the captured shape, so the
                // captured within-leaf offset still lands inside its buffer.
                assert(slot.ptr_offset < leaves[slot.leaf].get_shape().bytes());
                assert(slot.offset + sizeof(char*) <= np.buffer.size());
                const char* p = leaves[slot.leaf].data() + slot.ptr_offset;
                write_pointer(np.buffer.data() + slot.offset, p);
            }
            // The buffer, size, and config array are stored on the patch (not
            // locals) so they outlive this call; see graph_node_patch.
            np.buffer_size      = np.buffer.size();
            np.config           = pack_kernel_config(np.buffer.data(), &np.buffer_size);
            auto params         = np.params;
            params.extra        = np.config.data();
            params.kernelParams = nullptr;
            np.node.set_kernel_node_params(params);
        }
    }

    // Locate each captured output within the movable leaves (see
    // graph_state::output_rebind).
    static std::vector<graph_state::output_rebind>
    find_output_rebinds(const std::vector<argument>& outputs,
                        const std::vector<address_range>& bounds)
    {
        std::vector<optional<std::pair<std::size_t, std::size_t>>> locs;
        std::transform(outputs.begin(),
                       outputs.end(),
                       std::back_inserter(locs),
                       [&](const argument& out) { return locate_leaf(out.data(), bounds); });
        std::vector<graph_state::output_rebind> rebinds;
        auto indices = range(outputs.size());
        transform_if(
            indices.begin(),
            indices.end(),
            std::back_inserter(rebinds),
            [&](std::size_t i) { return locs[i].has_value(); },
            [&](std::size_t i) {
                return graph_state::output_rebind{i, locs[i]->first, locs[i]->second};
            });
        return rebinds;
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
        auto run_sub   = [&] { return run(sub, create_params(sub, args)); };

        hipStream_t stream = ctx.get_stream().get();
        // Neither the legacy/null stream nor tracing (which reads results
        // mid-run) can be combined with capture; fall back to a normal run. A
        // trace supplied via execution_environment cannot be detected here.
        if(stream == nullptr or value_of(MIGRAPHX_TRACE_EVAL{}) > 0)
            return pack_outputs(run_sub());

        std::lock_guard<std::mutex> lock(state->mutex);

        // See record(): capture only records the launches; the launch below
        // fills the buffers.
        if(not state->captured)
        {
            state->record(stream, run_sub);
            state->exec = state->graph.instantiate();
            // Only inspect the captured nodes when a parameter can actually move;
            // with no movable inputs the graph stays bound to stable buffers.
            if(not replace_inputs.empty())
            {
                auto leaves           = movable_leaves(args);
                auto bounds           = leaf_bounds(leaves);
                state->patchable      = build_patch_plan(bounds, sub);
                state->output_rebinds = find_output_rebinds(state->outputs, bounds);
                state->applied_ptrs   = leaf_ptrs(leaves);
            }
            state->captured = true;
        }
        else if(not replace_inputs.empty())
        {
            // Re-bind only when a parameter buffer has moved since the last run.
            auto leaves       = movable_leaves(args);
            auto current_ptrs = leaf_ptrs(leaves);
            if(current_ptrs != state->applied_ptrs)
            {
                // Patch the kernel nodes in place, or re-record a non-patchable
                // graph; either way re-sync the executable graph.
                if(state->patchable)
                {
                    patch_kernel_nodes(leaves);
                    // Kernels now write into the moved buffers; the cached
                    // outputs viewing those buffers must move with them.
                    state->rebind_outputs(leaves);
                }
                else
                {
                    state->record(stream, run_sub);
                }
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
