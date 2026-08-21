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
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/split_factor.hpp>
#include <migraphx/bit.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

// Past this many per-thread accumulator slots the `block` reduce spills VGPRs to
// scratch and its code object explodes (page-faulted on gfx1100 for slme); above
// it we fall back to the lazy-reread `block_large` reduce. Gating on the
// iteration count (not relements) covers every shape and block_size. Safety cap;
// retune if it regresses medium reductions. See
// test/gpu/fused_reduce_block_size_guard.cpp.
static constexpr std::size_t block_reduce_max_iterations = 16;

// ceil(relements / block_size): per-thread accumulator slots in the `block` reduce.
static std::size_t block_reduce_iterations(std::size_t relements, std::size_t block_size)
{
    return (relements + block_size - 1) / block_size;
}

static const char* const simple_reduce_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/reduce.hpp>
#include <migraphx/kernels/vectorize.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {
MIGRAPHX_GLOBAL void reduce_kernel(void* input_p, void* output_p) 
{
    
    transform_args(make_tensors(), ${transformers})(input_p, output_p)([](auto input, auto output) {

        simple_reduce<reduce::${algo}>(${reduction}, ${init}, input, output, ${read}, ${write});
    });
}
    
}

} // namespace migraphx

)__migraphx__";

static std::vector<std::size_t> get_reduce_lens(const std::vector<std::size_t>& input_lens,
                                                const std::vector<std::size_t>& output_lens)
{
    std::vector<std::size_t> reduce_lens;
    std::transform(output_lens.begin(),
                   output_lens.end(),
                   input_lens.begin(),
                   std::back_inserter(reduce_lens),
                   [](auto x, auto y) -> std::size_t {
                       if(x == y)
                           return 1;
                       else
                           return y;
                   });
    return reduce_lens;
}

static shape get_input_shape(const std::vector<shape>& inputs)
{
    auto it = std::max_element(inputs.begin(), inputs.end(), by(std::less<>{}, [](const shape& s) {
                                   return s.elements();
                               }));
    return *it;
}

template <class T>
static shape get_reduced_shape(const shape& s, const std::vector<T>& axes)
{
    auto lens = s.lens();
    std::fill(lens.begin(), lens.end(), 1);
    for(const auto& axis : axes)
        lens[axis] = s.lens()[axis];
    return s.with_lens(lens);
}

template <class T>
static shape get_output_shape(const shape& s, const std::vector<T>& axes)
{
    auto lens = s.lens();
    for(const auto& axis : axes)
        lens[axis] = 1;
    return s.with_lens(lens);
}

/// The minimum stride of the input along the reduced dimensions, skipping the
/// dimensions where the input only has a single element since their strides
/// are meaningless. Returns the max value when the input has no reduced
/// dimensions at all.
template <class ReduceLens>
static std::size_t min_reduce_stride(const shape& input, const ReduceLens& rlens)
{
    const auto init = std::numeric_limits<std::size_t>::max();
    auto is         = range(rlens.size());
    return transform_accumulate(
        is.begin(), is.end(), init, MIGRAPHX_LIFT(std::min), [&](auto i) -> std::size_t {
            if(rlens[i] == 1 or input.lens()[i] == 1)
                return init;
            return input.strides()[i];
        });
}

template <class ReduceLens>
static std::string get_reduce_algo(context& ctx, const std::vector<shape>& inputs, ReduceLens rlens)
{
    auto relements = std::accumulate(rlens.begin(), rlens.end(), 1, std::multiplies<>{});
    // Use the memory layout of the dominant inputs to decide the algorithm
    // since the small inputs will be served from cache either way
    auto max_bytes = std::max_element(inputs.begin(),
                                      inputs.end(),
                                      by(std::less<>{}, [](const shape& s) { return s.bytes(); }))
                         ->bytes();
    // Use lane when any dominant input is strided along the reduction: the
    // lane algorithm reads the strided input coalesced across the outputs,
    // whereas the block algorithm would waste most of every cacheline on it.
    // Dense inputs still stream whole cachelines per lane.
    bool is_strided_reduce = std::any_of(inputs.begin(), inputs.end(), [&](const shape& input) {
        if(input.bytes() * 4 < max_bytes)
            return false;
        auto min_stride = min_reduce_stride(input, rlens);
        // Inputs with no reduced dimensions do not read along the reduction
        if(min_stride == std::numeric_limits<std::size_t>::max())
            return false;
        return min_stride > 2;
    });
    if(is_strided_reduce)
        return "lane";
    if(relements <= ctx.get_current_device().get_wavefront_size())
        return "wave";
    return "block";
}

static std::string get_reduce_algo(context& ctx, const std::vector<shape>& inputs)
{
    auto rlens = get_reduce_lens(inputs.front().lens(), inputs.back().lens());
    return get_reduce_algo(ctx, inputs, rlens);
}

static std::size_t compute_subwave_size(context& ctx, std::size_t n)
{
    std::size_t max_wavefront_size = ctx.get_current_device().get_wavefront_size();
    std::size_t wavefront_size     = 1;
    while(wavefront_size <= n and wavefront_size < max_wavefront_size)
        wavefront_size *= 2;
    return wavefront_size;
}

struct reduce_tile
{
    std::size_t axis = 0;
    std::size_t size = 1;
};

struct strided_tile
{
    std::size_t block_size = 256;
    std::size_t out_tile   = 64;
    bool large             = false;

    std::string algo() const
    {
        return "block_strided<" + std::to_string(out_tile) + (large ? ", true" : "") + ">";
    }
};

/// The lane algorithm needs one thread per output. When there are too few
/// outputs to fill the device, a workgroup instead computes a tile of
/// consecutive outputs and the remaining lanes parallelize each reduction.
static optional<strided_tile>
find_strided_tile(context& ctx, std::size_t relements, std::size_t block_size)
{
    strided_tile result{block_size, ctx.get_current_device().get_wavefront_size()};
    if(result.block_size <= result.out_tile or (result.block_size % result.out_tile) != 0)
        return nullopt;
    auto seg_lanes = result.block_size / result.out_tile;
    // The unrolled stride loop with per-thread register storage is limited to
    // 256 iterations, so larger reductions re-read the inputs lazily instead
    result.large = (relements + seg_lanes - 1) / seg_lanes > 255;
    return result;
}

/// Find a non-reduced axis where most of the input bytes are broadcast, so all
/// the outputs along that axis read the same data. Computing such outputs in
/// the same workgroup lets the broadcast input be loaded from cache instead of
/// being streamed from memory once per output.
static optional<reduce_tile> find_reduce_tile(const std::vector<shape>& inputs,
                                              std::size_t noutputs,
                                              const shape& reduce_output_shape,
                                              const std::vector<std::size_t>& reduce_lens)
{
    const std::size_t min_tile  = 2;
    const std::size_t max_tile  = 16;
    const std::size_t min_bytes = 8388608; // 8MB
    // The loads can only be reused from cache when the reduction reads are
    // dense, otherwise the cacheline footprint of a single output exceeds the
    // cache
    auto reusable = [&](const shape& input) {
        auto min_stride = min_reduce_stride(input, reduce_lens);
        return min_stride <= 2 or min_stride == std::numeric_limits<std::size_t>::max();
    };
    // Sum the memory footprint of the inputs that are reused(or not) across the
    // outputs along the axis, skipping the output shapes since they are only
    // written once per element
    auto axis_bytes = [&](std::size_t axis, bool reused) {
        return transform_accumulate(inputs.begin(),
                                    inputs.end() - noutputs,
                                    std::size_t{0},
                                    std::plus<>{},
                                    [&](const shape& input) -> std::size_t {
                                        if((input.strides()[axis] == 0 and reusable(input)) ==
                                           reused)
                                            return input.bytes();
                                        return 0;
                                    });
    };
    using candidate    = std::pair<std::size_t, reduce_tile>;
    auto get_candidate = [&](std::size_t axis) -> optional<candidate> {
        auto extent = reduce_output_shape.lens()[axis];
        if(extent < min_tile)
            return nullopt;
        auto tile = extent / split_dim(extent, max_tile);
        if(tile < min_tile or tile > max_tile)
            return nullopt;
        auto saved = axis_bytes(axis, true);
        if(saved < min_bytes)
            return nullopt;
        // The non-broadcast inputs are re-read once per output in the tile, so
        // tiling should only be done when the broadcast bytes dominate
        if(saved < axis_bytes(axis, false) * 4)
            return nullopt;
        return candidate{saved, reduce_tile{axis, tile}};
    };
    auto is = range(reduce_output_shape.lens().size());
    std::vector<optional<candidate>> candidates;
    std::transform(is.begin(), is.end(), std::back_inserter(candidates), get_candidate);
    auto it = std::max_element(candidates.begin(),
                               candidates.end(),
                               by(std::less<>{}, [](const optional<candidate>& c) -> std::size_t {
                                   return c.has_value() ? c->first : 0;
                               }));
    if(it == candidates.end() or not it->has_value())
        return nullopt;
    return (*it)->second;
}

/// This will adjust the input shapes so a partial reduction is done per workgroup.
/// This is done by splitting the reduction axis so each split group becomes
/// part of the batch. So if we want to do a split redution of a tensor
/// {K}, then this will create a tensor of {K/N, N} where N is the number of
/// split groups. To compute the number of split groups it finds the largest
/// divisor that can divide K to make it less than min_size.
/// The max_splits parameter limits the maximum number of split groups to prevent excessive
/// splitting.
static std::vector<shape> split_reduce(const std::vector<shape>& inputs,
                                       std::size_t min_size   = 8192,
                                       std::size_t max_splits = 16)
{
    std::vector<shape> result;
    auto input_shape         = inputs.front();
    const auto& reduce_shape = inputs[inputs.size() - 2];
    const auto& output_shape = inputs[inputs.size() - 1];

    auto is          = range(reduce_shape.lens().size());
    using array_type = std::array<std::size_t, 2>;
    auto initial     = array_type{std::numeric_limits<std::size_t>::max(),
                              std::numeric_limits<std::size_t>::max()};
    auto faxis       = transform_accumulate(
        is.begin(), is.end(), initial, MIGRAPHX_LIFT(std::min), [&](auto i) -> array_type {
            if(input_shape.lens()[i] == output_shape.lens()[i])
                return initial;
            return {input_shape.strides()[i], std::size_t(i)};
        })[1];

    assert(faxis < reduce_shape.lens().size());

    std::size_t r = input_shape.lens()[faxis];
    // r0 is the size of the reduction along the other axes(that is not faxis)
    std::size_t r0 = reduce_shape.elements() / r;
    // Scale the min_size by r0 to account for the reduction from other axes.
    std::size_t n = split_dim(r, std::min<std::size_t>(min_size / r0, 1), max_splits);
    assert(n != 1);
    std::transform(
        inputs.begin(), inputs.end(), std::back_inserter(result), [&](const shape& s) -> shape {
            auto lens    = s.lens();
            auto strides = s.strides();

            lens.push_back(n);
            if(lens[faxis] == 1)
            {
                strides.push_back(0);
            }
            else
            {
                lens[faxis] /= n;
                strides.push_back(strides[faxis] * lens[faxis]);
            }

            return {s.type(), lens, strides};
        });
    return reduce_dims(normalize_permutation(result));
}

struct simple_reduce_compiler : compiler<simple_reduce_compiler>
{
    std::vector<std::string> names() const
    {
        return {"simple_reduce",
                "reduce_sum",
                "reduce_mean",
                "reduce_max",
                "reduce_min",
                "reduce_prod",
                "reduce_any",
                "reduce_all"};
    }

    static std::size_t get_reduce_elements(const std::vector<shape>& inputs)
    {
        return inputs.front().elements() / inputs.back().elements();
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.virtual_inputs = reduce_dims(inputs);
        auto faxis             = find_fast_axis({options.virtual_inputs.front()});
        vectorize vec{};
        auto nelements = options.virtual_inputs.back().elements();
        auto algo      = v.get("algo", get_reduce_algo(ctx, options.virtual_inputs));
        if(algo == "block" or algo == "wave")
        {
            // Vectorize if the axis is a reduction axis
            if(options.virtual_inputs.back().lens()[faxis] == 1)
                vec = vectorize::elements(ctx, faxis, options.virtual_inputs);
            auto relements  = get_reduce_elements(options.virtual_inputs) / vec.size;
            if(algo == "block")
            {
                auto block_size = compute_block_size(ctx, relements, 256);
                // Same iteration cap as fused_reduce (see block_reduce_max_iterations).
                if(block_reduce_iterations(relements, block_size) > block_reduce_max_iterations)
                    algo = "block_large";
                options.set_launch_params(
                    v, compute_global_for(ctx, nelements * block_size, 256), block_size);
            }
            else
            {
                auto subwave_size = compute_subwave_size(ctx, relements);
                algo              = "subwave<" + std::to_string(subwave_size) + ">";
                options.set_launch_params(v,
                                          compute_global_for(ctx, nelements * subwave_size, 256),
                                          ctx.get_current_device().get_wavefront_size());
            }
        }
        else if(algo == "lane")
        {
            auto relements   = get_reduce_elements(options.virtual_inputs);
            bool few_outputs = nelements < ctx.get_current_device().get_cu_count() * 1024;
            optional<strided_tile> stile;
            if(few_outputs)
                stile = find_strided_tile(ctx, relements, 256);
            if(stile.has_value())
            {
                algo         = stile->algo();
                auto ngroups = (nelements + stile->out_tile - 1) / stile->out_tile;
                options.set_launch_params(v,
                                          compute_global_for(ctx, ngroups * stile->block_size, 256),
                                          stile->block_size);
            }
            else
            {
                options.set_launch_params(v, compute_global_for(ctx, nelements, 256));
            }
        }
        else
        {
            MIGRAPHX_THROW("Unknown reduce algo: " + algo);
        }
        options.kernel_name  = "reduce_kernel";
        std::string identity = "[](auto x) { return x; }";
        auto src             = interpolate_string(simple_reduce_kernel,
                                      {{"reduction", v.at("reduction").to<std::string>()},
                                       {"init", v.get("init", std::string{"0"})},
                                       {"read", v.get("read", identity)},
                                       {"write", v.get("write", identity)},
                                       {"algo", algo},
                                       {"transformers", make_transformer_args(vec)},
                                       {"preamble", v.get("preamble", std::string{})}});
        options.emplace_param("-Wno-float-equal");
        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        value v = value::object{};
        reduce_op r{};
        r.set(ins, op);
        v["reduction"] = r.reduction;
        v["read"]      = r.read;
        v["write"]     = r.write;
        v["init"]      = r.init;
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }
};

static const char* const fused_reduce_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/reduce.hpp>
#include <migraphx/kernels/pointwise.hpp>
#include <migraphx/kernels/vectorize.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {
MIGRAPHX_GLOBAL void ${kernel}(${params})
{
    transform_args(make_tensors(), ${transformers}, rotate_and_pack_last<${noutputs}>())(${args})([](auto y, auto... xs) {
        fused_reduce<reduce::${algo}, ${reduced}>(y, ${assign}{}, partial(${lambda})(xs...));
    });
}
    
}

} // namespace migraphx

)__migraphx__";

namespace {
struct fused_reduce_plan
{
    std::vector<shape> finputs        = {};
    std::vector<shape> virtual_inputs = {};
    shape reduce_output_shape         = {};
    shape reduction_shape             = {};
    vectorize vec                     = {};
    std::string algo                  = {};
    std::string assign                = {};
    std::size_t relements             = 0;
};

// Computes the virtual inputs, default reduction algorithm, vectorization, and vectorized
// number of reduction elements. This is shared by both compilation and tuning so that the
// tuning config's default solution matches what compile_op would pick on its own.
fused_reduce_plan
compute_fused_reduce_plan(context& ctx, const std::vector<shape>& inputs, const value& v)
{
    fused_reduce_plan plan;
    plan.assign         = v.get("assign", "assign_none");
    auto axes           = v.at("axes").to_vector<std::size_t>();
    plan.finputs        = flatten_tuple_shapes(inputs);
    plan.virtual_inputs = plan.finputs;
    plan.virtual_inputs.push_back(get_reduced_shape(get_input_shape(plan.finputs), axes));
    plan.virtual_inputs.push_back(get_output_shape(get_input_shape(plan.finputs), axes));
    plan.virtual_inputs = reduce_dims(normalize_permutation(plan.virtual_inputs));
    if(plan.assign != "assign_none")
        plan.virtual_inputs = split_reduce(plan.virtual_inputs);
    plan.reduce_output_shape = plan.virtual_inputs.back();
    plan.virtual_inputs.pop_back();
    plan.reduction_shape = plan.virtual_inputs.back();
    plan.virtual_inputs.pop_back();

    auto faxis = find_fast_axis({plan.virtual_inputs.front()});
    plan.algo =
        v.get("algo", get_reduce_algo(ctx, plan.virtual_inputs, plan.reduction_shape.lens()));
    bool no_vectorize = v.get("no_vectorize", false);
    if((plan.algo == "block" or plan.algo == "block_tile" or plan.algo == "wave") and
       plan.reduce_output_shape.lens()[faxis] == 1 and not no_vectorize)
        plan.vec = vectorize::elements(ctx, faxis, plan.virtual_inputs);
    plan.relements = plan.reduction_shape.elements() / plan.vec.size;
    return plan;
}

/// The lane algorithm should be replaced with block_strided when there are
/// too few outputs to fill the device with one lane per output, or when a
/// dominant input is dense along the reduction: the segmented lanes read the
/// dense input in coalesced segments which lane cant do.
bool prefer_block_strided(context& ctx, const fused_reduce_plan& plan, std::size_t noutputs)
{
    bool few_outputs =
        plan.reduce_output_shape.elements() < ctx.get_current_device().get_cu_count() * 1024;
    auto max_bytes = std::max_element(plan.virtual_inputs.begin(),
                                      plan.virtual_inputs.end(),
                                      by(std::less<>{}, [](const shape& s) { return s.bytes(); }))
                         ->bytes();
    bool mixed_density = std::any_of(
        plan.virtual_inputs.begin(), plan.virtual_inputs.end() - noutputs, [&](const shape& input) {
            if(input.bytes() * 4 < max_bytes)
                return false;
            return min_reduce_stride(input, plan.reduction_shape.lens()) <= 2;
        });
    return few_outputs or mixed_density;
}
} // namespace

struct fused_reduce_compiler : compiler<fused_reduce_compiler>
{
    std::vector<std::string> names() const { return {"fused_reduce", "split_fused_reduce"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        auto plan      = compute_fused_reduce_plan(ctx, inputs, v);
        auto noutputs  = plan.finputs.size() - inputs.size() + 1;
        auto algo      = plan.algo;
        auto relements = plan.relements;
        auto nelements = plan.reduce_output_shape.elements();

        hip_compile_options options;
        options.inputs         = plan.finputs;
        options.output         = inputs.back();
        options.virtual_inputs = plan.virtual_inputs;
        if(algo == "block" or algo == "block_tile")
        {
            auto n_per_block = v.get("n_per_block", std::size_t{1});
            auto block_size  = v.get("block_size", compute_block_size(ctx, relements, 1024));
            assert(n_per_block > 0);
            assert(block_size > 0);
            assert(nelements % n_per_block == 0);
            // Fall back to block_large once the per-thread register array exceeds
            // the safety cap (see block_reduce_max_iterations).
            if(block_reduce_iterations(relements, block_size) > block_reduce_max_iterations)
            {
                algo = "block_large";
            }
            else if(algo == "block_tile")
            {
                auto tile_axis = v.at("tile_axis").to<std::size_t>();
                // Smaller workgroups keep the reused loads resident in cache
                block_size = v.get(
                    "block_size", compute_block_size(ctx, relements, n_per_block == 2 ? 512 : 256));
                algo = "block_tile<" + std::to_string(tile_axis) + ", " +
                       std::to_string(n_per_block) + ">";
            }
            options.set_launch_params(
                v, compute_global_for(ctx, nelements * block_size / n_per_block, 256), block_size);
        }
        else if(algo == "wave")
        {
            auto subwave_size = v.get("subwave_size", compute_subwave_size(ctx, relements));
            algo              = "subwave<" + std::to_string(subwave_size) + ">";
            options.set_launch_params(v,
                                      compute_global_for(ctx, nelements * subwave_size, 256),
                                      ctx.get_current_device().get_wavefront_size());
        }
        else if(algo == "block_strided")
        {
            auto stile = find_strided_tile(ctx, relements, v.get("block_size", 256));
            if(not stile.has_value())
                MIGRAPHX_THROW("Invalid block_size for block_strided reduce");
            algo         = stile->algo();
            auto ngroups = (nelements + stile->out_tile - 1) / stile->out_tile;
            options.set_launch_params(
                v, compute_global_for(ctx, ngroups * stile->block_size, 256), stile->block_size);
        }
        else if(algo == "lane")
        {
            options.set_launch_params(v, compute_global_for(ctx, nelements, 256));
        }
        else
        {
            MIGRAPHX_THROW("Unknown reduce algo: " + algo);
        }
        options.kernel_name = v.get("kernel", "reduce_kernel");
        auto reduced        = "decltype(" + generate_make_shape(plan.reduce_output_shape) + ")";
        auto src =
            interpolate_string(fused_reduce_kernel,
                               {{"kernel", options.kernel_name},
                                {"params", enum_params(plan.finputs.size(), "void * private_p")},
                                {"args", enum_params(plan.finputs.size(), "private_p")},
                                {"assign", plan.assign},
                                {"algo", algo},
                                {"reduced", reduced},
                                {"lambda", v.at("lambda").to<std::string>()},
                                {"transformers", make_transformer_args(plan.vec)},
                                {"noutputs", std::to_string(noutputs)},
                                {"preamble", v.get("preamble", std::string{})}});
        options.emplace_param("-Wno-float-equal");
        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace
    compile(context& ctx, instruction_ref ins, const operation& op, const value& solution) const
    {
        assert(not ins->module_inputs().empty());
        auto v        = op.to_value();
        for(const auto& x : solution)
            v.insert(x);
        auto* rm      = ins->module_inputs().front();
        auto shapes   = to_shapes(ins->inputs());
        v["preamble"] = generate_reduce(*rm, "fused_reduce_op");
        v["lambda"]   = "MIGRAPHX_LIFT(fused_reduce_op)";
        v["kernel"]   = generate_name_from_ops(*rm) + "_kernel";
        return compile_op(ctx, shapes, v);
    }

    /// Add a solution for the algo with the given block size, plus the
    /// larger-block alternative when it differs. The extra parameters are
    /// included in each solution.
    static void add_block_size_solutions(tuning_config& tc,
                                         const std::string& algo,
                                         std::size_t block_size,
                                         std::size_t large_block_size,
                                         const value& extra = value::object{})
    {
        auto solution          = extra;
        solution["algo"]       = algo;
        solution["block_size"] = block_size;
        tc.solutions.push_back(solution);
        if(large_block_size != block_size)
        {
            solution["block_size"] = large_block_size;
            tc.solutions.push_back(solution);
        }
    }

    optional<tuning_config>
    get_tuning_config(context& ctx, instruction_ref ins, const operation& op, bool exhaustive) const
    {
        if(not contains({"fused_reduce", "split_fused_reduce"}, op.name()))
            return nullopt;
        tuning_config tc;
        auto shapes   = to_shapes(ins->inputs());
        tc.problem    = to_value(shapes);
        auto plan     = compute_fused_reduce_plan(ctx, shapes, op.to_value());
        auto noutputs = plan.finputs.size() - shapes.size() + 1;
        auto tile     = find_reduce_tile(
            plan.virtual_inputs, noutputs, plan.reduce_output_shape, plan.reduction_shape.lens());
        if(not exhaustive)
        {
            // Without exhaustive tuning, offer the heuristic default algorithm plus a few
            // alternatives so benchmarking can decide: block_tile when a tile is
            // found, a larger block size (max 1024 instead of 256) for block, and
            // block_strided when the lane heuristics prefer it.
            if(plan.algo == "block")
            {
                if(tile.has_value() and plan.assign == "assign_none")
                {
                    // For the cache-bound tiled reduction a smaller workgroup
                    // that leaves about 4 elements per lane pipelines enough
                    // loads to often beat the default block size, so offer
                    // both and let benchmarking decide
                    std::size_t max_block = tile->size == 2 ? 512 : 256;
                    add_block_size_solutions(
                        tc,
                        "block_tile",
                        compute_block_size(
                            ctx, std::max<std::size_t>(plan.relements / 4, 1), max_block),
                        compute_block_size(ctx, plan.relements, max_block),
                        {{"tile_axis", tile->axis}, {"n_per_block", tile->size}});
                }
                add_block_size_solutions(tc,
                                         "block",
                                         compute_block_size(ctx, plan.relements, 256),
                                         compute_block_size(ctx, plan.relements, 1024));
            }
            else if(plan.algo == "lane" and prefer_block_strided(ctx, plan, noutputs) and
                    find_strided_tile(ctx, plan.relements, 256).has_value())
            {
                // A block_strided workgroup computes a tile of out_tile outputs at once, so
                // its block size is fitted to the parallel work across the whole tile
                // rather than a single reduction
                auto swork = ctx.get_current_device().get_wavefront_size() * plan.relements;
                add_block_size_solutions(tc,
                                         "block_strided",
                                         compute_block_size(ctx, swork, 256),
                                         compute_block_size(ctx, swork, 1024));
                tc.solutions.push_back({{"algo", "lane"}});
                add_block_size_solutions(tc,
                                         "block",
                                         compute_block_size(ctx, plan.relements, 256),
                                         compute_block_size(ctx, plan.relements, 1024));
            }
            else
            {
                tc.solutions.push_back({{"algo", plan.algo}});
            }
            return tc;
        }
        auto relements = plan.reduction_shape.elements();
        std::unordered_set<std::size_t> tile_sizes;
        for(auto per_lane : {1, 2, 4, 8, 16})
        {
            std::size_t x = relements / per_lane;
            for(auto max_block : {256, 512, 1024})
                tile_sizes.insert(compute_block_size(ctx, x, max_block));
            if(x < ctx.get_current_device().get_wavefront_size())
                tile_sizes.insert(bit_ceil(x));
        }
        for(auto tile_size : tile_sizes)
        {
            if(tile_size > ctx.get_current_device().get_wavefront_size())
                tc.solutions.push_back({{"algo", "block"}, {"block_size", tile_size}});
            else
                tc.solutions.push_back({{"algo", "wave"}, {"subwave_size", tile_size}});
        }
        tc.solutions.push_back({{"algo", "lane"}});
        for(auto block_size : {128, 256, 512, 1024})
            tc.solutions.push_back({{"algo", "block_strided"}, {"block_size", block_size}});
        if(tile.has_value() and plan.assign == "assign_none")
        {
            for(auto block_size : {64, 128, 256, 512})
                tc.solutions.push_back({{"algo", "block_tile"},
                                        {"block_size", block_size},
                                        {"tile_axis", tile->axis},
                                        {"n_per_block", tile->size}});
        }
        return tc;
    }
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
