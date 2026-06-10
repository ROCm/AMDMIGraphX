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
    // The unrolled stride loop is limited to 256 iterations
    if((relements + seg_lanes - 1) / seg_lanes > 255)
        return nullopt;
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
                if(relements >= block_size * 256)
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
                algo         = "block_strided<" + std::to_string(stile->out_tile) + ">";
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

struct fused_reduce_compiler : compiler<fused_reduce_compiler>
{
    std::vector<std::string> names() const { return {"fused_reduce", "split_fused_reduce"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        auto assign         = v.get("assign", "assign_none");
        auto axes           = v.at("axes").to_vector<std::size_t>();
        auto finputs        = flatten(inputs);
        auto noutputs       = finputs.size() - inputs.size() + 1;
        auto virtual_inputs = finputs;
        virtual_inputs.push_back(get_reduced_shape(get_input_shape(finputs), axes));
        virtual_inputs.push_back(get_output_shape(get_input_shape(finputs), axes));
        virtual_inputs = reduce_dims(normalize_permutation(virtual_inputs));
        if(assign != "assign_none")
            virtual_inputs = split_reduce(virtual_inputs);
        auto reduce_output_shape = virtual_inputs.back();
        virtual_inputs.pop_back();
        auto reduction_shape = virtual_inputs.back();
        virtual_inputs.pop_back();

        hip_compile_options options;
        options.inputs         = finputs;
        options.output         = inputs.back();
        options.virtual_inputs = virtual_inputs;
        auto faxis             = find_fast_axis({options.virtual_inputs.front()});
        vectorize vec{};
        auto nelements = reduce_output_shape.elements();
        auto algo =
            v.get("algo", get_reduce_algo(ctx, options.virtual_inputs, reduction_shape.lens()));
        optional<reduce_tile> tile = nullopt;
        if(assign == "assign_none" and
           (algo == "block_tile" or (algo == "block" and not v.contains("algo"))))
            tile = find_reduce_tile(
                options.virtual_inputs, noutputs, reduce_output_shape, reduction_shape.lens());
        if(algo == "block_tile")
            algo = "block";
        bool no_vectorize = v.get("no_vectorize", false);
        if(algo == "block" or algo == "wave")
        {
            if(reduce_output_shape.lens()[faxis] == 1 and not no_vectorize)
                vec = vectorize::elements(ctx, faxis, options.virtual_inputs);
            auto relements = reduction_shape.elements() / vec.size;
            if(algo == "block")
            {
                auto block_size = v.get("block_size", compute_block_size(ctx, relements, 1024));
                assert(block_size > 0);
                if(relements >= (block_size - 1) * 256)
                {
                    algo = "block_large";
                }
                else if(tile.has_value())
                {
                    // Smaller workgroups keep the reused loads resident in cache
                    auto max_block = tile->size == 2 ? 512 : 256;
                    block_size = v.get("block_size", compute_block_size(ctx, relements, max_block));
                    algo       = "block_tile<" + std::to_string(tile->axis) + ", " +
                           std::to_string(tile->size) + ">";
                    nelements /= tile->size;
                }
                options.set_launch_params(
                    v, compute_global_for(ctx, nelements * block_size, 256), block_size);
            }
            else
            {
                auto subwave_size = v.get("subwave_size", compute_subwave_size(ctx, relements));
                algo              = "subwave<" + std::to_string(subwave_size) + ">";
                options.set_launch_params(v,
                                          compute_global_for(ctx, nelements * subwave_size, 256),
                                          ctx.get_current_device().get_wavefront_size());
            }
        }
        else if(algo == "lane" or algo == "block_strided")
        {
            auto relements = reduction_shape.elements();
            optional<strided_tile> stile;
            bool few_outputs = nelements < ctx.get_current_device().get_cu_count() * 1024;
            // When a dominant input is dense along the reduction, the
            // segmented lanes read it in coalesced segments which lane cant do
            auto max_bytes =
                std::max_element(options.virtual_inputs.begin(),
                                 options.virtual_inputs.end(),
                                 by(std::less<>{}, [](const shape& s) { return s.bytes(); }))
                    ->bytes();
            bool mixed_density =
                std::any_of(options.virtual_inputs.begin(),
                            options.virtual_inputs.end() - noutputs,
                            [&](const shape& input) {
                                if(input.bytes() * 4 < max_bytes)
                                    return false;
                                return min_reduce_stride(input, reduction_shape.lens()) <= 2;
                            });
            if(assign == "assign_none" and
               (algo == "block_strided" or
                ((few_outputs or mixed_density) and not v.contains("algo"))))
                stile = find_strided_tile(ctx, relements, v.get("block_size", 256));
            if(stile.has_value())
            {
                algo         = "block_strided<" + std::to_string(stile->out_tile) + ">";
                auto ngroups = (nelements + stile->out_tile - 1) / stile->out_tile;
                options.set_launch_params(v,
                                          compute_global_for(ctx, ngroups * stile->block_size, 256),
                                          stile->block_size);
            }
            else
            {
                algo = "lane";
                options.set_launch_params(v, compute_global_for(ctx, nelements, 256));
            }
        }
        else
        {
            MIGRAPHX_THROW("Unknown reduce algo: " + algo);
        }
        options.kernel_name = v.get("kernel", "reduce_kernel");
        auto src            = interpolate_string(
            fused_reduce_kernel,
            {{"kernel", options.kernel_name},
                        {"params", enum_params(finputs.size(), "void * private_p")},
                        {"args", enum_params(finputs.size(), "private_p")},
                        {"assign", assign},
                        {"algo", algo},
                        {"reduced", "decltype(" + generate_make_shape(reduce_output_shape) + ")"},
                        {"lambda", v.at("lambda").to<std::string>()},
                        {"transformers", make_transformer_args(vec)},
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

    optional<tuning_config> get_tuning_config(const context& ctx,
                                              instruction_ref ins,
                                              const operation& op,
                                              bool exhaustive) const
    {
        if(not exhaustive)
            return nullopt;
        if(op.name() != "fused_reduce")
            return nullopt;
        tuning_config tc;
        auto shapes       = to_shapes(ins->inputs());
        tc.problem        = to_value(shapes);
        auto axes         = op.to_value().at("axes").to_vector<std::size_t>();
        auto input_shape  = get_input_shape(shapes);
        auto reduce_shape = get_reduced_shape(input_shape, axes);
        auto relements    = reduce_shape.elements();
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
        for(auto block_size : {128, 256, 1024})
            tc.solutions.push_back({{"algo", "block_strided"}, {"block_size", block_size}});
        auto finputs        = flatten(shapes);
        auto noutputs       = finputs.size() - shapes.size() + 1;
        auto virtual_inputs = finputs;
        virtual_inputs.push_back(get_reduced_shape(get_input_shape(finputs), axes));
        virtual_inputs.push_back(get_output_shape(get_input_shape(finputs), axes));
        virtual_inputs           = reduce_dims(normalize_permutation(virtual_inputs));
        auto reduce_output_shape = virtual_inputs.back();
        virtual_inputs.pop_back();
        auto reduction_shape = virtual_inputs.back();
        virtual_inputs.pop_back();
        if(find_reduce_tile(virtual_inputs, noutputs, reduce_output_shape, reduction_shape.lens())
               .has_value())
        {
            for(auto block_size : {64, 128, 256, 512})
                tc.solutions.push_back({{"algo", "block_tile"}, {"block_size", block_size}});
        }
        return tc;
    }
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
