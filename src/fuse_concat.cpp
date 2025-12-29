/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/fuse_concat.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/op/multi_gather_concat.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct fused_concat
{
    int64_t axis = 0;
    // Gather fusion attributes (optional, used when fusing gather+concat)
    bool gather_fusion     = false;
    int64_t gather_axis    = 0;
    std::size_t num_gathers = 0;

    std::string name() const { return "fused_concat"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"),
                    f(self.gather_fusion, "gather_fusion"),
                    f(self.gather_axis, "gather_axis"),
                    f(self.num_gathers, "num_gathers"));
    }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        // Gather fusion mode: inputs are [emb1, idx1, emb2, idx2, ..., output_alloc]
        if(gather_fusion)
        {
            // The output shape is the last input (the allocation)
            return inputs.back();
        }

        check_shapes{inputs, *this}.same_ndims();
        // original concat can have multiple inputs. Let's say it has `n` input args.
        // Each of those `n` input args are converted into pointwise modules that take atleast 1
        // input parameter. Fused concat will have `n+1` module arguments. `n+1`th module is the
        // post pointwise module which can take 0 or more input arguments.
        if((inputs.size() + 1) < mods.size())
            MIGRAPHX_THROW("FUSED_CONCAT: Missing fused modules inputs parameters");
        auto input_iter = inputs.begin();
        std::vector<shape> concat_inputs;
        for(module_ref mod : range(mods.begin(), mods.end() - 1))
        {
            concat_inputs.push_back(*input_iter);
            input_iter += mod->get_parameter_names().size();
        }
        module_ref post_mod          = mods.back();
        // post_mod has one input argument that is result of concat and will get generated from
        // pre-mods internally. Therefore deduct 1 from post_mod params while asserting.
        assert(input_iter + (post_mod->get_parameter_names().size() - 1) == inputs.end());
        auto type                    = std::prev(post_mod->end())->get_shape().type();
        const auto& first_shape_lens = concat_inputs.front().lens();
        auto mismatch_it =
            std::find_if_not(concat_inputs.begin() + 1, concat_inputs.end(), [&](auto s) {
                const auto& lens = s.lens();
                return std::equal(lens.begin(),
                                  lens.begin() + axis,
                                  first_shape_lens.begin(),
                                  first_shape_lens.begin() + axis) and
                       std::equal(lens.begin() + axis + 1,
                                  lens.end(),
                                  first_shape_lens.begin() + axis + 1,
                                  first_shape_lens.end());
            });
        if(mismatch_it != concat_inputs.end())
            MIGRAPHX_THROW("FUSED_CONCAT: all input dimensions should match along non-axis of " +
                           std::to_string(axis) + ": {" + to_string_range(first_shape_lens) +
                           "} != {" + to_string_range(mismatch_it->lens()) + "}");

        std::size_t new_dim_axis = transform_accumulate(
            concat_inputs.begin(), concat_inputs.end(), 0, std::plus<>{}, [&](const auto& input) {
                return input.lens()[axis];
            });
        auto new_lens  = concat_inputs.front().lens();
        new_lens[axis] = new_dim_axis;
        return shape::from_permutation(type, new_lens, find_permutation(inputs));
    }
};
MIGRAPHX_REGISTER_OP(fused_concat);

namespace {

bool is_fusable_pointwise(instruction_ref ins)
{
    return ins->name() == "pointwise" and ins->outputs().size() == 1 and
           ins->get_shape().type() != shape::tuple_type;
}

template <class... Ts>
auto fusable_pointwise(Ts... xs)
{
    return match::name("pointwise")(match::not_tuple(), xs...);
}

template <std::size_t N, std::size_t Max = 2>
struct concat_counter
{
    static_assert(N < Max, "Factor N must be less than Max");
    std::shared_ptr<unsigned int> counter = std::make_shared<unsigned int>(0);
    unsigned int get_noop_counter() const
    {
        if(counter == nullptr)
            MIGRAPHX_THROW("Invalid counter");
        return N + Max * (*counter)++;
    }
};

struct find_concat_pointwise : concat_counter<0>
{
    auto matcher() const
    {
        auto pointwise_used_once = fusable_pointwise(match::used_once());
        return match::name("concat")(match::used_once(),
                                     match::any_of[match::inputs()](pointwise_used_once));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto concat_ins = r.result;

        std::vector<instruction_ref> inputs;
        size_t num_noops = 0;
        for(auto input : concat_ins->inputs())
        {
            if(input->name() == "pointwise" and input->outputs().size() == 1)
            {
                inputs.insert(inputs.end(), input->inputs().begin(), input->inputs().end());
            }
            else
            {
                num_noops++;
                inputs.push_back(input);
            }
        }
        if(num_noops > std::max(size_t{1}, concat_ins->inputs().size() / 4))
        {
            return;
        }
        std::vector<module_ref> module_inputs;
        std::transform(concat_ins->inputs().begin(),
                       concat_ins->inputs().end(),
                       std::back_inserter(module_inputs),
                       [&](instruction_ref input) {
                           if(is_fusable_pointwise(input))
                           {
                               auto* pm = input->module_inputs().front();
                               return mpm.create_module("concat:" + pm->name(), *pm);
                           }
                           auto* pm = mpm.create_module("concat:noop" +
                                                        std::to_string(get_noop_counter()));
                           auto x   = pm->add_parameter("x0", shape{input->get_shape().type()});
                           pm->add_return({x});
                           return pm;
                       });
        auto* post_pm = mpm.create_module("noop:concat" + std::to_string(get_noop_counter()));
        auto x        = post_pm->add_parameter("!x0", shape{concat_ins->get_shape().type()});
        post_pm->add_return({x});
        module_inputs.push_back(post_pm);
        mpm.get_module().replace_instruction(
            concat_ins,
            make_op("fused_concat", concat_ins->normalized_operator().to_value()),
            inputs,
            module_inputs);
    }
};

struct find_pointwise_concat_pointwise : concat_counter<1>
{
    auto matcher() const
    {
        auto pointwise = fusable_pointwise(match::used_once());
        auto concat =
            match::name("concat")(match::used_once(), match::any_of[match::inputs()](pointwise));
        return fusable_pointwise(match::any_of[match::inputs()](concat.bind("concat")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins        = r.result;
        auto concat_ins = r.instructions["concat"];

        auto concat_arg = std::find(ins->inputs().begin(), ins->inputs().end(), concat_ins) -
                          ins->inputs().begin();
        std::vector<instruction_ref> inputs;
        for(auto input : concat_ins->inputs())
        {
            if(input->name() == "pointwise" and input->outputs().size() == 1)
                inputs.insert(inputs.end(), input->inputs().begin(), input->inputs().end());
            else
                inputs.push_back(input);
        }
        std::copy_if(ins->inputs().begin(),
                     ins->inputs().end(),
                     std::back_inserter(inputs),
                     [&](auto input) { return input != concat_ins; });

        std::vector<module_ref> module_inputs;
        std::transform(concat_ins->inputs().begin(),
                       concat_ins->inputs().end(),
                       std::back_inserter(module_inputs),
                       [&](instruction_ref input) {
                           if(is_fusable_pointwise(input))
                           {
                               auto* pm = input->module_inputs().front();
                               return mpm.create_module("concat:" + pm->name(), *pm);
                           }
                           auto* pm = mpm.create_module("concat:noop" +
                                                        std::to_string(get_noop_counter()));
                           auto x  = pm->add_parameter("x0", shape{input->get_shape().type()});
                           pm->add_return({x});
                           return pm;
                       });

        auto* post_pm                  = ins->module_inputs().front();
        auto* rm                       = mpm.create_module(post_pm->name() + ":concat", *post_pm);
        std::vector<std::string> names = rm->get_parameter_names();
        std::sort(names.begin(), names.end());
        auto concat_param_name = names[concat_arg];
        auto concat_param      = rm->get_parameter(concat_param_name);
        auto param = rm->add_parameter("!" + concat_param_name, concat_param->get_shape());
        rm->replace_instruction(concat_param, param);
        rm->remove_instruction(concat_param);

        module_inputs.push_back(rm);
        mpm.get_module().replace_instruction(
            ins,
            make_op("fused_concat", concat_ins->normalized_operator().to_value()),
            inputs,
            module_inputs);
    }
};

// Helper to check if instruction is multibroadcast(pointwise) pattern
bool is_multibroadcast_pointwise(instruction_ref ins)
{
    if(ins->name() != "multibroadcast")
        return false;
    if(ins->outputs().size() != 1)
        return false;
    auto input = ins->inputs().front();
    return is_fusable_pointwise(input) and input->outputs().size() == 1;
}

// Push pointwise through multibroadcast when it feeds into concat
// Pattern: pointwise(x) → multibroadcast → concat
// Transform to: x → multibroadcast → pointwise → concat
// Then existing find_concat_pointwise can fuse pointwise → concat
struct find_push_pointwise_through_multibroadcast
{
    auto matcher() const
    {
        // Match: concat where at least one input is multibroadcast(pointwise)
        auto pw_mb = match::name("multibroadcast")(
            match::used_once(),
            match::args(fusable_pointwise(match::used_once())));
        return match::name("concat")(match::any_of[match::inputs()](pw_mb));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto concat_ins = r.result;
        auto& m         = mpm.get_module();

        // Process ALL multibroadcast(pointwise) inputs of this concat
        for(auto input : concat_ins->inputs())
        {
            if(not is_multibroadcast_pointwise(input))
                continue;

            auto mb_ins = input;
            auto pw_ins = mb_ins->inputs().front();

            // Get the original pointwise inputs
            auto pw_inputs = pw_ins->inputs();

            // Apply multibroadcast to each pointwise input
            std::vector<instruction_ref> new_pw_inputs;
            for(auto pw_input : pw_inputs)
            {
                // Create multibroadcast for this input with the same output lens
                auto mb_op  = mb_ins->get_operator();
                auto new_mb = m.insert_instruction(mb_ins, mb_op, pw_input);
                new_pw_inputs.push_back(new_mb);
            }

            // Create new pointwise with broadcast inputs
            auto new_pw = m.insert_instruction(
                mb_ins, pw_ins->get_operator(), new_pw_inputs, pw_ins->module_inputs());

            // Replace the multibroadcast output with the new pointwise
            m.replace_instruction(mb_ins, new_pw);
        }
    }
};

// Fuse multi-source gather → slice → concat pattern
// Pattern: concat[axis=2](
//            slice[k0, axis=0](gather(emb0, idx0)),
//            slice[k1, axis=0](gather(emb1, idx1)),
//            ...
//          ) → [1, N, sum(embed_dims)]
//
// This avoids materializing all gather outputs and does everything in one kernel
struct find_multi_gather_concat
{
    auto matcher() const
    {
        // Match any concat - we'll check inputs in apply()
        return match::name("concat");
    }

    // Helper to check if an input matches slice(gather(...)) pattern
    struct source_info
    {
        instruction_ref embedding;
        instruction_ref indices;
        int64_t row_index;     // Which row to select from gather output
        int64_t embed_dim;     // Embedding dimension for this source
        instruction_ref original_input;
        std::size_t position;  // Position in original concat inputs
    };

    static bool try_match_slice_gather(instruction_ref input, std::size_t pos, source_info& out)
    {
        // Must be a slice
        if(input->name() != "slice")
            return false;

        auto slice_v = input->get_operator().to_value();
        auto axes    = slice_v.at("axes").to_vector<int64_t>();
        auto starts  = slice_v.at("starts").to_vector<int64_t>();
        auto ends    = slice_v.at("ends").to_vector<int64_t>();

        // Only handle single-axis slices on axis 0 with size 1
        if(axes.size() != 1 || axes[0] != 0)
            return false;
        if(ends[0] - starts[0] != 1)
            return false;

        auto slice_source = input->inputs().front();

        // Must be a gather
        if(slice_source->name() != "gather")
            return false;

        auto gather_inputs = slice_source->inputs();
        if(gather_inputs.size() < 2)
            return false;

        auto embedding = gather_inputs[0];
        auto indices   = gather_inputs[1];

        // Embedding must be 2D [vocab_size, embed_dim]
        if(embedding->get_shape().ndim() != 2)
            return false;

        // Get embedding dimension
        auto embed_dim = static_cast<int64_t>(embedding->get_shape().lens()[1]);

        out = {embedding, indices, starts[0], embed_dim, input, pos};
        return true;
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto concat_ins = r.result;
        auto& m         = mpm.get_module();

        auto concat_v    = concat_ins->get_operator().to_value();
        auto concat_axis = concat_v.contains("axis") ? concat_v.at("axis").to<int64_t>() : int64_t{0};

        // Normalize negative axis
        auto out_shape = concat_ins->get_shape();
        if(concat_axis < 0)
            concat_axis += static_cast<int64_t>(out_shape.ndim());

        // Only handle concat on last axis (embedding dimension)
        if(concat_axis != static_cast<int64_t>(out_shape.ndim() - 1))
            return;

        auto inputs = concat_ins->inputs();

        // Collect matching and non-matching inputs
        std::vector<source_info> matching;
        std::vector<std::pair<std::size_t, instruction_ref>> non_matching;

        for(std::size_t i = 0; i < inputs.size(); i++)
        {
            source_info info;
            if(try_match_slice_gather(inputs[i], i, info))
            {
                matching.push_back(info);
            }
            else
            {
                non_matching.push_back({i, inputs[i]});
            }
        }

        // Need at least 3 matching sources to be worth fusing
        if(matching.size() < 3)
            return;

        // All matching indices should have compatible shapes (same dims except first)
        auto first_idx_lens = matching[0].indices->get_shape().lens();
        for(const auto& src : matching)
        {
            auto idx_lens = src.indices->get_shape().lens();
            if(idx_lens.size() != first_idx_lens.size())
                return;
            for(std::size_t d = 1; d < idx_lens.size(); d++)
            {
                if(idx_lens[d] != first_idx_lens[d])
                    return;
            }
        }

        // Build metadata for the fused operation
        std::vector<int64_t> row_indices;
        std::vector<int64_t> embed_dims;
        std::vector<int64_t> col_offsets;
        int64_t current_offset = 0;
        int64_t total_fused_embed_dim = 0;

        std::vector<instruction_ref> fused_inputs;

        for(const auto& src : matching)
        {
            row_indices.push_back(src.row_index);
            embed_dims.push_back(src.embed_dim);
            col_offsets.push_back(current_offset);

            fused_inputs.push_back(src.embedding);
            fused_inputs.push_back(src.indices);

            current_offset += src.embed_dim;
            total_fused_embed_dim += src.embed_dim;
        }

        // Create the fused operation
        auto fused_op = make_op("multi_gather_concat",
                                {{"row_indices", row_indices},
                                 {"embed_dims", embed_dims},
                                 {"col_offsets", col_offsets},
                                 {"total_embed_dim", total_fused_embed_dim},
                                 {"num_sources", static_cast<int64_t>(matching.size())}});

        // Get output shape for the fused op
        // Shape: [1, seq_len, total_fused_embed_dim]
        std::vector<std::size_t> fused_shape;
        fused_shape.push_back(1);  // After slice on axis 0
        for(std::size_t d = 1; d < first_idx_lens.size(); d++)
            fused_shape.push_back(first_idx_lens[d]);
        fused_shape.push_back(static_cast<std::size_t>(total_fused_embed_dim));

        // For partial fusion, verify non-matching inputs are shape-compatible
        // All inputs must have same shape on non-concat axes
        if(!non_matching.empty())
        {
            for(const auto& [pos, inp] : non_matching)
            {
                auto inp_shape = inp->get_shape().lens();
                // Must have same number of dimensions
                if(inp_shape.size() != fused_shape.size())
                    return;
                // Check all axes except concat axis
                for(std::size_t d = 0; d < inp_shape.size(); d++)
                {
                    if(d == static_cast<std::size_t>(concat_axis))
                        continue;
                    if(inp_shape[d] != fused_shape[d])
                        return;
                }
            }
        }

        if(non_matching.empty())
        {
            // All inputs matched - simple replacement
            m.replace_instruction(concat_ins, fused_op, fused_inputs);
        }
        else
        {
            // Partial fusion: create fused op, then concat with non-matching inputs
            // Insert fused instruction before concat
            auto fused_ins = m.insert_instruction(concat_ins, fused_op, fused_inputs);

            // Build new concat inputs: replace all matching inputs with fused result
            // Position the fused result at the first matching input's position
            std::vector<instruction_ref> new_concat_inputs;
            std::size_t first_match_pos = matching[0].position;
            bool fused_inserted = false;

            for(std::size_t i = 0; i < inputs.size(); i++)
            {
                // Check if this position had a matching input
                bool is_matching = false;
                for(const auto& m_src : matching)
                {
                    if(m_src.position == i)
                    {
                        is_matching = true;
                        break;
                    }
                }

                if(is_matching)
                {
                    // Insert fused result at first matching position only
                    if(!fused_inserted)
                    {
                        new_concat_inputs.push_back(fused_ins);
                        fused_inserted = true;
                    }
                    // Skip other matching inputs (they're now in fused_ins)
                }
                else
                {
                    new_concat_inputs.push_back(inputs[i]);
                }
            }

            // Replace concat with new concat that includes fused result
            auto new_concat = make_op("concat", {{"axis", concat_axis}});
            m.replace_instruction(concat_ins, new_concat, new_concat_inputs);
        }
    }
};

} // namespace

void fuse_concat::apply(module_pass_manager& mpm) const
{
    // First, push pointwise through multibroadcast when feeding concat
    // This transforms: pointwise → multibroadcast → concat
    // Into: multibroadcast → pointwise → concat
    // So that find_concat_pointwise can fuse pointwise into concat
    match::find_matches(mpm, find_push_pointwise_through_multibroadcast{});
    mpm.run_pass(migraphx::dead_code_elimination{});

    // Fuse multi-source gather → slice → concat pattern
    match::find_matches(mpm, find_multi_gather_concat{});
    mpm.run_pass(migraphx::dead_code_elimination{});

    match::find_matches(mpm, find_pointwise_concat_pointwise{});
    mpm.run_pass(migraphx::dead_code_elimination{});
    match::find_matches(mpm, find_concat_pointwise{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
