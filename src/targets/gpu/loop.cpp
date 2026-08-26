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
#include <cstdint>
#include <migraphx/instruction.hpp>
#include <migraphx/run_loop.hpp>
#include <migraphx/gpu/loop.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/hip.hpp>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_loop::compute_shape(std::vector<shape> inputs, std::vector<module_ref> mods) const
{
    auto input_num = (inputs.size() - 2) / 2;
    inputs.erase(inputs.begin() + input_num, inputs.end());
    return op.compute_shape(inputs, std::move(mods));
}

struct gpu_loop
{
    int64_t max_iterations = 0;

    template <class T>
    void copy(context& ctx, const argument& src, T& dst) const
    {
        argument arg_dst{src.get_shape(), &dst};
        copy_from_gpu(ctx, src, arg_dst);
    }

    template <class T>
    void copy(context& ctx, T src, const argument& dst) const
    {
        argument arg_src{dst.get_shape(), &src};
        copy_to_gpu(ctx, arg_src, dst);
    }

    void append(context& ctx,
                const std::vector<argument>& iter_state,
                const std::vector<argument>& concatenated_outputs,
                const std::vector<int64_t>& scan_output_dirs,
                int64_t curr_iter,
                int64_t num_iters) const
    {
        assert(iter_state.size() == concatenated_outputs.size());
        migraphx::for_each(
            iter_state.begin(),
            iter_state.end(),
            concatenated_outputs.begin(),
            [&, index = std::size_t{0}](const auto& src, const auto& dst) mutable {
                auto dir          = scan_output_dirs.empty() ? 0 : scan_output_dirs[index];
                auto output_index = (1 - dir) * curr_iter + dir * (num_iters - 1 - curr_iter);
                auto output_size  = src.get_shape().bytes();
                assert((output_index + 1) * output_size <= dst.get_shape().bytes());
                argument output{src.get_shape(), dst.data() + output_index * output_size};
                if(src.data() != output.data())
                    gpu_copy(ctx, src, output);
                index++;
            });
    }

    void set_zero(context& ctx, const std::vector<argument>& concatenated_outputs, int iter) const
    {
        if(iter >= max_iterations)
            return;

        auto elem_num = max_iterations - iter;
        for(const auto& out : concatenated_outputs)
        {
            auto s    = out.get_shape();
            auto size = s.bytes() / max_iterations;
            auto lens = s.lens();
            lens[0]   = elem_num;
            shape ss{s.type(), lens};
            assert(ss.bytes() + iter * size <= out.get_shape().bytes());
            gpu_fill(ctx, argument(ss, out.data() + iter * size), 0);
        }
    }

    std::unordered_map<std::string, std::vector<std::size_t>>
    get_output_params(const module& m) const
    {
        auto param_names = m.get_parameter_names();
        std::vector<std::string> output_names;
        std::copy_if(param_names.begin(),
                     param_names.end(),
                     std::back_inserter(output_names),
                     [](const auto& name) { return contains(name, "#output_"); });

        auto parameters = m.get_parameters();
        auto returns    = m.get_returns();
        auto indices    = range(returns.size());
        std::unordered_map<std::string, std::vector<std::size_t>> result;
        std::transform(
            output_names.begin(),
            output_names.end(),
            std::inserter(result, result.end()),
            [&](const auto& name) {
                auto parameter =
                    std::find_if(parameters.begin(), parameters.end(), [&](instruction_ref ins) {
                        return any_cast<builtin::param>(ins->get_operator()).parameter == name;
                    });
                assert(parameter != parameters.end());

                std::vector<std::pair<std::size_t, std::size_t>> output_positions;
                transform_if(
                    indices.begin(),
                    indices.end(),
                    std::back_inserter(output_positions),
                    [&](std::size_t index) {
                        auto aliases = instruction::get_output_alias(returns[index]);
                        return contains(aliases, *parameter);
                    },
                    [&](std::size_t index) {
                        auto tuple_index = std::size_t{0};
                        if((*parameter)->get_shape().type() == shape::tuple_type)
                        {
                            auto output = returns[index];
                            if(output->name() != "get_tuple_elem")
                                MIGRAPHX_THROW("GPU_LOOP: tuple output parameter \"" + name +
                                               "\" does not map to tuple element returns");
                            tuple_index =
                                output->get_operator().to_value().at("index").to<std::size_t>();
                        }
                        return std::make_pair(tuple_index, index);
                    });
                std::sort(output_positions.begin(),
                          output_positions.end(),
                          [](const auto& x, const auto& y) {
                              return std::tie(x.first, y.second) < std::tie(y.first, x.second);
                          });
                auto unique_end =
                    std::unique(output_positions.begin(),
                                output_positions.end(),
                                [](const auto& x, const auto& y) { return x.first == y.first; });
                std::vector<std::size_t> output_indices;
                std::transform(output_positions.begin(),
                               unique_end,
                               std::back_inserter(output_indices),
                               [](const auto& item) { return item.second; });
                return std::make_pair(name, std::move(output_indices));
            });
        return result;
    }
};

argument
hip_loop::compute(context& ctx,
                  const shape&,
                  const std::vector<argument>& args,
                  const std::vector<module_ref>& mods,
                  const std::function<std::vector<argument>(
                      module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
{
    return run_loop(gpu_loop{op.max_iterations}, op.scan_output_directions, ctx, args, mods, run);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
