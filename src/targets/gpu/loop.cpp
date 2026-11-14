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

    // The body writes each scan output to its slot through an output parameter, unless it
    // returns another buffer, such as gpu::fixed_pad returning its input when there is nothing
    // to pad. Copy such a result into the slot.
    void append(context& ctx,
                const std::vector<argument>& iter_state,
                const std::vector<argument>& concatenated_outputs,
                const std::vector<int64_t>& scan_output_dirs,
                int64_t curr_iter,
                int64_t num_iters) const
    {
        assert(iter_state.size() == concatenated_outputs.size());
        for(std::size_t i = 0; i < iter_state.size(); ++i)
        {
            const auto& iter_stat = iter_state.at(i);
            auto dir              = scan_output_dirs.empty() ? 0 : scan_output_dirs[i];
            auto idx              = (1 - dir) * curr_iter + dir * (num_iters - 1 - curr_iter);
            auto size             = iter_stat.get_shape().bytes();
            auto* slot            = concatenated_outputs.at(i).data() + idx * size;
            if(iter_stat.data() == slot)
                continue;
            gpu_copy(ctx, iter_stat, argument{iter_stat.get_shape(), slot});
        }
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

    std::unordered_map<std::string, int> get_output_params(const module& m) const
    {
        auto get_output_index = [](const std::string& name) {
            std::string out_prefix = "#output_";
            auto loc               = name.find(out_prefix);
            if(loc != std::string::npos)
            {
                return std::stoi(name.substr(loc + out_prefix.size()));
            }

            return -1;
        };

        const auto& param_names = m.get_parameter_names();
        std::unordered_map<std::string, int> result;
        for(const auto& name : param_names)
        {
            auto index = get_output_index(name);
            if(index == -1)
                continue;
            result[name] = index;
        }

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
