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
#ifndef MIGRAPHX_GUARD_OPERATORS_SELECT_MODULE_HPP
#define MIGRAPHX_GUARD_OPERATORS_SELECT_MODULE_HPP

#include <migraphx/argument.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/module.hpp>
#include <cstddef>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

// Smallest-bucket fallback for runtime dispatch.
//   compatible: same type+rank, and the input lens fit inside the submodule's
//               input parameter bounds (for a bucket submodule the parameter is
//               dynamic and fixed_pad expands it up to max_lens; for a legacy
//               submodule the parameter is static so the bounds collapse to the
//               static lens, i.e. an exact match).
//   winner:     compatible submodule with the smallest total (max) element
//               count, i.e. the least padding.
// Returns end() if no compatible submodule exists.
inline std::vector<module_ref>::const_iterator find_smallest_compatible_submodule(
    const std::vector<module_ref>& submodule_list,
    const std::vector<argument>& args,
    const std::function<std::vector<std::string>(module_ref)>& get_input_parameter_names_fn)
{
    auto best              = submodule_list.cend();
    std::size_t best_score = 0;
    for(auto it = submodule_list.cbegin(); it != submodule_list.cend(); ++it)
    {
        auto in_param_names = get_input_parameter_names_fn(*it);
        if(in_param_names.size() > args.size())
            continue;
        auto param_shapes = (*it)->get_parameter_shapes();
        bool compatible   = true;
        std::size_t score = 1;
        for(std::size_t i = 0; i < in_param_names.size(); ++i)
        {
            const auto& a      = args[i];
            const auto& ps     = param_shapes.at(in_param_names[i]);
            const auto& a_lens = a.get_shape().lens();
            // Dynamic (bucket) params carry a [min_lens, max_lens] range;
            // static (legacy) params collapse both bounds to the static lens.
            const auto lo = ps.dynamic() ? ps.min_lens() : ps.lens();
            const auto hi = ps.dynamic() ? ps.max_lens() : ps.lens();
            if(ps.type() != a.get_shape().type() or hi.size() != a_lens.size())
            {
                compatible = false;
                break;
            }
            for(std::size_t d = 0; d < hi.size(); ++d)
            {
                if(a_lens[d] < lo[d] or a_lens[d] > hi[d])
                {
                    compatible = false;
                    break;
                }
                score *= hi[d];
            }
            if(not compatible)
                break;
        }
        if(compatible and (best == submodule_list.cend() or score < best_score))
        {
            best       = it;
            best_score = score;
        }
    }
    return best;
}

struct select_module
{
    shape output_dyn_shapes;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_dyn_shapes, "output_dyn_shapes"));
    }

    std::string name() const { return "select_module"; }

    shape compute_shape(const std::vector<shape>& inputs, const std::vector<module_ref>&) const
    {
        check_shapes{inputs, *this, true}.has_at_least(1);
        return shape{output_dyn_shapes};
    }

    std::vector<std::string> get_input_parameter_names(module_ref mod) const
    {
        auto param_names = mod->get_parameter_names();
        std::vector<std::string> ret;
        std::copy_if(param_names.cbegin(),
                     param_names.cend(),
                     std::back_inserter(ret),
                     [](const auto& pn) { return not contains(pn, "#output_"); });
        std::sort(ret.begin(), ret.end());
        return ret;
    }

    std::vector<std::string> get_output_parameter_names(module_ref mod) const
    {
        auto param_names = mod->get_parameter_names();
        std::vector<std::string> ret;
        std::copy_if(param_names.cbegin(),
                     param_names.cend(),
                     std::back_inserter(ret),
                     [](const auto& pn) { return contains(pn, "#output_"); });
        // needs to be sorted to ensure output parameter ordering
        std::sort(ret.begin(), ret.end());
        return ret;
    }

    argument compute(const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& submodule_list,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        // Find submodule with input parameter shapes exactly the same as the input instruction
        // arguments. Assuming instruction arguments are in the same order as the instruction
        // parameters.
        auto module_iter =
            std::find_if(submodule_list.cbegin(), submodule_list.cend(), [&](module_ref mr) {
                auto in_param_names = get_input_parameter_names(mr);
                auto param_shapes   = mr->get_parameter_shapes();
                assert(in_param_names.size() <= args.size());
                return std::equal(in_param_names.cbegin(),
                                  in_param_names.cend(),
                                  args.cbegin(),
                                  [&](const auto& p_name, const auto& a) {
                                      return a.get_shape() == param_shapes[p_name];
                                  });
            });

        // Smallest-compatible-bucket fallback (bucket_by_optimals mode). Whether
        // any buckets exist is a compile-time decision in split_single_dyn_dim.
        // Bucket submodules take a dynamic input and pad it up to the bucket
        // size in-graph via fixed_pad, so this dispatch is device-agnostic and
        // does not touch argument buffers on the host.
        if(module_iter == submodule_list.end())
        {
            module_iter =
                find_smallest_compatible_submodule(submodule_list, args, [this](module_ref mr) {
                    return this->get_input_parameter_names(mr);
                });
        }

        if(module_iter == submodule_list.end())
        {
            MIGRAPHX_THROW("SELECT_MODULE: no compatible submodules found for given input shapes");
        }

        auto* module_to_run = *module_iter;
        std::unordered_map<std::string, argument> p_map;

        // Input parameters. Forward the runtime argument unchanged: an exact
        // match already fits the (static) parameter, and a bucket submodule
        // pads the input up to its static bucket size in-graph via fixed_pad.
        auto in_param_names = get_input_parameter_names(module_to_run);
        auto param_shapes   = module_to_run->get_parameter_shapes();
        assert(in_param_names.size() <= args.size());
        for(std::size_t i = 0; i < in_param_names.size(); ++i)
        {
            p_map.emplace(in_param_names[i], args[i]);
        }

        // One tuple output parameter in main module to multiple output parameters in submodule
        auto out_param_names    = get_output_parameter_names(module_to_run);
        auto output_sub_objects = args.back().get_sub_objects();
        assert(out_param_names.size() == output_sub_objects.size());
        std::transform(out_param_names.begin(),
                       out_param_names.end(),
                       output_sub_objects.begin(),
                       std::inserter(p_map, p_map.end()),
                       [&](auto&& name, auto&& a) {
                           auto ps = param_shapes.at(name);
                           if(a.get_shape() != ps)
                           {
                               assert(ps.bytes() <= a.get_shape().bytes());
                               return std::make_pair(name, a.reshape(ps));
                           }
                           else
                           {
                               return std::make_pair(name, a);
                           }
                       });
        auto results = run(module_to_run, p_map);
        return argument{results};
    }

    std::vector<std::size_t> output_alias(const std::vector<shape>& shapes) const
    {
        return {shapes.size() - 1};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
