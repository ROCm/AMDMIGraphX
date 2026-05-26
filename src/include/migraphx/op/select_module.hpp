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
#include <migraphx/env.hpp>
#include <migraphx/module.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DYN_DIM_BUCKET_BY_OPTIMALS)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_SELECT_MODULE_CACHE)

// Hash a sequence of argument shapes into a single 64-bit token. We hash
// type + lens (strides intentionally excluded so callers can pass strided
// views and still hit the cache). This is the dispatch-cache key.
inline std::size_t hash_arg_shapes(const std::vector<argument>& args)
{
    // NOLINTBEGIN(hicpp-signed-bitwise)
    std::uint64_t h               = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t prime = 0x100000001b3ULL;
    for(const auto& a : args)
    {
        const auto& s = a.get_shape();
        h ^= static_cast<std::uint64_t>(static_cast<unsigned int>(s.type()));
        h *= prime;
        for(auto l : s.lens())
        {
            h ^= static_cast<std::uint64_t>(l) + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
        }
    }
    return static_cast<std::size_t>(h);
    // NOLINTEND(hicpp-signed-bitwise)
}

// Smallest compatible bucket lookup used when bucket dispatch is enabled.
// `get_input_parameter_names_fn` returns the submodule's input parameter
// names in the same (sorted) order as the corresponding args.  A submodule
// is "compatible" with the runtime args when every input parameter's static
// shape has the same element type and rank as the corresponding arg and
// each of its lens is >= the arg's lens componentwise.  Among compatible
// submodules, the one with the smallest total static element count wins
// (smallest bucket large enough to hold the input).  Returns end() if no
// compatible submodule exists.
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
            const auto& a  = args[i];
            const auto& ps = param_shapes.at(in_param_names[i]);
            if(ps.type() != a.get_shape().type() or ps.lens().size() != a.get_shape().lens().size())
            {
                compatible = false;
                break;
            }
            for(std::size_t d = 0; d < ps.lens().size(); ++d)
            {
                if(ps.lens()[d] < a.get_shape().lens()[d])
                {
                    compatible = false;
                    break;
                }
                score *= ps.lens()[d];
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

    // Per-instance fast-path cache for compute().  Without this, every
    // inference re-runs the full dispatch search and rebuilds the input
    // parameter-name / parameter-shape lookup vectors -- pure host work
    // before the kernel launches.  For a hot loop that keeps feeding the
    // same input shape, we hash the input shapes once, cache the chosen
    // submodule pointer + its sorted input/output names + their static
    // shapes, and on a hit jump straight to running the submodule.
    //
    // The cache lives in `mutable` state because `compute()` is a const
    // method on the operator.  This is single-threaded by design; the
    // MIGraphX runtime currently never invokes `compute()` for the same
    // operator instance concurrently, so no lock is needed.
    struct dispatch_cache_t
    {
        bool valid               = false;
        std::size_t shape_hash   = 0;
        std::size_t module_index = static_cast<std::size_t>(-1);
        std::vector<std::string> in_names;
        std::vector<std::string> out_names;
        std::vector<shape> in_param_shapes; // ordered, indexed by i
        std::unordered_map<std::string, shape> out_param_shapes;
        bool needs_input_pad = false; // bucket-dispatch input shape != bucket shape
    };
    mutable dispatch_cache_t dispatch_cache;

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
        // Runtime kill-switch for the cache (mainly for A/B benchmarks
        // and emergency rollback).  Defaults to OFF (cache enabled).
        // Cached statically on first call to avoid the env lookup cost
        // on the hot path.  When set, the entire cache infrastructure
        // is bypassed -- no read, no populate -- so the measured cost
        // matches the pre-cache implementation.
        // cppcheck-suppress migraphx-UseCachedEnvVar
        // Uncached so tests in the same binary that toggle this env
        // var (kill-switch coverage tests) can actually flip the path.
        const bool cache_disabled = enabled("MIGRAPHX_DISABLE_SELECT_MODULE_CACHE");
        if(cache_disabled)
            return compute_legacy(args, submodule_list, run);

        // When there are very few submodules to scan, the slow path's
        // find_if + a handful of get_parameter_names/get_parameter_shapes
        // allocations cost less than the cache's hash + populate +
        // managed-vector lookups on every call. Empirically (MI308X
        // 10000-iter median of 5) the cache fast-path starts paying off
        // around 4+ submodules.  Below that threshold the cache is
        // either neutral (lost in noise) or slightly worse, so we just
        // use the legacy code path.  The freeze path has 0 submodules
        // and bypasses select_module entirely, so this threshold only
        // affects bucket-dispatch users with a small number of optimals.
        static constexpr std::size_t cache_min_submodules = 4;
        if(submodule_list.size() < cache_min_submodules)
            return compute_legacy(args, submodule_list, run);

        // ---- Fast path: dispatch cache hit ----
        // If we already chose a submodule for this exact input-shape
        // signature on a previous call, skip the entire find_if scan,
        // the name-vector + param_shapes_map allocations, and reuse the
        // cached lookup data to build p_map directly.
        const auto h = hash_arg_shapes(args);
        if(dispatch_cache.valid and dispatch_cache.shape_hash == h and
           dispatch_cache.module_index < submodule_list.size())
        {
            return compute_with_cache(args, submodule_list, run);
        }

        // ---- Slow path: full dispatch + populate the cache ----
        // 1) Try exact-shape match.
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

        // 2) Smallest-bucket fallback if bucket dispatch is enabled.
        bool bucket_dispatch = false;
        // cppcheck-suppress migraphx-UseCachedEnvVar
        // Uncached so the bucket-dispatch behaviour responds to env
        // toggles between tests within the same binary.  The compute()
        // hot path is gated on first-hit anyway.
        if(module_iter == submodule_list.end() and enabled("MIGRAPHX_DYN_DIM_BUCKET_BY_OPTIMALS"))
        {
            module_iter =
                find_smallest_compatible_submodule(submodule_list, args, [this](module_ref mr) {
                    return this->get_input_parameter_names(mr);
                });
            bucket_dispatch = (module_iter != submodule_list.end());
        }

        if(module_iter == submodule_list.end())
        {
            MIGRAPHX_THROW("SELECT_MODULE: no compatible submodules found for given input shapes");
        }

        auto* module_to_run   = *module_iter;
        auto in_param_names   = get_input_parameter_names(module_to_run);
        auto out_param_names  = get_output_parameter_names(module_to_run);
        auto param_shapes_map = module_to_run->get_parameter_shapes();

        // Populate the dispatch cache so the next call with the same input
        // shape hash hits the fast path. We only cache when the result is
        // valid (not in the error path).
        dispatch_cache.shape_hash = h;
        dispatch_cache.module_index =
            static_cast<std::size_t>(std::distance(submodule_list.cbegin(), module_iter));
        dispatch_cache.in_names  = in_param_names;
        dispatch_cache.out_names = out_param_names;
        dispatch_cache.in_param_shapes.clear();
        dispatch_cache.in_param_shapes.reserve(in_param_names.size());
        for(const auto& name : in_param_names)
            dispatch_cache.in_param_shapes.push_back(param_shapes_map.at(name));
        dispatch_cache.out_param_shapes = std::move(param_shapes_map);
        dispatch_cache.needs_input_pad  = bucket_dispatch;
        dispatch_cache.valid            = true;

        return compute_with_cache(args, submodule_list, run);
    }

    // Pre-cache implementation kept verbatim so MIGRAPHX_DISABLE_SELECT_MODULE_CACHE
    // measures a true A/B baseline.  Identical to compute() at HEAD~ before
    // the dispatch cache was introduced -- linear find_if over submodules,
    // fresh get_input/output_parameter_names allocations per call, fresh
    // get_parameter_shapes map per call.
    argument
    compute_legacy(const std::vector<argument>& args,
                   const std::vector<module_ref>& submodule_list,
                   const std::function<std::vector<argument>(
                       module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
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

        bool bucket_dispatch = false;
        // cppcheck-suppress migraphx-UseCachedEnvVar
        // Uncached so the bucket-dispatch behaviour responds to env
        // toggles between tests within the same binary.  The compute()
        // hot path is gated on first-hit anyway.
        if(module_iter == submodule_list.end() and enabled("MIGRAPHX_DYN_DIM_BUCKET_BY_OPTIMALS"))
        {
            module_iter =
                find_smallest_compatible_submodule(submodule_list, args, [this](module_ref mr) {
                    return this->get_input_parameter_names(mr);
                });
            bucket_dispatch = (module_iter != submodule_list.end());
        }

        if(module_iter == submodule_list.end())
        {
            MIGRAPHX_THROW("SELECT_MODULE: no compatible submodules found for given input shapes");
        }

        auto* module_to_run = *module_iter;
        std::unordered_map<std::string, argument> p_map;
        auto in_param_names = get_input_parameter_names(module_to_run);
        assert(in_param_names.size() <= args.size());
        if(bucket_dispatch)
        {
            auto param_shapes = module_to_run->get_parameter_shapes();
            for(std::size_t i = 0; i < in_param_names.size(); ++i)
            {
                const auto& name = in_param_names[i];
                const auto& a    = args[i];
                const auto& ps   = param_shapes.at(name);
                if(a.get_shape() == ps)
                {
                    p_map.emplace(name, a);
                }
                else
                {
                    argument padded{ps};
                    std::memset(padded.data(), 0, ps.bytes());
                    std::memcpy(padded.data(), a.data(), a.get_shape().bytes());
                    p_map.emplace(name, std::move(padded));
                }
            }
        }
        else
        {
            std::transform(in_param_names.begin(),
                           in_param_names.end(),
                           args.begin(),
                           std::inserter(p_map, p_map.end()),
                           [&](auto&& name, auto&& a) { return std::make_pair(name, a); });
        }

        auto out_param_names    = get_output_parameter_names(module_to_run);
        auto param_shapes       = module_to_run->get_parameter_shapes();
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
                           return std::make_pair(name, a);
                       });
        auto results = run(module_to_run, p_map);
        return argument{results};
    }

    // Run the cached dispatch decision. Pulled out of compute() so both
    // the hit-path and the miss-path-after-population go through the same
    // code, which makes it easier to keep the two in sync.
    argument compute_with_cache(
        const std::vector<argument>& args,
        const std::vector<module_ref>& submodule_list,
        const std::function<std::vector<argument>(
            module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        auto* module_to_run = submodule_list[dispatch_cache.module_index];
        std::unordered_map<std::string, argument> p_map;
        p_map.reserve(dispatch_cache.in_names.size() + dispatch_cache.out_names.size());

        // Inputs. If bucket dispatch chose a larger bucket than the input,
        // pad on the host (see comment above; GPU users pre-pad on the
        // caller side).
        if(dispatch_cache.needs_input_pad)
        {
            for(std::size_t i = 0; i < dispatch_cache.in_names.size(); ++i)
            {
                const auto& name = dispatch_cache.in_names[i];
                const auto& a    = args[i];
                const auto& ps   = dispatch_cache.in_param_shapes[i];
                if(a.get_shape() == ps)
                {
                    p_map.emplace(name, a);
                }
                else
                {
                    argument padded{ps};
                    std::memset(padded.data(), 0, ps.bytes());
                    std::memcpy(padded.data(), a.data(), a.get_shape().bytes());
                    p_map.emplace(name, std::move(padded));
                }
            }
        }
        else
        {
            for(std::size_t i = 0; i < dispatch_cache.in_names.size(); ++i)
                p_map.emplace(dispatch_cache.in_names[i], args[i]);
        }

        // Outputs: reshape the pre-allocated buffer (last arg, a tuple of
        // sub-objects) to the submodule's static output shapes when sizes
        // differ. This was already handled by the previous implementation;
        // we replicate the same logic via the cached output names/shapes.
        auto output_sub_objects = args.back().get_sub_objects();
        assert(dispatch_cache.out_names.size() == output_sub_objects.size());
        for(std::size_t i = 0; i < dispatch_cache.out_names.size(); ++i)
        {
            const auto& name = dispatch_cache.out_names[i];
            const auto& a    = output_sub_objects[i];
            const auto& ps   = dispatch_cache.out_param_shapes.at(name);
            if(a.get_shape() != ps)
            {
                assert(ps.bytes() <= a.get_shape().bytes());
                p_map.emplace(name, a.reshape(ps));
            }
            else
            {
                p_map.emplace(name, a);
            }
        }

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
