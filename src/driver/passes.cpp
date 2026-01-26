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

#include "passes.hpp"

#include <migraphx/auto_contiguous.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_allocation.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/eliminate_concat.hpp>
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/eliminate_data_type.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/eliminate_pad.hpp>
#include <migraphx/fuse_pointwise.hpp>
#include <migraphx/fuse_reduce.hpp>
#include <migraphx/inline_module.hpp>
#include <migraphx/insert_pad.hpp>
#include <migraphx/normalize_ops.hpp>
#include <migraphx/optimize_module.hpp>
#include <migraphx/promote_literals.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/rewrite_dot.hpp>
#include <migraphx/rewrite_gelu.hpp>
#include <migraphx/rewrite_pooling.hpp>
#include <migraphx/rewrite_quantization.hpp>
#include <migraphx/rewrite_rnn.hpp>
#include <migraphx/simplify_algebra.hpp>
#include <migraphx/simplify_dyn_ops.hpp>
#include <migraphx/simplify_qdq.hpp>
#include <migraphx/simplify_reshapes.hpp>

#include <migraphx/ranges.hpp>
#include <unordered_map>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

struct pass_with_context : pass
{
    pass_with_context(const pass& p, std::shared_ptr<context> pctx = nullptr) : pass(p), ctx(pctx)
    {
    }
    std::shared_ptr<context> ctx = nullptr;
};

static std::unordered_map<std::string, pass> create_passes_lookup()
{
    std::unordered_map<std::string, pass> result;
    // clang-format off
    std::initializer_list<pass> passes = {
        auto_contiguous{},
        dead_code_elimination{},
        eliminate_allocation{},
        eliminate_common_subexpression{},
        eliminate_concat{},
        eliminate_contiguous{},
        eliminate_data_type{},
        eliminate_identity{},
        eliminate_pad{},
        fuse_pointwise{},
        fuse_reduce{},
        inline_module{},
        insert_pad{},
        normalize_ops{},
        optimize_module{},
        promote_literals{},
        propagate_constant{},
        rewrite_dot{},
        rewrite_gelu{},
        rewrite_pooling{},
        rewrite_quantization{},
        rewrite_rnn{},
        simplify_algebra{},
        simplify_dyn_ops{},
        simplify_qdq{},
        simplify_reshapes{},
    };
    // clang-format on
    for(const auto& pass : passes)
        result[pass.name()] = pass;
    result["eliminate_dead_code"] = dead_code_elimination{};
    return result;
}

std::optional<pass> get_pass(const std::string& name)
{
    static const std::unordered_map<std::string, pass> lookup = create_passes_lookup();
    if(contains(lookup, name))
        return lookup.at(name);
    auto fields = split_string(name, '@');
    if(fields.size() != 2)
        return std::nullopt;
    auto base_name   = fields[0];
    auto target_name = fields[1];
    auto t           = make_target(target_name);
    auto ctx         = std::make_shared<context>(t.get_context());
    auto passes      = t.get_passes(*ctx, {});
    auto it          = std::find_if(
        passes.begin(), passes.end(), [&](const pass& p) { return p.name() == base_name; });
    if(it == passes.end())
        return std::nullopt;
    return pass_with_context(*it, ctx);
}

std::vector<pass> get_passes(const std::vector<std::string>& names)
{
    std::vector<pass> result;
    // static const std::unordered_map<std::string, pass> lookup = create_passes_lookup();
    std::transform(
        names.begin(), names.end(), std::back_inserter(result), [](const std::string& name) {
            auto p = get_pass(name);
            if(not p)
                MIGRAPHX_THROW("Unknown pass: " + name);
            return *p;
        });
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
