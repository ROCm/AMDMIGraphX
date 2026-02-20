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

#include <migraphx/rewrite_kv_cache.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/builtin.hpp>
#include <migraphx/op/concat.hpp>
#include <regex>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

// Check if a parameter name matches the KV-cache pattern
// Matches: past.0.key, past.0.value, past_key_values.0.key, etc.
bool is_past_kv_param(const std::string& name)
{
    // Match patterns like:
    // - past.{N}.key / past.{N}.value
    // - past_{N}_key / past_{N}_value  
    // - past_key_values.{N}.key / past_key_values.{N}.value
    static const std::regex past_pattern(
        R"(past[._](?:key_values[._])?(\d+)[._](key|value))",
        std::regex::icase);
    return std::regex_search(name, past_pattern);
}


/**
 * Matcher for standard KV-cache concat patterns.
 * 
 * Matches: concat(past_kv, new_kv, axis=2) where past_kv is a graph parameter
 * with a name matching the KV-cache naming convention.
 */
struct find_kv_cache_concat
{
    // Cache the seqlens_k parameter reference once found
    mutable instruction_ref seqlens_k_param;
    mutable bool seqlens_k_searched = false;

    auto matcher() const
    {
        // Match any concat operation - we'll filter in apply()
        return match::name("concat");
    }

    // Find or create the seqlens_k parameter
    instruction_ref find_seqlens_k(module& m) const
    {
        if(seqlens_k_searched)
            return seqlens_k_param;

        seqlens_k_searched = true;
        seqlens_k_param    = m.end();

        // Search for existing seqlens_k parameter
        for(auto ins : iterator_for(m))
        {
            if(ins->name() != "@param")
                continue;

            auto p    = any_cast<builtin::param>(ins->get_operator());
            auto name = p.parameter;

            // Check for common seqlens_k naming patterns
            if(name == "seqlens_k" || name == "sequence_lengths" || name == "past_sequence_length" ||
               name.find("seqlen") != std::string::npos)
            {
                seqlens_k_param = ins;
                return seqlens_k_param;
            }
        }

        return seqlens_k_param;
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto concat_ins = r.result;

        // Get concat axis
        auto concat_op = any_cast<op::concat>(concat_ins->get_operator());
        auto axis      = concat_op.axis;

        // Get the shape to determine actual axis (handle negative)
        auto out_shape = concat_ins->get_shape();
        auto ndim      = out_shape.ndim();
        if(axis < 0)
            axis += static_cast<int64_t>(ndim);

        // KV-cache concat should be on axis 2 (sequence dimension)
        // Shape is typically: (batch, num_heads, seq_len, head_dim)
        if(axis != 2)
            return;

        // Check inputs - need exactly 2 for this pattern
        auto inputs = concat_ins->inputs();
        if(inputs.size() != 2)
            return;

        // Find which input is the past_kv parameter
        instruction_ref past_kv  = m.end();
        instruction_ref new_kv   = m.end();
        std::string past_kv_name;

        for(auto inp : inputs)
        {
            if(inp->name() == "@param")
            {
                auto p    = any_cast<builtin::param>(inp->get_operator());
                auto name = p.parameter;

                if(is_past_kv_param(name))
                {
                    past_kv      = inp;
                    past_kv_name = name;
                }
            }
        }

        // If no past_kv parameter found, this isn't a KV-cache concat
        if(past_kv == m.end())
            return;

        // The other input is new_kv
        for(auto inp : inputs)
        {
            if(inp != past_kv)
            {
                new_kv = inp;
                break;
            }
        }

        if(new_kv == m.end())
            return;

        // Find seqlens_k parameter - this is REQUIRED for concat_past_present
        // The model must have been exported with seqlens_k as an input parameter
        // If not found, we cannot safely apply this optimization
        auto seqlens_k = find_seqlens_k(m);
        if(seqlens_k == m.end())
        {
            // seqlens_k not found - cannot apply optimization
            // The model needs to be re-exported with seqlens_k as an input
            // to enable KV-cache optimization
            return;
        }

        // Get number of KV heads from shape
        // Shape: (batch, num_heads, seq_len, head_dim)
        auto past_shape   = past_kv->get_shape();
        auto kv_num_heads = past_shape.lens()[1];

        // Create the concat_past_present operation
        // Inputs: (present/new_kv, seqlens_k, past_kv)
        // Note: concat_past_present returns the past buffer (in-place update)
        auto cpp_op = make_op("concat_past_present", {{"kv_num_heads", kv_num_heads}});

        // Replace concat with concat_past_present
        // The order is: (new_kv, seqlens_k, past_kv) based on the operator definition
        m.replace_instruction(concat_ins, cpp_op, {new_kv, seqlens_k, past_kv});
    }
};

} // namespace

void rewrite_kv_cache::apply(module& m) const { match::find_matches(m, find_kv_cache_concat{}); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
