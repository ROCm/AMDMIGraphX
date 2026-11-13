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
#include <migraphx/gpu/record_compiled_for.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/builtin.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/json.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/stringutils.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

value compile_options_value(const compile_options& options, const value& backend_options)
{
    return {{"offload_copy", options.offload_copy},
            {"fast_math", options.fast_math},
            {"exhaustive_tune", options.exhaustive_tune},
            {"recompilable", options.recompilable},
            {"compile_mode", to_string(options.compile_mode)},
            {"backend_options", backend_options}};
}

static instruction_ref find_tagged_comment(const module& m, const std::string& tag)
{
    return std::find_if(m.begin(), m.end(), [&](const instruction& ins) {
        return ins.name() == "@comment" and
               any_cast<builtin::comment>(ins.get_operator()).tag == tag;
    });
}

// Replace the comment with the same tag in place, or add one at the start of m
static void set_tagged_comment(module& m, const std::string& tag, const std::string& text)
{
    auto it = find_tagged_comment(m, tag);
    // A builtin op cannot replace an instruction, so insert the new comment and remove the old
    m.insert_instruction(it == m.end() ? m.begin() : it, builtin::comment{text, tag});
    if(it != m.end())
        m.remove_instruction(it);
}

static optional<std::string> get_tagged_comment(const module& m, const std::string& tag)
{
    auto it = find_tagged_comment(m, tag);
    if(it == m.end())
        return nullopt;
    return any_cast<builtin::comment>(it->get_operator()).text;
}

void record_compiled_for::apply(module_pass_manager& mpm) const
{
    assert(ctx != nullptr);
    if(&mpm.get_module() != mpm.get_root_module())
        return;
    auto& m            = mpm.get_module();
    const auto& device = ctx->get_current_device().get_device_description();
    set_tagged_comment(m, "gpu_target", to_json_string(to_value(device)));
    if(not options.is_null())
        set_tagged_comment(m, "gpu_compile_options", to_json_string(options));
}

optional<device_description> get_compiled_for(const module& m)
{
    auto text = get_tagged_comment(m, "gpu_target");
    if(not text.has_value())
        return nullopt;
    return from_value<device_description>(from_json_string(*text));
}

value get_compile_options(const module& m)
{
    auto text = get_tagged_comment(m, "gpu_compile_options");
    if(not text.has_value())
        return {};
    return from_json_string(*text);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
