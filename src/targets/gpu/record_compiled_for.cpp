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

void record_compiled_for::apply(module_pass_manager& mpm) const
{
    if(&mpm.get_module() != mpm.get_root_module())
        return;
    auto& m = mpm.get_module();
    set_compiled_for(m, ctx->get_current_device().get_device_description());
    set_compile_options(m, merge_compile_options(get_compile_options(m), options));
}

value compile_options_value(const compile_options& options, const value& backend_options)
{
    return {{"offload_copy", options.offload_copy},
            {"fast_math", options.fast_math},
            {"exhaustive_tune", options.exhaustive_tune},
            {"recompilable", options.recompilable},
            {"compile_mode", to_string(options.compile_mode)},
            {"backend_options", backend_options}};
}

static value get_or(const value& v, const std::string& key, const value& fallback)
{
    if(v.is_object() and v.contains(key))
        return v.at(key);
    return fallback;
}

bool is_exhaustive(const value& options)
{
    return get_or(options, "exhaustive_tune", false).to<bool>() or
           get_or(options, "compile_mode", "").to<std::string>() == "max";
}

value merge_compile_options(const value& recorded, const value& user)
{
    if(not recorded.is_object())
        return user;
    auto result               = recorded;
    result["exhaustive_tune"] = is_exhaustive(recorded) or is_exhaustive(user);
    auto backend              = get_or(recorded, "backend_options", value::object{});
    // An empty object does not survive serialization as an object
    if(not backend.is_object())
        backend = value::object{};
    for(const auto& opt : get_or(user, "backend_options", value::object{}))
        backend[opt.get_key()] = opt.without_key();
    result["backend_options"] = backend;
    return result;
}

static instruction_ref find_tagged_comment(const module& m, const std::string& tag)
{
    return std::find_if(m.begin(), m.end(), [&](const instruction& ins) {
        return ins.name() == "@comment" and
               any_cast<builtin::comment>(ins.get_operator()).tag == tag;
    });
}

// Comments are written at the start of m, replacing the one with the same tag
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

void set_compiled_for(module& m, const device_description& device)
{
    set_tagged_comment(m, "gpu_target", to_json_string(to_value(device)));
}

optional<device_description> get_compiled_for(const module& m)
{
    auto text = get_tagged_comment(m, "gpu_target");
    if(not text.has_value())
        return nullopt;
    return from_value<device_description>(from_json_string(*text));
}

void set_compile_options(module& m, const value& options)
{
    set_tagged_comment(m, "gpu_compile_options", to_json_string(options));
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
