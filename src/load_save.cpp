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
#include <migraphx/instruction.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/json.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/external_data.hpp>
#include <migraphx/mapped_file.hpp>
#include <migraphx/filesystem.hpp>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <numeric>
#include <system_error>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static program load_from(const char* buffer,
                         std::size_t size,
                         const fs::path& base_dir,
                         const file_options& options)
{
    external_data_load_options load_options;
    load_options.base_dir = base_dir;
    load_options.allowed_dirs.assign(options.external_data_allowed_paths.begin(),
                                     options.external_data_allowed_paths.end());
    load_options.verify_checksum = options.verify_external_weights;
    external_data_reader reader{load_options};
    serialize_context sctx;
    sctx.reader = &reader;
    program p;
    if(options.format == "msgpack")
    {
        p.from_value(from_msgpack(buffer, size), options.finalize, sctx);
    }
    else if(options.format == "json")
    {
        p.from_value(from_json_string(buffer, size), options.finalize, sctx);
    }
    else
    {
        MIGRAPHX_THROW("Unknown format: " + options.format);
    }
    return p;
}

program load(const std::string& filename, const file_options& options)
{
    auto buffer   = read_buffer(filename);
    auto base_dir = options.external_data_path.empty() ? fs::path{filename}.parent_path()
                                                       : fs::path{options.external_data_path};
    return load_from(buffer.data(), buffer.size(), base_dir, options);
}
program load_buffer(const std::vector<char>& buffer, const file_options& options)
{
    return load_buffer(buffer.data(), buffer.size(), options);
}
program load_buffer(const char* buffer, std::size_t size, const file_options& options)
{
    return load_from(buffer, size, options.external_data_path, options);
}

static bool external_weights_enabled(const file_options& options)
{
    return options.external_weights_threshold.has_value() or options.reuse_external_data;
}

static external_data_save_options get_save_options(const file_options& options,
                                                   const fs::path& weights_base,
                                                   const fs::path& base_dir)
{
    external_data_save_options result;
    result.threshold           = options.external_weights_threshold;
    result.reuse_external_data = options.reuse_external_data;
    result.alignment           = options.external_weights_alignment;
    result.weights_base        = weights_base;
    result.base_dir            = base_dir;
    return result;
}

static std::vector<char>
serialize_program(const program& p, const std::string& format, const serialize_context& sctx);

static bool same_file(const fs::path& a, const fs::path& b)
{
    if(lexically_normalize(fs::absolute(a)) == lexically_normalize(fs::absolute(b)))
        return true;
    std::error_code ec;
    return fs::equivalent(a, b, ec) and not ec;
}

static std::vector<std::string> external_file_paths(const value& v,
                                                    std::vector<std::string> result = {})
{
    if(v.contains("external") and v.at("external").contains("file_path"))
        result.push_back(v.at("external").at("file_path").to<std::string>());
    return std::accumulate(v.begin(), v.end(), std::move(result), [](auto acc, const value& x) {
        return external_file_paths(x, std::move(acc));
    });
}

// The generations of weights_base that the program at dest references, which saving over it
// leaves unused; empty when there are none or dest cannot be read as a program
static std::vector<fs::path>
replaced_generations(const fs::path& dest, const fs::path& weights_base, const std::string& format)
{
    try
    {
        const auto base = lexically_normalize(fs::absolute(weights_base));
        if(not fs::exists(dest) or std::none_of(fs::directory_iterator{base.parent_path()},
                                                fs::directory_iterator{},
                                                [&](const fs::directory_entry& e) {
                                                    return is_weights_generation(base, e.path());
                                                }))
            return {};
        auto buffer    = read_buffer(dest);
        value v        = format == "json" ? from_json_string(buffer.data(), buffer.size())
                                          : from_msgpack(buffer.data(), buffer.size());
        auto paths     = external_file_paths(v);
        const auto dir = lexically_normalize(fs::absolute(dest)).parent_path();
        std::vector<fs::path> result;
        transform_if(
            paths.begin(),
            paths.end(),
            std::back_inserter(result),
            [&](const std::string& p) {
                return is_weights_generation(base, lexically_normalize(dir / p));
            },
            [&](const std::string& p) { return lexically_normalize(dir / p); });
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }
    catch(const std::exception&)
    {
        return {};
    }
}

static void remove_unused_generations(const std::vector<fs::path>& generations,
                                      const std::vector<fs::path>& referenced)
{
    // A generation still mapped by a program loaded in this process is kept for it to save again
    std::vector<fs::path> unused;
    std::copy_if(
        generations.begin(), generations.end(), std::back_inserter(unused), [&](const fs::path& g) {
            return not mapped_file::is_mapped(g) and
                   std::none_of(referenced.begin(), referenced.end(), [&](const fs::path& r) {
                       return same_file(g, r);
                   });
        });
    // Best effort, since the new program is already published
    for(const auto& g : unused)
    {
        std::error_code ec;
        fs::remove(g, ec);
    }
}

void save(const program& p, const std::string& filename, const file_options& options)
{
    const fs::path dest{filename};
    if(not external_weights_enabled(options))
    {
        auto buffer = save_buffer(p, options);
        mapped_file::replace_unless_mapped(dest, [&] { write_buffer(dest, buffer); });
        return;
    }
    const fs::path weights_base = options.external_weights_file.empty()
                                      ? fs::path{filename + external_data_extension}
                                      : fs::path{options.external_weights_file};
    if(same_file(dest, weights_base))
        MIGRAPHX_THROW("External weights file must differ from the program file: " + filename);
    if(mapped_file::is_mapped(dest))
        MIGRAPHX_THROW("Cannot overwrite " + filename + " while a loaded program maps it");
    // References are relative to the program file's directory, so the pair can be moved together
    const auto dir = lexically_normalize(fs::absolute(dest)).parent_path();
    external_data_writer writer{get_save_options(options, weights_base, dir)};
    serialize_context sctx;
    sctx.writer            = &writer;
    auto buffer            = serialize_program(p, options.format, sctx);
    const auto& referenced = writer.referenced_files();
    if(std::any_of(referenced.begin(), referenced.end(), [&](const fs::path& r) {
           return same_file(dest, r);
       }))
        MIGRAPHX_THROW("Cannot overwrite " + filename +
                       " since the program reads its external data from it");
    auto replaced = replaced_generations(dest, weights_base, options.format);
    writer.commit();
    try
    {
        write_atomically(dest, buffer, [&](const fs::path& tmp) {
            mapped_file::replace_unless_mapped(dest, [&] { fs::rename(tmp, dest); });
        });
    }
    catch(...)
    {
        writer.discard();
        throw;
    }
    remove_unused_generations(replaced, referenced);
}

// MIOpen doesn't support serializing fusion plans with Find-2.0 APIs
static void print_miopen_warning(const program& p)
{
    auto mods = p.get_modules();
    if(std::any_of(mods.begin(), mods.end(), [](const auto* m) {
           return std::any_of(m->begin(), m->end(), [](const instruction& i) {
               return i.name() == "gpu::miopen_fusion";
           });
       }))
    {
        log::warn()
            << "Program has miopen_fusion instructions for which tuned solutions "
               "are not stored inside serialized MIGraphX program. Consider serializing with "
               "MIGRAPHX_DISABLE_MIOPEN_FUSION=1 flag set.";
    }
}

static std::vector<char>
serialize_program(const program& p, const std::string& format, const serialize_context& sctx)
{
    value v = p.to_value(sctx);
    print_miopen_warning(p);
    std::vector<char> buffer;
    if(format == "msgpack")
    {
        buffer = to_msgpack(v);
    }
    else if(format == "json")
    {
        std::string s = to_json_string(v);
        buffer        = std::vector<char>(s.begin(), s.end());
    }
    else
    {
        MIGRAPHX_THROW("Unknown format: " + format);
    }
    return buffer;
}

std::vector<char> save_buffer(const program& p, const file_options& options)
{
    if(not external_weights_enabled(options))
        return serialize_program(p, options.format, {});
    if(options.external_weights_threshold.has_value() and options.external_weights_file.empty())
        MIGRAPHX_THROW("save_buffer: external weights require external_weights_file");
    external_data_writer writer{
        get_save_options(options, options.external_weights_file, options.external_data_path)};
    serialize_context sctx;
    sctx.writer = &writer;
    auto buffer = serialize_program(p, options.format, sctx);
    writer.commit();
    return buffer;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
