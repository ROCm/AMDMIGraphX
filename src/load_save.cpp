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
#include <migraphx/file_buffer.hpp>
#include <migraphx/json.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/external_data.hpp>
#include <migraphx/filesystem.hpp>
#include <fstream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static program load_from(const char* buffer,
                         std::size_t size,
                         const fs::path& base_dir,
                         const file_options& options)
{
    scoped_external_data_reader reader{
        external_data_load_options{base_dir, options.verify_external_weights}};
    program p;
    if(options.format == "msgpack")
    {
        p.from_value(from_msgpack(buffer, size), options.finalize);
    }
    else if(options.format == "json")
    {
        p.from_value(from_json_string(buffer, size), options.finalize);
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
                                                   const fs::path& weights_file,
                                                   const fs::path& base_dir)
{
    external_data_save_options result;
    result.threshold           = options.external_weights_threshold;
    result.reuse_external_data = options.reuse_external_data;
    result.alignment           = options.external_weights_alignment;
    result.weights_file        = weights_file;
    result.base_dir            = base_dir;
    return result;
}

static std::vector<char> serialize_program(const program& p, const std::string& format);

void save(const program& p, const std::string& filename, const file_options& options)
{
    if(not external_weights_enabled(options))
    {
        write_buffer(filename, save_buffer(p, options));
        return;
    }
    auto weights_file = options.external_weights_file.empty()
                            ? fs::path{filename + external_data_extension}
                            : fs::path{options.external_weights_file};
    // References are relative to the program file's directory, so the pair can be moved together
    scoped_external_data_writer writer{
        get_save_options(options, weights_file, fs::absolute(fs::path{filename}).parent_path())};
    auto buffer = serialize_program(p, options.format);
    writer.commit();
    write_atomically(filename, buffer);
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

static std::vector<char> serialize_program(const program& p, const std::string& format)
{
    value v = p.to_value();
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
        return serialize_program(p, options.format);
    if(options.external_weights_threshold.has_value() and options.external_weights_file.empty())
        MIGRAPHX_THROW("save_buffer: external weights require external_weights_file");
    scoped_external_data_writer writer{
        get_save_options(options, options.external_weights_file, options.external_data_path)};
    auto buffer = serialize_program(p, options.format);
    writer.commit();
    return buffer;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
