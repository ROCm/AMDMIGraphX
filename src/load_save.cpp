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
#include <migraphx/filesystem.hpp>
#include <migraphx/hash.hpp>
#include <migraphx/json.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/logger.hpp>
#include <fstream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

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

// Marker key for an externalized weight: a {weight_ref_key: "<content-hash>"}
// object stands in for the binary blob that lives in the shared weights file.
static const std::string weight_ref_key = "@migraphx_weight_ref";

static value parse_buffer(const char* buffer, std::size_t size, const std::string& format)
{
    if(format == "msgpack")
        return from_msgpack(buffer, size);
    if(format == "json")
        return from_json_string(buffer, size);
    MIGRAPHX_THROW("Unknown format: " + format);
}

static std::vector<char> serialize_value(const value& v, const std::string& format)
{
    if(format == "msgpack")
        return to_msgpack(v);
    if(format == "json")
    {
        std::string s = to_json_string(v);
        return {s.begin(), s.end()};
    }
    MIGRAPHX_THROW("Unknown format: " + format);
}

// Content-addressed key for a weight blob. The byte length is appended so two
// blobs must match in both hash and size to be treated as identical.
static std::string weight_hash(const value::binary& bin)
{
    std::size_t h = 0;
    for(auto b : bin)
        hash_combine(h, b);
    return std::to_string(h) + ":" + std::to_string(bin.size());
}

// Replace every binary blob at or below min_bytes with a reference object,
// accumulating the unique blobs into weights keyed by content hash.
static void extract_weights(value& v, value& weights, std::size_t min_bytes)
{
    if(const auto* bin = v.if_binary())
    {
        if(bin->size() < min_bytes)
            return;
        auto key = weight_hash(*bin);
        if(weights.contains(key))
            assert(weights.at(key).get_binary() == *bin); // content-addressed: equal key => equal bytes
        else
            weights[key] = *bin;
        value ref           = value::object{};
        ref[weight_ref_key] = key;
        v                   = ref; // assigning a keyless value preserves v's own key
        return;
    }
    if(v.is_object() or v.is_array())
        for(auto& child : v)
            extract_weights(child, weights, min_bytes);
}

// Inverse of extract_weights: replace every reference object with its blob.
static void embed_weights(value& v, const value& weights)
{
    if(v.is_object())
    {
        const value* ref = v.find(weight_ref_key);
        if(ref != nullptr and ref != v.end())
        {
            auto key          = ref->to<std::string>();
            const value* blob = weights.find(key);
            if(blob == nullptr or blob == weights.end())
                MIGRAPHX_THROW("Missing weight '" + key + "' in external weights file");
            v = blob->without_key(); // keep v's own key
            return;
        }
    }
    if(v.is_object() or v.is_array())
        for(auto& child : v)
            embed_weights(child, weights);
}

program load(const std::string& filename, const file_options& options)
{
    if(options.weights_file.empty())
        return load_buffer(read_buffer(filename), options);

    auto buffer  = read_buffer(filename);
    value v      = parse_buffer(buffer.data(), buffer.size(), options.format);
    auto wbuffer = read_buffer(options.weights_file);
    embed_weights(v, parse_buffer(wbuffer.data(), wbuffer.size(), options.format));
    program p;
    p.from_value(v);
    return p;
}
program load_buffer(const std::vector<char>& buffer, const file_options& options)
{
    return load_buffer(buffer.data(), buffer.size(), options);
}
program load_buffer(const char* buffer, std::size_t size, const file_options& options)
{
    program p;
    p.from_value(parse_buffer(buffer, size, options.format));
    return p;
}

void save(const program& p, const std::string& filename, const file_options& options)
{
    if(options.weights_file.empty())
    {
        write_buffer(filename, save_buffer(p, options));
        return;
    }

    value v = p.to_value();
    print_miopen_warning(p);
    // Merge into any existing sidecar so weights shared with previously saved
    // programs are reused rather than duplicated.
    value weights = value::object{};
    if(fs::exists(options.weights_file))
    {
        auto wbuffer = read_buffer(options.weights_file);
        weights      = parse_buffer(wbuffer.data(), wbuffer.size(), options.format);
    }
    extract_weights(v, weights, options.min_external_weight_bytes);
    write_buffer(options.weights_file, serialize_value(weights, options.format));
    write_buffer(filename, serialize_value(v, options.format));
}

std::vector<char> save_buffer(const program& p, const file_options& options)
{
    print_miopen_warning(p);
    return serialize_value(p.to_value(), options.format);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
