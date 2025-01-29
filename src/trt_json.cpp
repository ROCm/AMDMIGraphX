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
#include <migraphx/json.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/trt_json.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::string type_to_trt_type_string(shape::type_t type)
{
    switch(type)
    {
    case shape::bool_type: return "BOOL";
    case shape::half_type: return "FP16";
    case shape::float_type: return "FP32";
    case shape::double_type: return "FP64";
    case shape::uint8_type: return "UINT8";
    case shape::int8_type: return "INT8";
    case shape::uint16_type: return "UINT16";
    case shape::int16_type: return "INT16";
    case shape::uint32_type: return "UINT32";
    case shape::int32_type: return "INT32";
    case shape::uint64_type: return "UINT64";
    case shape::int64_type: return "INT64";
    case shape::bf16_type: return "BF16";
    case shape::tuple_type: return "TUPLE";
    // TODO fp8 types
    default: MIGRAPHX_THROW("Unsupported type: " + shape::name(type));
    }
}

std::string to_trt_json_string(const program& p, std::optional<size_t> indent)
{
    std::unordered_map<instruction_ref, std::string> ins_to_name;
    std::unordered_map<std::string, unsigned int> name_count;

    auto* mm = p.get_main_module();

    // Parameters are added as Bindings
    const auto& param_names = mm->get_parameter_names();
    for(const auto& param_name : param_names)
        ins_to_name[mm->get_parameter(param_name)] = param_name;

    value vlayers;
    for(const auto ins : iterator_for(*mm))
    {
        // Skip these instructions to avoid clutter
        static const std::vector<std::string> skip_instructions{
            "@param", "check_context::migraphx::gpu::context", "hip::hip_allocate_memory", "load"};
        if(contains(skip_instructions, ins->name()))
            continue;

        std::string ins_name;
        if(ins->name() == "gpu::code_object")
        {
            ins_name =
                ins->get_operator().to_value()["symbol_name"].without_key().to<std::string>();
        }
        else
        {
            ins_name = ins->name();
        }
        auto count       = name_count[ins_name]++;
        ins_name         = ins_name + "_" + std::to_string(count);
        ins_to_name[ins] = ins_name;

        value vlayer;
        vlayer["Name"]      = ins_to_name.at(ins);
        vlayer["LayerType"] = ins->name();

        value vlayer_inputs;
        for(auto input : ins->inputs())
        {
            if(input->name() == "load")
                continue;
            value vinput;
            auto name_suffix = input->name() == "@param" ? "" : "_out";
            vinput["Name"]   = ins_to_name.at(input) + name_suffix;
            // TODO treat dynamic dims differently
            vinput["Dimensions"]      = input->get_shape().lens();
            vinput["Format/Datatype"] = type_to_trt_type_string(input->get_shape().type());
            vlayer_inputs.push_back(vinput);
        }
        vlayer["Inputs"] = vlayer_inputs;

        value voutput;
        voutput["Name"]            = ins_name + "_out";
        voutput["Dimensions"]      = ins->get_shape().lens();
        voutput["Format/Datatype"] = type_to_trt_type_string(ins->get_shape().type());
        vlayer["Outputs"].push_back(voutput);

        auto val = ins->get_operator().to_value();
        static const std::vector<std::string> skip_keys{"code_object",
                                                        "shape",
                                                        "expected_inputs",
                                                        "output",
                                                        "symbol_name",
                                                        "literal",
                                                        "bytes",
                                                        "data",
                                                        "solution_object"};
        for(const auto& v : val)
        {
            if(not contains(skip_keys, v.get_key()))
            {
                vlayer[v.get_key()] = v.without_key();
            }
        }

        vlayers.push_back(vlayer);
    }
    value j;
    j["Layers"] = vlayers;

    // Bindings indicate network inputs and outputs
    value vbindings;
    for(const auto& param_name : param_names)
    {
        vbindings.push_back(param_name);
    }
    // Bind return as output
    vbindings.push_back(vlayers.back()["Outputs"][0]["Name"].to<std::string>());
    j["Bindings"] = vbindings;

    return indent ? to_pretty_json_string(j, *indent) : to_json_string(j);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
