/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/netron_output.hpp>
#include <migraphx/json.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/base64.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace {

// from https://onnx.ai/onnx/intro/concepts.html
int get_onnx_type(shape::type_t s_type)
{
    switch(s_type)
    {
    case shape::float_type: return 1;
    case shape::uint8_type: return 2;
    case shape::int8_type: return 3;
    case shape::uint16_type: return 4;
    case shape::int16_type: return 5;
    case shape::int32_type: return 6;
    case shape::int64_type: return 7;
    case shape::bool_type: return 9;
    case shape::half_type: return 10;
    case shape::double_type: return 11;
    case shape::uint32_type: return 12;
    case shape::uint64_type: return 13;
    case shape::bf16_type: return 16;
    case shape::fp8e4m3fn_type: return 17;
    case shape::fp8e4m3fnuz_type: return 18;
    case shape::fp8e5m2_type: return 19;
    case shape::fp8e5m2fnuz_type: return 20;
    case shape::tuple_type: return 0;
    }
    MIGRAPHX_THROW("MIGraphX type " + std::to_string(s_type) + " not supported");
}

auto make_attribute(const migraphx::value& val)
{
    value attribute;
    attribute["name"] = val.get_key();
    auto val_string   = val.to<std::string>();
    val_string        = val_string.substr(val_string.find(":") + 1);
    attribute["s"]    = base64_encode(val_string);
    attribute["type"] = "STRING";
    return attribute;
}

/// Returns a value with the JSON structure needed for a node
auto make_onnx_json_node(instruction_ref ins,
                         std::unordered_map<instruction_ref, std::string> ins_uids)
{
    value node;
    // TODO add support for module inputs
    value input_arr;
    for(instruction_ref input_ins : ins->inputs())
    {
        auto name = input_ins->name();
        if(name == "@literal" or name == "@param")
        {
            input_arr.push_back(ins_uids.at(input_ins));
        }
        // TODO make a better process for handling nodes to ignore
        else if(name.find("hip::hip_allocate_memory") != std::string::npos)
        {
            continue;
        }
        else
        {
            input_arr.push_back(ins_uids.at(input_ins) + "->" + ins_uids.at(ins));
        }
    }
    value output_arr;
    for(instruction_ref output_ins : ins->outputs())
    {
        if(output_ins->name() == "@return")
        {
            output_arr.push_back(ins_uids.at(output_ins));
        }
        else
        {
            output_arr.push_back(ins_uids.at(ins) + "->" + ins_uids.at(output_ins));
        }
    }
    node["input"]  = input_arr;
    node["output"] = output_arr;
    node["name"]   = ins_uids.at(ins);
    node["opType"] = ins->name();
    value op_attribute_arr;
    auto op_value = ins->get_operator().to_value();
    std::for_each(op_value.begin(), op_value.end(), [&](auto v) {
        const std::string& attr_key = v.get_key();
        if(v.is_binary())
        {
            return;
        }
        else if(attr_key == "symbol_name" or attr_key == "name")
        {
            node["opType"] = migraphx::from_value<std::string>(v);
        }
        else
        {
            op_attribute_arr.push_back(make_attribute(v));
        }
    });
    node["attribute"] = op_attribute_arr;
    return node;
}

// ONNX graph constant data called "initializer"
auto make_onnx_json_literal(instruction_ref ins,
                            std::unordered_map<instruction_ref, std::string> ins_uids)
{
    value lit;
    lit["dims"]     = ins->get_shape().lens();
    lit["dataType"] = get_onnx_type(ins->get_shape().type());
    lit["name"]     = ins_uids.at(ins);
    // ignoring literal data, setting to "NULL" in base64
    lit["rawData"] = "TlVMTA==";
    return lit;
}

// TODO handle dynamic shapes
// TODO handle subshapes
auto make_onnx_json_shape(const shape& s)
{
    value ret;
    value dim;
    auto shape_lens = s.lens();
    std::transform(shape_lens.begin(),
                   shape_lens.end(),
                   std::back_inserter(dim),
                   [](std::size_t len) { return len; });
    ret["dim"] = dim;
    return ret;
}

// ONNX graph edges called "valuetype"
auto make_onnx_json_edge(instruction_ref ins,
                         instruction_ref out_ins,
                         std::unordered_map<instruction_ref, std::string> ins_uids)
{
    value ret;
    shape ins_shape = ins->get_shape();
    ret["name"]     = ins_uids.at(ins) + "->" + ins_uids.at(out_ins);
    value type      = {{"tensorType",
                        {{"elemType", get_onnx_type(ins_shape.type())},
                         {"shape", make_onnx_json_shape(ins_shape)}}}};
    ret["type"]     = type;
    return ret;
}

auto make_onnx_json_in_out(instruction_ref ins,
                           std::unordered_map<instruction_ref, std::string> ins_uids)
{
    value ret;
    shape ins_shape = ins->get_shape();
    ret["name"]     = ins_uids.at(ins);
    value type      = {{"tensorType",
                        {{"elemType", get_onnx_type(ins_shape.type())},
                         {"shape", make_onnx_json_shape(ins_shape)}}}};
    ret["type"]     = type;
    return ret;
}

std::unordered_map<instruction_ref, std::string> make_ins_uids(const module& mod)
{
    std::unordered_map<instruction_ref, std::string> ret;
    int count = 0;
    for(auto ins : iterator_for(mod))
    {
        std::string var_name;
        var_name = mod.name() + ":";
        var_name.append(ins->name() + ":");
        var_name.append("@" + std::to_string(count));
        count++;
        ret.emplace(ins, var_name);
    }
    return ret;
}

value make_graph(const module* mod)
{
    value graph = {
        {"node", {}}, {"initializer", {}}, {"input", {}}, {"output", {}}, {"valueInfo", {}}};
    auto ins_uids = make_ins_uids(*mod);
    for(auto ins = mod->begin(); ins != mod->end(); ++ins)
    {
        const auto& name = ins->name();
        if(name == "@literal")
        {
            graph["initializer"].push_back(make_onnx_json_literal(ins, ins_uids));
        }
        else if(name == "@param")
        {
            graph["input"].push_back(make_onnx_json_in_out(ins, ins_uids));
        }
        else if(name == "@return")
        {
            graph["output"].push_back(make_onnx_json_in_out(ins, ins_uids));
        }
        else if(name.find("hip::hip_allocate_memory") != std::string::npos)
        {
            continue;
        }
        else
        {
            graph["node"].push_back(make_onnx_json_node(ins, ins_uids));
            const auto& outputs = ins->outputs();
            for(auto out_ins : outputs)
            {
                if(out_ins->name() != "@return")
                {
                    graph["valueInfo"].push_back(make_onnx_json_edge(ins, out_ins, ins_uids));
                }
            }
        }
    }
    return graph;
}

} // namespace

std::string make_netron_output(const program& prog)
{
    value output;
    auto prog_value           = prog.to_value();
    output["irVersion"]       = prog_value.at("version").to<std::string>();
    output["producerName"]    = "AMDMIGraphX";
    output["producerVersion"] = prog_value.at("migraphx_version").to<std::string>();
    for(auto& mod : prog.get_modules())
    {
        auto graph      = make_graph(mod);
        output["graph"] = graph;
    }
    return to_json_string(output);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
