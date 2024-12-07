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

#include <nlohmann/json.hpp>
#include <migraphx/json.hpp>

#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/serialize.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {
namespace {

using json = nlohmann::json;

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
    default: {
        MIGRAPHX_THROW("MIGraphX type " + std::to_string(s_type) + " not supported");
    }
    }
}

auto make_onnx_json_node(instruction_ref ins, std::unordered_map<instruction_ref, std::string> ins_uids)
{
    json node = json::object();
    json input_arr = json::array();
    for(auto input_ins : ins->inputs())
    {
        auto name = input_ins->name();
        if(name == "@literal" or name == "@param")
        {
            input_arr.push_back(ins_uids.at(input_ins));
        }
        else
        {
            input_arr.push_back(ins_uids.at(input_ins) + "->" + ins_uids.at(ins));
        }
    }
    json output_arr = json::array();
    for(auto output_ins : ins->outputs())
    {
        output_arr.push_back(ins_uids.at(ins) + "->" + ins_uids.at(output_ins));
    }
    node["input"]  = input_arr;
    node["output"] = output_arr;
    node["name"]   = ins_uids.at(ins);
    node["opType"] = ins->name();
    json op_attribute_arr = json::object();
    auto op_value = ins->get_operator().to_value();
    std::for_each(op_value.begin(), op_value.end(), [&](auto v){
        op_attribute_arr.emplace(v.get_key(), migraphx::from_value<std::string>(v));
    });
    node["attribute"] = op_attribute_arr;
    return node;
}

// ONNX graph constant data called "initializer"
auto make_onnx_json_literal(instruction_ref ins, std::unordered_map<instruction_ref, std::string> ins_uids)
{
    json lit = json::object();
    lit["dims"] = ins->get_shape().lens();
    // TODO figure out the data types number
    lit["dataType"] = 1;
    lit["name"] = ins_uids.at(ins);
    // NULL in base64
    lit["rawData"] = "TlVMTA==";
    return lit;
}

auto make_onnx_json_shape(const shape& s)
{
    json ret = json::object();
    json dim = json::array();
    for(std::size_t len : s.lens())
    {
        dim.push_back({"dimValue", len});
    }
    return ret;
}

// ONNX graph edges called "valuetype"
auto make_onnx_json_edge(instruction_ref ins, instruction_ref out_ins, std::unordered_map<instruction_ref, std::string> ins_uids)
{
    json ret = json::object();
    shape ins_shape = ins->get_shape();
    ret["name"] = ins_uids.at(ins) + "->" + ins_uids.at(out_ins);
    ret["type"] = {
        "tensorType", {
            {"elemType", get_onnx_type(ins_shape.type())}, 
            {"shape", make_onnx_json_shape(ins_shape)}
        }
    };
    return ret;
}

auto make_onnx_json_in_out(instruction_ref ins, std::unordered_map<instruction_ref, std::string> ins_uids)
{
    json ret = json::object();
    shape ins_shape = ins->get_shape();
    ret["name"] = ins_uids.at(ins);
    ret["type"] = {
        "tensorType", {
            {"elemType", get_onnx_type(ins_shape.type())}, 
            {"shape", make_onnx_json_shape(ins_shape)}
        }
    };
    return ret;
}

} // namespace

auto make_netron_output(const program& prog)
{
    json output;
    auto prog_value           = prog.to_value();
    output["irVersion"]       = prog_value.at("version").to<std::string>();
    output["producerName"]    = "AMDMIGraphX";
    output["producerVersion"] = prog_value.at("migraphx_version").to<std::string>();
    output["graph"]           = json::array();
    std::unordered_map<instruction_ref, std::string> names;
    for(auto& mod : prog.get_modules())
    {
        json graph      = {{"node", json::array()},
                           {"initializer", json::array()},
                           {"input", json::array()},
                           {"output", json::array()},
                           {"valueInfo", json::array()}};
        auto node_arr   = graph["node"];
        auto lit_arr    = graph["initializer"];
        auto input_arr  = graph["input"];
        auto output_arr = graph["output"];
        auto edge_arr   = graph["valueInfo"];
        names           = mod->print(
            [&](auto ins, auto ins_uids) {
                const auto& name = ins->name();
                if(name == "@literal")
                {
                    lit_arr.push_back(make_onnx_json_literal(ins, ins_uids));
                }
                else if(name == "@param")
                {
                    input_arr.push_back(make_onnx_json_in_out(ins, ins_uids));
                }
                else if(name == "@return")
                {
                    output_arr.push_back(make_onnx_json_in_out(ins, ins_uids));
                }
                else
                {
                    node_arr.push_back(make_onnx_json_node(ins, ins_uids));
                    const auto& outputs = ins->outputs();
                    for(auto out_ins : outputs)
                    {
                        node_arr.push_back(make_onnx_json_edge(ins, out_ins, ins_uids));
                    }
                }
            },
            names);
        output["graph"].push_back(graph);
    }
    return output.dump();
}

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
