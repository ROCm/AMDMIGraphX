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

#include <migraphx/onnx/json_output.hpp>

#include <nlohmann/json.hpp>
#include <migraphx/json.hpp>

#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

using json = nlohmann::json;

void write_program_to_onnx_json(const program& prog, std::string filename)
{
    json output;
    auto prog_value = prog.to_value();
    output["irVersion"] = prog_value.at("version").to<std::string>();
    output["producerName"] = "AMDMIGraphX";
    output["producerVersion"] = prog_value.at("migraphx_version").to<std::string>();
    output["graph"] = json::array();
    std::unordered_map<instruction_ref, std::string> names;
    for(auto& mod : prog.get_modules())
    {
        json graph = {
            {"node", json::array()},
            {"initializer", json::array()},
            {"input", json::array()},
            {"output", json::array()},
            {"valueInfo", json::array()}
        };
        auto node_arr = graph["node"];
        auto lit_arr = graph["initializer"];
        auto input_arr = graph["input"];
        auto output_arr = graph["output"];
        auto edge_arr = graph["valueInfo"];
        names = mod->print(
            [&](auto ins, auto ins_uids) {
                const auto& name = ins->name();
                if(name == "@literal")
                {
                    make_onnx_json_literal(ins, ins_uids);
                }
                else if(name == "@param")
                {
                    make_onnx_json_input(ins, ins_uids);
                }
                else if(name == "@return")
                {
                    make_onnx_json_output(ins, ins_uids);
                }
                else
                {
                    node_arr.push_back(make_onnx_json_node(ins, ins_uids));
                    const auto& outputs = ins.outputs();
                    for(auto out_ins : outputs)
                    {
                        node_arr.push_back(make_onnx_json_edge(ins, out_ins, ins_uids));
                    }
                }
            },
        names);
        output["graph"].push_back(graph);
    }
}

auto make_onnx_json_node(instruction ins, std::unordered_map<instruction_ref, std::string> ins_uids)
{
    json::object node;
    json::array input_arr;
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
    json::array output_arr;
    for(auto output_ins : ins->outputs())
    {
        output_arr.push_back(ins_uids.at(ins) + "->" + ins_uids.at(output_ins));
    }
    node["input"] = input_arr;
    node["output"] = output_arr;
    node["name"] = ins_uids.at(ins);
    node["opType"] = ins->name();
    json::object op_attribute_arr;
    auto op_value = ins->get_operator()->to_value();
    std::for_each(op_value.begin(), op_value.end(), [](auto v){
        // No idea if this works, magic from migraphx::from_value to get the right type
        op_attribute_arr.emplace({value.get_key(), migraphx::from_value(value);
    });
    node["attribute"] = op_attribute_arr;
    return node;
}

// ONNX graph constant data called "initializer"
void make_onnx_json_literal()
{

}

// ONNX graph edges called "valuetype"
void make_onnx_json_edge()
{

}

void make_onnx_json_input()
{

}

void make_onnx_json_output()
{

}

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
