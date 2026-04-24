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

#include <migraphx/netron_output.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/builtin.hpp>
#include <migraphx/stringutils.hpp>
#include <onnx.pb.h>
#include <algorithm>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace onnx = onnx_for_migraphx;

namespace {

int get_onnx_type(shape::type_t s_type)
{
    switch(s_type)
    {
    case shape::float_type: return onnx::TensorProto::FLOAT;
    case shape::uint8_type: return onnx::TensorProto::UINT8;
    case shape::int8_type: return onnx::TensorProto::INT8;
    case shape::uint16_type: return onnx::TensorProto::UINT16;
    case shape::int16_type: return onnx::TensorProto::INT16;
    case shape::int32_type: return onnx::TensorProto::INT32;
    case shape::int64_type: return onnx::TensorProto::INT64;
    case shape::bool_type: return onnx::TensorProto::BOOL;
    case shape::half_type: return onnx::TensorProto::FLOAT16;
    case shape::double_type: return onnx::TensorProto::DOUBLE;
    case shape::uint32_type: return onnx::TensorProto::UINT32;
    case shape::uint64_type: return onnx::TensorProto::UINT64;
    case shape::bf16_type: return onnx::TensorProto::BFLOAT16;
    case shape::fp8e4m3fn_type: return onnx::TensorProto::FLOAT8E4M3FN;
    case shape::fp8e4m3fnuz_type: return onnx::TensorProto::FLOAT8E4M3FNUZ;
    case shape::fp8e5m2_type: return onnx::TensorProto::FLOAT8E5M2;
    case shape::fp8e5m2fnuz_type: return onnx::TensorProto::FLOAT8E5M2FNUZ;
    case shape::tuple_type: return onnx::TensorProto::UNDEFINED;
    case shape::fp4x2_type: return onnx::TensorProto::FLOAT4E2M1;
    }
    MIGRAPHX_THROW("MIGraphX type " + std::to_string(s_type) + " not supported");
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
        if(ins->name() == "@param")
        {
            var_name.append(any_cast<builtin::param>(ins->get_operator()).parameter + ":");
        }
        var_name.append("@" + std::to_string(count));
        count++;
        ret.emplace(ins, var_name);
    }
    return ret;
}

void set_shape_proto(onnx::TensorShapeProto* shape_proto, const shape& s)
{
    for(std::size_t len : s.lens())
    {
        shape_proto->add_dim()->set_dim_value(len);
    }
}

void set_value_info(onnx::ValueInfoProto* vi, const std::string& name, const shape& s)
{
    vi->set_name(name);
    auto* type   = vi->mutable_type();
    auto* tensor = type->mutable_tensor_type();
    tensor->set_elem_type(get_onnx_type(s.type()));
    set_shape_proto(tensor->mutable_shape(), s);
}

void add_initializer(onnx::GraphProto* graph,
                     instruction_ref ins,
                     const std::unordered_map<instruction_ref, std::string>& ins_uids)
{
    auto* init = graph->add_initializer();
    init->set_name(ins_uids.at(ins));
    init->set_data_type(get_onnx_type(ins->get_shape().type()));
    for(std::size_t d : ins->get_shape().lens())
    {
        init->add_dims(d);
    }
}

void add_node(onnx::GraphProto* graph,
              instruction_ref ins,
              const std::unordered_map<instruction_ref, std::string>& ins_uids)
{
    auto* node = graph->add_node();

    std::string op_type = ins->name();
    auto op_value       = ins->get_operator().to_value();
    std::for_each(op_value.begin(), op_value.end(), [&](const auto& v) {
        const std::string& attr_key = v.get_key();
        if(v.is_binary() or attr_key == "code_object")
        {
            return;
        }
        else if(attr_key == "symbol_name" or attr_key == "name")
        {
            op_type = migraphx::from_value<std::string>(v);
        }
        else
        {
            auto* attr = node->add_attribute();
            attr->set_name(attr_key);

            auto val_string     = v.template to<std::string>();
            std::string sub_str = attr_key + ":";
            auto find_key       = val_string.find(sub_str);
            if(find_key != std::string::npos)
            {
                val_string = val_string.substr(find_key + sub_str.length() + 1);
            }
            attr->set_type(onnx::AttributeProto::STRING);
            attr->set_s(val_string);
        }
    });

    node->set_op_type(op_type);
    node->set_name(ins_uids.at(ins));

    for(instruction_ref input_ins : ins->inputs())
    {
        auto name = input_ins->name();
        if(name == "@literal" or name == "@param")
        {
            node->add_input(ins_uids.at(input_ins));
        }
        else if(name.find("hip::hip_allocate_memory") != std::string::npos)
        {
            continue;
        }
        else
        {
            node->add_input(ins_uids.at(input_ins) + "->" + ins_uids.at(ins));
        }
    }

    for(instruction_ref output_ins : ins->outputs())
    {
        if(output_ins->name() == "@return")
        {
            node->add_output(ins_uids.at(output_ins));
        }
        else
        {
            node->add_output(ins_uids.at(ins) + "->" + ins_uids.at(output_ins));
        }
    }

    if(not ins->get_debug_symbols().empty())
    {
        auto* attr = node->add_attribute();
        attr->set_name("debug symbols");
        attr->set_type(onnx::AttributeProto::STRING);
        attr->set_s(join_strings(ins->get_debug_symbols(), ", "));
    }
}

void build_graph(onnx::GraphProto* graph, const module* mod)
{
    auto ins_uids = make_ins_uids(*mod);
    for(auto ins = mod->begin(); ins != mod->end(); ++ins)
    {
        const auto& name = ins->name();
        if(name == "@literal")
        {
            add_initializer(graph, ins, ins_uids);
        }
        else if(name == "@param")
        {
            set_value_info(graph->add_input(), ins_uids.at(ins), ins->get_shape());
        }
        else if(name == "@return")
        {
            set_value_info(graph->add_output(), ins_uids.at(ins), ins->get_shape());
        }
        else if(name.find("hip::hip_allocate_memory") != std::string::npos)
        {
            continue;
        }
        else
        {
            add_node(graph, ins, ins_uids);
            for(auto out_ins : ins->outputs())
            {
                if(out_ins->name() != "@return")
                {
                    set_value_info(graph->add_value_info(),
                                   ins_uids.at(ins) + "->" + ins_uids.at(out_ins),
                                   ins->get_shape());
                }
            }
        }
    }
}

} // namespace

void write_netron_output(const program& prog, std::ostream& os)
{
    onnx::ModelProto model;
    auto prog_value = prog.to_value();
    model.set_ir_version(prog.get_program_file_version());
    model.set_producer_name("AMDMIGraphX");
    model.set_producer_version(prog_value.at("migraphx_version").to<std::string>());
    
    // only exporting the main module
    // TODO handle submodules as ONNX subgraphs
    build_graph(model.mutable_graph(), prog.get_main_module());

    model.SerializeToOstream(&os);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
