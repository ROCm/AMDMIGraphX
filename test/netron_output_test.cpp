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
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/onnx.hpp>
#include <onnx.pb.h>
#include <sstream>
#include <test.hpp>

namespace onnx = onnx_for_migraphx;

static void set_value_info(onnx::ValueInfoProto* vi,
                           const std::string& name,
                           int elem_type,
                           const std::vector<int64_t>& dims)
{
    vi->set_name(name);
    auto* tensor = vi->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(elem_type);
    for(auto d : dims)
        tensor->mutable_shape()->add_dim()->set_dim_value(d);
}

static void add_initializer(onnx::GraphProto* graph,
                             const std::string& name,
                             int data_type,
                             const std::vector<int64_t>& dims)
{
    auto* init = graph->add_initializer();
    init->set_name(name);
    init->set_data_type(data_type);
    for(auto d : dims)
        init->add_dims(d);
}

static onnx::NodeProto* add_node(onnx::GraphProto* graph,
                                  const std::string& op_type,
                                  const std::string& name,
                                  const std::vector<std::string>& inputs,
                                  const std::vector<std::string>& outputs)
{
    auto* node = graph->add_node();
    node->set_op_type(op_type);
    node->set_name(name);
    for(const auto& in : inputs)
        node->add_input(in);
    for(const auto& out : outputs)
        node->add_output(out);
    return node;
}

static void add_string_attribute(onnx::NodeProto* node,
                                  const std::string& name,
                                  const std::string& value)
{
    auto* attr = node->add_attribute();
    attr->set_name(name);
    attr->set_type(onnx::AttributeProto::STRING);
    attr->set_s(value);
}

static onnx::GraphProto parse_graph(const std::string& proto_binary)
{
    onnx::ModelProto model;
    model.ParseFromString(proto_binary);
    return model.graph();
}

TEST_CASE(netron_output_basic)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto y   = mm->add_parameter("y", {migraphx::shape::float_type, {2, 3}});
    auto sum = mm->add_instruction(migraphx::make_op("add"), x, y);
    mm->add_return({sum});

    std::ostringstream os;
    migraphx::write_netron_output(p, os);

    onnx::GraphProto expected;
    add_node(&expected,
             "add",
             "main:add:@2",
             {"main:@param:x:@1", "main:@param:y:@0"},
             {"main:@return:@3"});
    set_value_info(expected.add_input(), "main:@param:y:@0", onnx::TensorProto::FLOAT, {2, 3});
    set_value_info(expected.add_input(), "main:@param:x:@1", onnx::TensorProto::FLOAT, {2, 3});
    set_value_info(expected.add_output(), "main:@return:@3", onnx::TensorProto::FLOAT, {2, 3});

    EXPECT(parse_graph(os.str()).SerializeAsString() == expected.SerializeAsString());
}

TEST_CASE(netron_output_with_literal)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto lit = mm->add_literal(migraphx::literal{{migraphx::shape::float_type, {2, 3}},
                                                  {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}});
    auto sum = mm->add_instruction(migraphx::make_op("add"), x, lit);
    mm->add_return({sum});

    std::ostringstream os;
    migraphx::write_netron_output(p, os);

    onnx::GraphProto expected;
    add_node(&expected,
             "add",
             "main:add:@2",
             {"main:@param:x:@1", "main:@literal:@0"},
             {"main:@return:@3"});
    add_initializer(&expected, "main:@literal:@0", onnx::TensorProto::FLOAT, {2, 3});
    set_value_info(expected.add_input(), "main:@param:x:@1", onnx::TensorProto::FLOAT, {2, 3});
    set_value_info(expected.add_output(), "main:@return:@3", onnx::TensorProto::FLOAT, {2, 3});

    EXPECT(parse_graph(os.str()).SerializeAsString() == expected.SerializeAsString());
}

TEST_CASE(netron_output_multiple_types)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto a = mm->add_parameter("a", {migraphx::shape::int32_type, {2, 3}});
    mm->add_parameter("b", {migraphx::shape::int64_type, {4}});
    mm->add_parameter("c", {migraphx::shape::half_type, {2, 3}});
    mm->add_parameter("d", {migraphx::shape::double_type, {2, 3}});
    mm->add_parameter("e", {migraphx::shape::bool_type, {2}});
    mm->add_parameter("f", {migraphx::shape::uint8_type, {2, 3}});
    mm->add_parameter("g", {migraphx::shape::bf16_type, {2, 3}});

    auto lit = mm->add_literal(migraphx::literal{{migraphx::shape::int32_type, {2, 3}},
                                                  {1, 2, 3, 4, 5, 6}});
    auto sum = mm->add_instruction(migraphx::make_op("add"), a, lit);
    mm->add_return({sum});

    std::ostringstream os;
    migraphx::write_netron_output(p, os);

    onnx::GraphProto expected;
    add_node(&expected,
             "add",
             "main:add:@8",
             {"main:@param:a:@7", "main:@literal:@0"},
             {"main:@return:@9"});
    add_initializer(&expected, "main:@literal:@0", onnx::TensorProto::INT32, {2, 3});
    set_value_info(expected.add_input(), "main:@param:g:@1", onnx::TensorProto::BFLOAT16, {2, 3});
    set_value_info(expected.add_input(), "main:@param:f:@2", onnx::TensorProto::UINT8, {2, 3});
    set_value_info(expected.add_input(), "main:@param:e:@3", onnx::TensorProto::BOOL, {2});
    set_value_info(expected.add_input(), "main:@param:d:@4", onnx::TensorProto::DOUBLE, {2, 3});
    set_value_info(expected.add_input(), "main:@param:c:@5", onnx::TensorProto::FLOAT16, {2, 3});
    set_value_info(expected.add_input(), "main:@param:b:@6", onnx::TensorProto::INT64, {4});
    set_value_info(expected.add_input(), "main:@param:a:@7", onnx::TensorProto::INT32, {2, 3});
    set_value_info(expected.add_output(), "main:@return:@9", onnx::TensorProto::INT32, {2, 3});

    EXPECT(parse_graph(os.str()).SerializeAsString() == expected.SerializeAsString());
}

TEST_CASE(netron_output_op_attributes_and_chain)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto y   = mm->add_parameter("y", {migraphx::shape::float_type, {2, 3}});
    auto sm  = mm->add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), x);
    auto sum = mm->add_instruction(migraphx::make_op("add"), sm, y);
    mm->add_return({sum});

    std::ostringstream os;
    migraphx::write_netron_output(p, os);

    onnx::GraphProto expected;
    auto* sm_node = add_node(&expected,
                              "softmax",
                              "main:softmax:@2",
                              {"main:@param:x:@1"},
                              {"main:softmax:@2->main:add:@3"});
    add_string_attribute(sm_node, "axis", "1");
    add_node(&expected,
             "add",
             "main:add:@3",
             {"main:softmax:@2->main:add:@3", "main:@param:y:@0"},
             {"main:@return:@4"});
    set_value_info(expected.add_input(), "main:@param:y:@0", onnx::TensorProto::FLOAT, {2, 3});
    set_value_info(expected.add_input(), "main:@param:x:@1", onnx::TensorProto::FLOAT, {2, 3});
    set_value_info(expected.add_output(), "main:@return:@4", onnx::TensorProto::FLOAT, {2, 3});
    set_value_info(
        expected.add_value_info(), "main:softmax:@2->main:add:@3", onnx::TensorProto::FLOAT, {2, 3});

    EXPECT(parse_graph(os.str()).SerializeAsString() == expected.SerializeAsString());
}

TEST_CASE(netron_output_debug_symbols)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto y   = mm->add_parameter("y", {migraphx::shape::float_type, {2, 3}});
    auto sum = mm->add_instruction(migraphx::make_op("add"), x, y);
    mm->add_debug_symbols(sum, {"test_file.onnx:42", "origin_op:Add"});
    mm->add_return({sum});

    std::ostringstream os;
    migraphx::write_netron_output(p, os);

    onnx::GraphProto expected;
    auto* node = add_node(&expected,
                           "add",
                           "main:add:@2",
                           {"main:@param:x:@1", "main:@param:y:@0"},
                           {"main:@return:@3"});
    add_string_attribute(node, "debug symbols", "origin_op:Add, test_file.onnx:42");
    set_value_info(expected.add_input(), "main:@param:y:@0", onnx::TensorProto::FLOAT, {2, 3});
    set_value_info(expected.add_input(), "main:@param:x:@1", onnx::TensorProto::FLOAT, {2, 3});
    set_value_info(expected.add_output(), "main:@return:@3", onnx::TensorProto::FLOAT, {2, 3});

    EXPECT(parse_graph(os.str()).SerializeAsString() == expected.SerializeAsString());
}

TEST_CASE(netron_output_roundtrip)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto y   = mm->add_parameter("y", {migraphx::shape::float_type, {2, 3}});
    auto sum = mm->add_instruction(migraphx::make_op("add"), x, y);
    mm->add_return({sum});

    std::ostringstream os;
    migraphx::write_netron_output(p, os);
    std::string output = os.str();

    migraphx::onnx_options options;
    options.skip_unknown_operators = true;
    auto p2 = migraphx::parse_onnx_buffer(output.data(), output.size(), options);
    EXPECT(p2.get_main_module()->size() > 0);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
