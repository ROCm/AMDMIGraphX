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
#include "migraphx/iterator_for.hpp"
#include <migraphx/serialize.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/literal.hpp>
#include <nlohmann/json.hpp>
#include <migraphx/trt_json.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using json = nlohmann::json;

static void value_to_json(const value& val, json& j);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

namespace nlohmann {
template <>
struct adl_serializer<migraphx::value>
{
    static void to_json(json& j, const migraphx::value& val) { migraphx::value_to_json(val, j); }

    static void from_json(const json& j, migraphx::value& val) {}
};
} // namespace nlohmann

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using json = nlohmann::json;

template <class T>
static void value_to_json(const T& x, json& j)
{
    j = x;
}

static void value_to_json(const value::binary& x, json& j)
{
    j          = json::object();
    j["bytes"] = std::vector<int>(x.begin(), x.end());
}

static void value_to_json(const std::vector<value>& x, json& j)
{
    for(const auto& v : x)
    {
        if(v.get_key().empty())
        {
            j.push_back(v);
        }
        else
        {
            j[v.get_key()] = v.without_key();
        }
    }
}

static void value_to_json(std::nullptr_t&, json& j) { j = {}; }

static void value_to_json(const value& val, json& j)
{
    if(val.is_array())
    {
        j = json::array();
    }

    if(val.is_object())
    {
        j = json::object();
    }

    val.visit([&](auto v) { value_to_json(v, j); });
}
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using json = nlohmann::json;

std::string type_to_trt_type_string(migraphx::shape::type_t type)
{
    switch(type)
    {
    case migraphx::shape::float_type: return "FP32";
    case migraphx::shape::int8_type: return "INT8";
    case migraphx::shape::tuple_type: return "TUPLE";
    default: throw 5;
    }
}

std::string to_trt_json_string(const program& p)
{
    std::cout << p << std::endl;
    json j        = json::object();
    auto& jlayers = j["Layers"] = json::array();

    auto* mm = p.get_main_module();

    std::unordered_map<instruction_ref, std::string> ins_to_name;
    std::unordered_map<std::string, unsigned int> name_count;

    const auto& param_names = mm->get_parameter_names();
    for(const auto& param_name : param_names)
        ins_to_name[mm->get_parameter(param_name)] = param_name;

    for(auto ins : iterator_for(*mm))
    {
        if(ins->name() == "@param")
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
        ins_to_name[ins] = ins_name + "_" + std::to_string(count);

        auto jlayer         = json::object();
        jlayer["Name"]      = ins_to_name[ins];
        jlayer["LayerType"] = ins->name();
        auto& jlayer_inputs = jlayer["Inputs"] = json::array();

        for(auto input : ins->inputs())
        {
            auto jinput      = json::object();
            auto name_suffix = input->name() == "@param" ? "" : "_out";
            jinput["Name"]   = ins_to_name.at(input) + name_suffix;
            // TODO treat dynamic dims differently
            jinput["Dimensions"]      = input->get_shape().lens();
            jinput["Format/Datatype"] = type_to_trt_type_string(input->get_shape().type());
            jlayer_inputs.push_back(jinput);
        }

        auto& jlayer_outputs = jlayer["Outputs"] = json::array();
        auto joutput                             = json::object();
        joutput["Name"]                          = ins_to_name[ins] + "_out";
        joutput["Dimensions"]                    = ins->get_shape().lens();
        joutput["Format/Datatype"]               = type_to_trt_type_string(ins->get_shape().type());
        jlayer_outputs.push_back(joutput);

        // TODO add attributes
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
                jlayer[v.get_key()] = v.without_key();
            }
        }

        jlayers.push_back(jlayer);
    }

    auto& jbindings = j["Bindings"] = json::array();
    for(const auto& param_name : param_names)
    {
        jbindings.push_back(param_name);
    }
    return j.dump(2);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
