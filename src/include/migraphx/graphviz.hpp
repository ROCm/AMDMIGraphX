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

#ifndef MIGRAPHX_GUARD_UTILS_GRAPHVIZ_HPP
#define MIGRAPHX_GUARD_UTILS_GRAPHVIZ_HPP

#include <string>
#include <iomanip>
#include <sstream>
#include <migraphx/stringutils.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace graphviz {

struct html_table_style
{
    int border      = 0;
    int cellborder  = 0;
    int cellpadding = 4;
    int cellspacing = 0;
};

struct graphviz_node_style
{
    std::string fillcolor = "lightgray";
    std::string fontcolor = "black";
    std::string style     = "rounded,filled";
    std::string shape     = "none";
    std::string fontname  = "Helvetica";
};

struct graphviz_node_content
{
    std::string title;
    std::vector<std::string> body_lines;
    std::optional<std::pair<double, double>> perf_data;

    html_table_style html_style{};
    graphviz_node_style node_style{};
};

inline std::string enclose_name(const std::string& name)
{
    return '"' + replace_string(name, "\"", "\\\"") + '"';
}

inline std::string html_cell(const std::string& content, const std::string& align = "center")
{
    return "<TR><TD ALIGN=\"" + align + "\">" + content + "</TD></TR>";
}

inline std::string html_bold(const std::string& content) { return "<B>" + content + "</B>"; }

inline std::string html_table_start(const html_table_style& style)
{
    return "<<TABLE BORDER=\"" + std::to_string(style.border) + "\" CELLBORDER=\"" +
           std::to_string(style.cellborder) + "\" CELLPADDING=\"" +
           std::to_string(style.cellpadding) + "\" CELLSPACING=\"" +
           std::to_string(style.cellspacing) + "\" COLOR=\"transparent\">";
}

inline std::string html_table_end() { return "</TABLE>>"; }

inline std::string html_color_style(const std::string& fill)
{
    return " style=\"rounded,filled\" fillcolor=\"" + fill + "\"";
}

inline std::string block_style(std::string color = "lightgray")
{
    return " color=black fillcolor=" + color +
           " fontname=Helvetica shape=none style=\"rounded,filled\"";
}

inline std::string format_shape_name(const migraphx::shape& s, std::string linebreak = "\\n")
{
    if(s.sub_shapes().empty())
    {
        if(s.dynamic())
        {
            return "dynamic" + linebreak + s.type_string() + linebreak + "{" + to_string_range(s.dyn_dims()) +
                   "}";
        }
        return s.type_string() + linebreak + "{" + to_string_range(s.lens()) + "}, {" +
               to_string_range(s.strides()) + "}";
    }
    return "[" + to_string_range(s.sub_shapes()) + "]";
}

inline std::string build_html_label(const graphviz_node_content& content)
{
    std::ostringstream ss;

    ss << html_table_start(content.html_style);
    ss << html_cell(html_bold(content.title));

    if(content.perf_data)
    {
        // TODO:
        // ss << graphviz::html_cell(": " + std::to_string(avg) + "ms, " + std::to_string(percent) +
        // "%");
    }

    std::for_each(content.body_lines.begin(),
                  content.body_lines.end(),
                  [&ss](const std::string line) { ss << html_cell(line); });

    ss << html_table_end();
    return ss.str();
}

inline std::string build_plain_label(const std::string& title, const std::string& body)
{
    std::ostringstream ss;
    ss << "\"" << title << "\\n" << body << "\"";
    return ss.str();
}

inline std::string build_node_style(const graphviz_node_style& node_style)
{
    std::ostringstream ss;
    ss << "style=\"" << node_style.style << "\" ";
    ss << "fillcolor=" << node_style.fillcolor << " ";
    ss << "shape=" << node_style.shape << " ";
    ss << "fontname=" << node_style.fontname;
    return ss.str();
}

inline graphviz_node_content get_node_content(const instruction_ref& ins)
{
    const auto& op         = ins->get_operator();
    const std::string name = ins->name();

    graphviz_node_content content;

    if(name == "@param")
    {
        // title should be param
        // body should be the shape
        content.title = name;
        content.body_lines.push_back(graphviz::format_shape_name(ins->get_shape(), "<BR/>"));

        content.html_style = {0, 0, 0, 0};
        content.node_style = {"khaki", "black", "filled", "rectangle", "Helvectica"};
    }
    else if(name == "gpu::code_object")
    {
        // TODO: handle gpu::code_object case
        // title should be symbol_name
        // body should be std::string label = to_string(ins->get_operator());
        content.title = op.to_value()["symbol_name"].to<std::string>();
        content.body_lines.push_back(to_string(op));

        content.node_style.fillcolor = "royalblue";
    }
    else
    {
        content.title = name;
        content.body_lines.push_back(to_string(op));

        const auto& attr = op.attributes();

        if(attr.contains("style"))
            content.node_style.style = attr.at("style").to<std::string>();
        if(attr.contains("fillcolor"))
            content.node_style.fillcolor = attr.at("fillcolor").to<std::string>();
        if(attr.contains("fontcolor"))
            content.node_style.fontcolor = attr.at("fontcolor").to<std::string>();
    }
    return content;
}

} // namespace graphviz
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
