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

#include <migraphx/graphviz.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace graphviz {

static std::string html_bold(const std::string& content) { return "<B>" + content + "</B>"; }

static std::string html_table_start(const html_table_style& style)
{
    return R"(<<TABLE BORDER=")" + std::to_string(style.border) + R"(" CELLBORDER=")" +
           std::to_string(style.cellborder) + R"(" CELLPADDING=")" +
           std::to_string(style.cellpadding) + R"(" CELLSPACING=")" +
           std::to_string(style.cellspacing) + R"(" COLOR="transparent">)";
}

static std::string html_table_end() { return "</TABLE>>"; }

static std::string html_cell(const std::string& content, const std::string& align = "center")
{
    return "<TR ALIGN=\"" + align + "\"><TD>" + content + "</TD></TR>";
}

static bool is_hex_color(std::string color) { return not color.empty() and color[0] == '#'; }

std::string enclose_name(const std::string& name)
{
    return '"' + replace_string(name, "\"", "\\\"") + '"';
}

std::string format_shape_name(const migraphx::shape& s, const std::string& linebreak)
{
    if(s.sub_shapes().empty())
    {
        if(s.dynamic())
        {
            return "dynamic" + linebreak + s.type_string() + linebreak + "{" +
                   to_string_range(s.dyn_dims()) + "}";
        }
        return s.type_string() + linebreak + "{" + to_string_range(s.lens()) + "}, {" +
               to_string_range(s.strides()) + "}";
    }
    return "[" + to_string_range(s.sub_shapes()) + "]";
}

std::string build_html_label(const graphviz_node_content& content)
{
    std::ostringstream ss;

    ss << html_table_start(content.html_style);
    ss << html_cell(html_bold(content.title));

    std::for_each(content.body_lines.begin(),
                  content.body_lines.end(),
                  [&ss](const std::string& line) { ss << html_cell(line); });

    ss << html_table_end();
    return ss.str();
}

std::string build_node_style(const graphviz_node_style& node_style)
{
    std::ostringstream ss;
    ss << "style=\"" << node_style.style << "\" ";

    if(is_hex_color(node_style.fillcolor))
        ss << "fillcolor=\"" << node_style.fillcolor << "\" ";
    else
        ss << "fillcolor=" << node_style.fillcolor << " ";

    if(is_hex_color(node_style.fontcolor))
        ss << "fontcolor=\"" << node_style.fontcolor << "\" ";
    else
        ss << "fontcolor=" << node_style.fontcolor << " ";

    if(node_style.bordercolor.empty() or is_hex_color(node_style.bordercolor))
        ss << "color=\"" << node_style.bordercolor << "\" ";
    else
        ss << "color=" << node_style.bordercolor << " ";

    ss << "shape=" << node_style.shape << " ";
    ss << "fontname=" << node_style.fontname;
    return ss.str();
}

std::string get_graph_color(const instruction_ref& ins)
{
    const auto& op   = ins->get_operator();
    const auto& attr = op.attributes();

    bool context_free = is_context_free(op);
    bool alias        = not op.output_alias(to_shapes(ins->inputs())).empty();

    if(ins->can_eval())
    {
        return "#ADD8E6"; // lightblue
    }
    else if(attr.contains("pointwise"))
    {
        return "#9ACD32"; // yellowgreen
    }
    else if(starts_with(op.name(), "reduce"))
    {
        return "#90EE90"; // light green
    }
    else if(context_free and alias)
    {
        return "#98FB98"; // palegreen
    }
    else if(context_free and not alias)
    {
        return "#FFA500"; // orange
    }
    else if(not context_free and alias)
    {
        return "#EFBF04"; // gold
    }
    else if(attr.contains("fillcolor"))
    {
        return attr.at("fillcolor").to<std::string>();
    }
    else
    {
        return "#D3D3D3"; // lightgray
    }
}

graphviz_node_content get_node_content(const instruction_ref& ins)
{
    const auto& op         = ins->get_operator();
    const std::string name = ins->name();

    graphviz_node_content content;

    if(name == "@param") // for params, get typing information
    {
        content.title = name;
        content.body_lines.push_back(graphviz::format_shape_name(ins->get_shape(), "<BR/>"));

        content.html_style = {0, 0, 0, 0};
        content.node_style = {
            "#F0E68C" /* khaki */, "#000000" /* black */, "filled", "rectangle", "Helvectica"};
    }
    else if(name == "@literal") // for literals, just put @literal for name
    {
        content.title = name;

        content.html_style       = {0, 0, 0, 0};
        content.node_style.style = "filled";
        content.node_style.shape = "rectangle";
    }
    else if(name == "gpu::code_object") // use code_object_op::symbol_name for title
    {
        content.title = op.to_value()["symbol_name"].to<std::string>();
        content.body_lines.push_back(to_string(op));

        content.node_style.fillcolor = "#E9D66B"; // arylideyellow
    }
    else
    {
        // default case
        content.title = name;

        if(std::string op_to_string = to_string(op);
           name != op_to_string) // stops title == body, don't like doing compare
            content.body_lines.push_back(op_to_string);

        const auto& attr = op.attributes();
        if(attr.contains("style"))
            content.node_style.style = attr.at("style").to<std::string>();
        if(attr.contains("fontcolor"))
            content.node_style.fontcolor = attr.at("fontcolor").to<std::string>();
    }

    if(content.node_style.fillcolor.empty())
        content.node_style.fillcolor = get_graph_color(ins);

    return content;
}

} // namespace graphviz
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
