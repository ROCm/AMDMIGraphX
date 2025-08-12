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

/* Operation color map:
    binary: #000000 , black (w/white font)
    unary: #CD5C5C , indianred
    convolution: #4682B4 , steelblue
    load: #1E90FF, dodger blue
    broadcast: #9ACD32, yellowgreen
    pointwise:  #9ACD32, yellowgreen
    slice: #FFA500, orange
    gpu::code_object_op: #E9D66B, arylideyellow
    pooling: #3CB371, mediumseagreen
    reduce: #8470FF, lightslateblue

    To add new colors for operations, add
    attributes to the value yourOp::attributes() const {...}
    member of your selected operation. See the graphviz_node_style
    struct for naming.

    For example:
        value rocm_op::attributes() const {{"fillcolor", "#EF0707"}}
    will set the fillcolor for the rocm_op node to be hex #ef0707
*/

/**
 * Struct for html-style table parameters created with
 * <TABLE>...<TABLE/>
 */
struct html_table_style
{
    int border      = 0;
    int cellborder  = 0;
    int cellpadding = 4;
    int cellspacing = 0;
};

/**
 * Struct for tracking graphviz node style using default
 * values for nodes with no attributes
 */
struct graphviz_node_style
{
    std::string fillcolor   = "";        // defaults to white
    std::string fontcolor   = "#000000"; // black
    std::string style     = "rounded,filled";
    std::string shape     = "none";
    std::string fontname  = "Helvetica";
    std::string bordercolor = ""; // defaults to none when shape is none
};

/**
 * Struct for storing all content and style settings
 * for a graphviz node. Stores title, lines of the body
 * text, perf_data (to be implemented), and above style
 * structs
 */
struct graphviz_node_content
{
    std::string title;
    std::vector<std::string> body_lines;
    std::optional<std::pair<double, double>> perf_data;

    html_table_style html_style{};
    graphviz_node_style node_style{};
};

/**
 * Escape all quotes in string
 */
std::string enclose_name(const std::string& name);

/**
 * Formats migraphx::shape for printing so we dont have to use the
 * migraphx::shape::operator<< which places content on one line
 */
std::string format_shape_name(const migraphx::shape& s, const std::string& linebreak = "\\n");

/**
 * Builds html-style table for content, given a graphviz_node_content struct
 */
std::string build_html_label(const graphviz_node_content& content);

/**
 * Builds the node style string given a graphviz_node_style struct
 */
std::string build_node_style(const graphviz_node_style& node_style);

/**
 * Given an instruction_ref ins we build a graphviz_node_content object to store
 * all of our necessary data. In this function we do formatting for specific
 * instructions: param, literal, and gpu::code_object
 */
graphviz_node_content get_node_content(const instruction_ref& ins);

/**
 * Given an instruction_ref ins, determine the coloring based on alias and
 * context-free qualities
 */
std::string get_graph_color(const instruction_ref& ins);

} // namespace graphviz
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
