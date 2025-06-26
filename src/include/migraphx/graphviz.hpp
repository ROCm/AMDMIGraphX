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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace graphviz {

inline std::string enclose_name(const std::string& name)
{
    return '"' + replace_string(name, "\"", "\\\"") + '"';
}

inline std::string html_cell(const std::string& content, const std::string& align = "center")
{
    return "<TR><TD ALIGN=\"" + align + "\">" + content + "</TD></TR>";
}

inline std::string html_bold(const std::string& content)
{
    return "<B>" + content + "</B>";
}

inline std::string html_table_start()
{
    return "<<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLPADDING=\"4\" CELLSPACING=\"0\" COLOR=\"transparent\">";
}

inline std::string html_table_end() 
{
    return "</TABLE>>";
}

inline std::string html_color_style(const std::string& fill)
{
    return " style=\"rounded,filled\" fillcolor=\"" + fill + "\"";
}

inline std::string block_style(std::string color="lightgray")
{
    return " color=black fillcolor=" + color + " fontname=Helvetica shape=none style=\"rounded,filled\"";    
}

inline std::string format_shape_name(const migraphx::shape& s, bool labeled = false) 
{
    if(s.sub_shapes().empty())
    {
        if(s.dynamic())
        {
            return "dynamic\\n"  + s.type_string() + "\\n{" + to_string_range(s.dyn_dims()) + "}";
        }
        return s.type_string() +  "\\n{" + to_string_range(s.lens()) + "}, {" + to_string_range(s.strides()) + "}";
    }
    return "[" + to_string_range(s.sub_shapes()) + "]";
}

} // namespace graphviz
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif 
