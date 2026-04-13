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
#include "dxgml_parser.hpp"
#include <migraphx/errors.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>

#include <string>
#include <sstream>
#include <vector>
#include <cctype>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// ---------------------------------------------------------------------------
// Text helpers
// ---------------------------------------------------------------------------

static std::string trim(const std::string& s)
{
    auto b = s.find_first_not_of(" \t\r\n");
    if(b == std::string::npos)
        return {};
    auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static bool starts_with(const std::string& s, const std::string& prefix)
{
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

// Strip a line-comment (// ...) from a string (respects quoted strings minimally).
static std::string strip_line_comment(const std::string& s)
{
    bool in_str = false;
    for(std::size_t i = 0; i < s.size(); ++i)
    {
        if(s[i] == '"')
            in_str = !in_str;
        if(!in_str && s[i] == '/' && i + 1 < s.size() && s[i + 1] == '/')
            return s.substr(0, i);
    }
    return s;
}

// Find the matching closing delimiter for the opening bracket at position `start`.
// Handles nested delimiters of the same type (e.g., nested {}).
// open_ch / close_ch: e.g., '{'/'}' or '('/')' or '<'/'>'.
static std::size_t find_matching(const std::string& s,
                                 std::size_t start,
                                 char open_ch,
                                 char close_ch)
{
    int depth = 0;
    for(std::size_t i = start; i < s.size(); ++i)
    {
        if(s[i] == open_ch)
            ++depth;
        else if(s[i] == close_ch)
        {
            --depth;
            if(depth == 0)
                return i;
        }
    }
    return std::string::npos;
}

// ---------------------------------------------------------------------------
// Type parsing
// ---------------------------------------------------------------------------

// "!dxgml.float32" | "f32" | "!dxgml.float16" | "f16" | ...
shape::type_t dxgml_parser::parse_element_type(const std::string& raw) const
{
    std::string e = trim(raw);
    if(e == "!dxgml.float32" || e == "f32")   return shape::float_type;
    if(e == "!dxgml.float16" || e == "f16")   return shape::half_type;
    if(e == "!dxgml.bfloat16" || e == "bf16") return shape::bf16_type;
    if(e == "!dxgml.float64" || e == "f64")   return shape::double_type;
    if(e == "!dxgml.int8"  || e == "i8")      return shape::int8_type;
    if(e == "!dxgml.int16" || e == "i16")     return shape::int16_type;
    if(e == "!dxgml.int32" || e == "i32")     return shape::int32_type;
    if(e == "!dxgml.int64" || e == "i64")     return shape::int64_type;
    if(e == "!dxgml.uint8"  || e == "ui8")    return shape::uint8_type;
    if(e == "!dxgml.uint16" || e == "ui16")   return shape::uint16_type;
    if(e == "!dxgml.uint32" || e == "ui32")   return shape::uint32_type;
    if(e == "!dxgml.uint64" || e == "ui64")   return shape::uint64_type;
    if(e == "!dxgml.bool")                    return shape::bool_type;
    MIGRAPHX_THROW("DxGML: unsupported element type: " + e);
}

// Parse "!dxgml.tensor<AxBx...x!dxgml.float16>" or "!dxgml.tensor<1x512x3000x!dxgml.float16>"
// Returns the corresponding MIGraphX shape.
shape dxgml_parser::parse_tensor_type(const std::string& ts) const
{
    // Find '<' ... '>'
    auto lt = ts.find('<');
    auto gt = ts.rfind('>');
    if(lt == std::string::npos || gt == std::string::npos || gt <= lt)
        MIGRAPHX_THROW("DxGML: malformed tensor type: " + ts);

    std::string inner = ts.substr(lt + 1, gt - lt - 1);

    // Tokenise by 'x', but only when the token so far doesn't start with '!'
    // (element type tokens start with '!').
    std::vector<std::string> tokens;
    std::string tok;
    for(char c : inner)
    {
        if(c == 'x' && !tok.empty() && tok.find('!') == std::string::npos)
        {
            tokens.push_back(tok);
            tok.clear();
        }
        else
        {
            tok += c;
        }
    }
    if(!tok.empty())
        tokens.push_back(tok);

    if(tokens.size() < 2)
        MIGRAPHX_THROW("DxGML: cannot parse tensor type: " + ts);

    std::string elem = tokens.back();
    tokens.pop_back();

    std::vector<std::size_t> lens;
    for(const auto& t : tokens)
        lens.push_back(static_cast<std::size_t>(std::stoull(trim(t))));

    return shape{parse_element_type(elem), lens};
}

// ---------------------------------------------------------------------------
// Attribute helpers
// ---------------------------------------------------------------------------

// Extract the value string for a named key from an attribute block.
// The block is the raw text between { } of an op's attribute section.
// Returns empty string if key not found.
std::string dxgml_parser::get_attr_str(const std::string& block, const std::string& key) const
{
    // Search for "key = " pattern
    std::size_t pos = 0;
    while(pos < block.size())
    {
        auto kp = block.find(key, pos);
        if(kp == std::string::npos)
            break;

        // Ensure this is a standalone key (not part of another word)
        if(kp > 0 && (std::isalnum(static_cast<unsigned char>(block[kp - 1])) || block[kp - 1] == '_'))
        {
            pos = kp + 1;
            continue;
        }

        // Find the '=' after the key
        auto eq = block.find_first_not_of(" \t", kp + key.size());
        if(eq == std::string::npos || block[eq] != '=')
        {
            pos = kp + 1;
            continue;
        }

        // Value starts after '='
        auto vs = block.find_first_not_of(" \t", eq + 1);
        if(vs == std::string::npos)
            return {};

        // Determine the end of the value: stop at ',' or end of block,
        // respecting nesting of <>, {}, ().
        std::string val;
        int d_angle = 0, d_brace = 0, d_paren = 0;
        std::size_t i = vs;
        while(i < block.size())
        {
            char c = block[i];
            if(c == '<')  ++d_angle;
            else if(c == '>') { if(d_angle > 0) --d_angle; else break; }
            else if(c == '{')  ++d_brace;
            else if(c == '}') { if(d_brace > 0) --d_brace; else break; }
            else if(c == '(')  ++d_paren;
            else if(c == ')') { if(d_paren > 0) --d_paren; else break; }
            else if(c == ',' && d_angle == 0 && d_brace == 0 && d_paren == 0)
                break;
            val += c;
            ++i;
        }
        return trim(val);
    }
    return {};
}

// Parse "#dxgml.dense_integer_elements<[2, 2]> : !dxgml.tensor<2x!dxgml.int64>"
// or "#dxgml.dense_integer_elements<[1, 1]> : ..."
// Returns the vector of integers.
std::vector<std::size_t> dxgml_parser::parse_dense_int_vec(const std::string& s) const
{
    auto lb = s.find('[');
    auto rb = s.find(']');
    if(lb == std::string::npos || rb == std::string::npos)
        MIGRAPHX_THROW("DxGML: cannot parse dense int vec: " + s);

    std::string inner = s.substr(lb + 1, rb - lb - 1);
    std::vector<std::size_t> out;
    std::size_t pos = 0;
    while(pos < inner.size())
    {
        while(pos < inner.size() && (inner[pos] == ' ' || inner[pos] == ','))
            ++pos;
        if(pos >= inner.size())
            break;
        std::size_t end = pos;
        while(end < inner.size() && inner[end] != ',' && inner[end] != ' ')
            ++end;
        if(end > pos)
            out.push_back(static_cast<std::size_t>(std::stoull(inner.substr(pos, end - pos))));
        pos = end;
    }
    return out;
}

// Parse "#dxgml.integer<-1 : !dxgml.int64>" or bare "-1 : si64"
int64_t dxgml_parser::parse_int_scalar(const std::string& s) const
{
    std::string ts = trim(s);
    // Format 1: #dxgml.integer<N : ...>
    auto lt = ts.find('<');
    if(lt != std::string::npos)
    {
        auto colon = ts.find(':', lt);
        if(colon != std::string::npos)
        {
            std::string val = trim(ts.substr(lt + 1, colon - lt - 1));
            return std::stoll(val);
        }
    }
    // Format 2: "-1 : si64" or just "-1"
    auto colon = ts.find(':');
    std::string val = trim(colon != std::string::npos ? ts.substr(0, colon) : ts);
    if(!val.empty())
    {
        try { return std::stoll(val); } catch(...) {}
    }
    MIGRAPHX_THROW("DxGML: cannot parse int scalar: " + s);
}

// Parse "#dxgml.float<1.0e-05 : !dxgml.float64>" or "#dxgml.float<6.000000e+00 : !dxgml.float32>"
double dxgml_parser::parse_float_scalar(const std::string& s) const
{
    auto lt = s.find('<');
    auto colon = s.find(':', lt != std::string::npos ? lt : 0);
    if(lt == std::string::npos || colon == std::string::npos)
        MIGRAPHX_THROW("DxGML: cannot parse float scalar: " + s);
    std::string val = trim(s.substr(lt + 1, colon - lt - 1));
    return std::stod(val);
}

// ---------------------------------------------------------------------------
// Operand list parser: "(%arg0, %_conv1.weight, %_conv1.bias)"
// Returns vector of SSA names without '%'.
// ---------------------------------------------------------------------------
static std::vector<std::string> parse_operand_names(const std::string& ops_str)
{
    // ops_str is the content of parentheses (without the parens themselves)
    std::vector<std::string> names;
    std::size_t pos = 0;
    while(pos < ops_str.size())
    {
        auto pct = ops_str.find('%', pos);
        if(pct == std::string::npos)
            break;
        // Name runs until ',', ' ', ')', or end
        auto end = pct + 1;
        while(end < ops_str.size() &&
              ops_str[end] != ',' && ops_str[end] != ' ' &&
              ops_str[end] != ')' && ops_str[end] != '(')
            ++end;
        names.push_back(ops_str.substr(pct + 1, end - pct - 1));
        pos = end;
    }
    return names;
}

// ---------------------------------------------------------------------------
// Entry point argument list parser:
// "(%arg0: !dxgml.tensor<1x4x2160x3840x!dxgml.float16>)"
// ---------------------------------------------------------------------------
struct ArgInfo
{
    std::string name;  // without '%'
    std::string type;  // full type string e.g. "!dxgml.tensor<...>"
};

static std::vector<ArgInfo> parse_arg_list(const std::string& arg_list)
{
    // arg_list is the text between '(' and ')' of the entry_point signature
    std::vector<ArgInfo> args;
    std::size_t pos = 0;
    while(pos < arg_list.size())
    {
        auto pct = arg_list.find('%', pos);
        if(pct == std::string::npos)
            break;

        // Name: from '%' to ':'
        auto colon = arg_list.find(':', pct);
        if(colon == std::string::npos)
            break;
        std::string name = trim(arg_list.substr(pct + 1, colon - pct - 1));

        // Type: after ':', until next ',' or end (respecting <> nesting)
        auto ts = arg_list.find_first_not_of(" \t", colon + 1);
        if(ts == std::string::npos)
            break;

        int depth = 0;
        std::size_t te = ts;
        while(te < arg_list.size())
        {
            char c = arg_list[te];
            if(c == '<')  ++depth;
            else if(c == '>') { --depth; if(depth < 0) break; }
            else if(c == ',' && depth == 0) break;
            ++te;
        }
        std::string type = trim(arg_list.substr(ts, te - ts));
        args.push_back({name, type});
        pos = te + 1;
    }
    return args;
}

// ---------------------------------------------------------------------------
// Top-level text preprocessor: extract entry_point sig + body
// ---------------------------------------------------------------------------

struct EntryPointData
{
    std::string arg_list; // text between '(' and ')' of the signature
    std::string ret_type; // return type string (after '->')
    std::string body;     // body block text (between the matching braces)
    bool found = false;
};

static EntryPointData extract_entry_point(const std::string& src)
{
    EntryPointData data;

    // Strip {-# dialect_resources ... #-} at the end
    std::string text = src;
    auto res_pos = text.find("{-#");
    if(res_pos != std::string::npos)
        text.resize(res_pos);

    // Find "dxgml.entry_point"
    auto ep_pos = text.find("dxgml.entry_point");
    if(ep_pos == std::string::npos)
        return data;

    // Extract argument list between first '(' and matching ')'
    auto paren_open = text.find('(', ep_pos);
    if(paren_open == std::string::npos)
        return data;

    auto paren_close = find_matching(text, paren_open, '(', ')');
    if(paren_close == std::string::npos)
        return data;

    data.arg_list = text.substr(paren_open + 1, paren_close - paren_open - 1);

    // Extract return type (after '->')
    auto arrow = text.find("->", paren_close);
    if(arrow != std::string::npos)
    {
        // Return type runs from after '->' to the next '{' or 'attributes'
        auto rt_start = text.find_first_not_of(" \t", arrow + 2);
        if(rt_start != std::string::npos)
        {
            // May have 'attributes { ... }' before the body '{'
            auto rt_end = rt_start;
            int depth = 0;
            while(rt_end < text.size())
            {
                char c = text[rt_end];
                if(c == '<')  ++depth;
                else if(c == '>') --depth;
                else if(c == '{' && depth == 0) break;
                ++rt_end;
            }
            // Strip optional 'attributes {...}' from the ret_type string
            std::string rt_raw = trim(text.substr(rt_start, rt_end - rt_start));
            auto attr_kw = rt_raw.find("attributes");
            if(attr_kw != std::string::npos)
                rt_raw = trim(rt_raw.substr(0, attr_kw));
            data.ret_type = rt_raw;
        }
    }

    // Find the body { ... } — the first '{' after paren_close
    auto open_brace = text.find('{', paren_close);
    if(open_brace == std::string::npos)
        return data;

    // Skip optional 'attributes { ... }' block that may precede the body
    // by checking if 'attributes' appears between paren_close and open_brace
    {
        std::string between = text.substr(paren_close + 1, open_brace - paren_close - 1);
        if(between.find("attributes") != std::string::npos)
        {
            // This '{' is the attributes block — find the body '{' after it
            auto attr_close = find_matching(text, open_brace, '{', '}');
            if(attr_close == std::string::npos)
                return data;
            open_brace = text.find('{', attr_close + 1);
            if(open_brace == std::string::npos)
                return data;
        }
    }

    auto close_brace = find_matching(text, open_brace, '{', '}');
    if(close_brace == std::string::npos)
        return data;

    data.body  = text.substr(open_brace + 1, close_brace - open_brace - 1);
    data.found = true;
    return data;
}

// ---------------------------------------------------------------------------
// Op line parser
//
// Handles these forms:
//   %name = dxgml_op.relu(%arg0) : (TYPE) -> TYPE
//   %name = dxgml_op.convolution(%a, %b, %c) { attrs } : (TYPES) -> TYPE
//   %name = dxgml_op.constant(#dxgml.constant_resource<...>)
//   dxgml.return %name : TYPE
// ---------------------------------------------------------------------------

void dxgml_parser::parse_entry_point(const std::string& arg_list_str, const std::string& body)
{
    // Register block arguments as parameters.
    // module::add_parameter uses insert_parameter(begin(),...) which prepends,
    // so add args in reverse order so that arg0 ends up at position 0.
    auto args = parse_arg_list(arg_list_str);
    for(std::size_t i = args.size(); i-- > 0;)
    {
        const auto& arg = args[i];
        shape s             = parse_tensor_type(arg.type);
        auto param_name     = "arg" + std::to_string(i);
        instruction_ref ir  = mm->add_parameter(param_name, s);
        value_map[arg.name] = ir;
    }

    // Walk body lines.
    // We need to handle multi-line ops (attribute blocks span multiple lines).
    // Strategy: concatenate the body into one long string and scan for ops.
    std::string flat;
    {
        // Strip comments, join lines
        std::istringstream ss(body);
        std::string line;
        while(std::getline(ss, line))
        {
            line = strip_line_comment(line);
            flat += " " + line;
        }
    }

    // Scan for SSA assignments and dxgml.return
    std::size_t pos = 0;
    while(pos < flat.size())
    {
        // Skip whitespace
        while(pos < flat.size() && std::isspace(static_cast<unsigned char>(flat[pos])))
            ++pos;
        if(pos >= flat.size())
            break;

        // Check for dxgml.return
        const std::string ret_kw = "dxgml.return";
        if(flat.compare(pos, ret_kw.size(), ret_kw) == 0)
        {
            // Collect operands from the rest of the statement
            auto end = flat.find_first_of(";", pos);
            std::string stmt = flat.substr(pos, end != std::string::npos ? end - pos : std::string::npos);
            // Extract %name tokens
            auto names = parse_operand_names(stmt);
            std::vector<instruction_ref> rets;
            for(const auto& n : names)
            {
                auto it = value_map.find(n);
                if(it == value_map.end())
                    MIGRAPHX_THROW("DxGML: undefined SSA value in dxgml.return: %" + n);
                rets.push_back(it->second);
            }
            mm->add_return(rets);
            break; // return terminates the function
        }

        // func.return (from preprocessing — shouldn't appear now but be safe)
        const std::string fret_kw = "func.return";
        if(flat.compare(pos, fret_kw.size(), fret_kw) == 0)
        {
            auto end = flat.find_first_of(";", pos);
            std::string stmt = flat.substr(pos, end != std::string::npos ? end - pos : std::string::npos);
            auto names = parse_operand_names(stmt);
            std::vector<instruction_ref> rets;
            for(const auto& n : names)
            {
                auto it = value_map.find(n);
                if(it == value_map.end())
                    MIGRAPHX_THROW("DxGML: undefined SSA value in return: %" + n);
                rets.push_back(it->second);
            }
            mm->add_return(rets);
            break;
        }

        // Check for SSA assignment: %name = dxgml_op.something(...)
        if(flat[pos] == '%')
        {
            // Find the '='
            auto eq = flat.find('=', pos);
            if(eq == std::string::npos)
                break;

            // Result name (without %)
            std::string result_name = trim(flat.substr(pos + 1, eq - pos - 1));

            // Op starts after '='
            auto op_start = flat.find_first_not_of(" \t", eq + 1);
            if(op_start == std::string::npos)
                break;

            // Op name runs to '(' or ' '
            auto op_end = op_start;
            while(op_end < flat.size() && flat[op_end] != '(' && !std::isspace(static_cast<unsigned char>(flat[op_end])))
                ++op_end;
            std::string full_op_name = flat.substr(op_start, op_end - op_start);

            // Skip non-dxgml_op ops (e.g. func.func, etc.)
            const std::string dxgml_op_pfx = "dxgml_op.";
            if(!starts_with(full_op_name, dxgml_op_pfx))
            {
                // Skip to next semicolon or next '%'
                pos = op_end;
                // advance to next statement
                while(pos < flat.size() && flat[pos] != '%' && flat[pos] != ';')
                    ++pos;
                continue;
            }
            std::string op_name = full_op_name.substr(dxgml_op_pfx.size());

            // After op name, find '('
            auto paren_open = flat.find('(', op_end);
            if(paren_open == std::string::npos)
                break;

            auto paren_close = find_matching(flat, paren_open, '(', ')');
            if(paren_close == std::string::npos)
                break;

            std::string operands_raw = flat.substr(paren_open + 1, paren_close - paren_open - 1);

            // After ')': optional attribute block { ... } and type signature
            std::size_t after_paren = paren_close + 1;

            // Skip whitespace and newlines
            while(after_paren < flat.size() && std::isspace(static_cast<unsigned char>(flat[after_paren])))
                ++after_paren;

            std::string attrs_block;
            std::string type_sig;

            if(after_paren < flat.size() && flat[after_paren] == '{')
            {
                auto brace_close = find_matching(flat, after_paren, '{', '}');
                if(brace_close != std::string::npos)
                {
                    attrs_block = flat.substr(after_paren + 1, brace_close - after_paren - 1);
                    after_paren = brace_close + 1;
                }
            }

            // Optional type signature: ": (types) -> rettype"
            // Only present when ':' precedes the next statement's '%' at depth 0.
            // We scan forward from after_paren, tracking bracket depth, to find
            // whether ':' or '%' (start of next stmt) comes first.
            pos = after_paren;
            if(after_paren < flat.size())
            {
                // Find first ':' and first '%' from after_paren, at bracket depth 0
                std::size_t first_colon = std::string::npos;
                std::size_t first_pct   = std::string::npos;
                int scan_depth = 0;
                for(std::size_t si = after_paren; si < flat.size(); ++si)
                {
                    char c = flat[si];
                    if(c == '(' || c == '<')  ++scan_depth;
                    else if(c == ')' || c == '>') { if(scan_depth > 0) --scan_depth; }
                    if(scan_depth == 0)
                    {
                        if(c == ':' && first_colon == std::string::npos)
                        {
                            first_colon = si;
                            break; // we have what we need — stop here
                        }
                        if(c == '%' && first_pct == std::string::npos)
                        {
                            first_pct = si;
                            break; // '%' before ':' means no type sig
                        }
                        if(flat.compare(si, 12, "dxgml.return") == 0 ||
                           flat.compare(si, 11, "func.return") == 0)
                        {
                            first_pct = si; // treat return as end-of-stmt
                            break;
                        }
                    }
                }

                if(first_colon != std::string::npos &&
                   (first_pct == std::string::npos || first_colon < first_pct))
                {
                    // Type sig present: extract from colon+1 until next '%' or return kw
                    auto ts_start = first_colon + 1;
                    int depth = 0;
                    std::size_t te = ts_start;
                    while(te < flat.size())
                    {
                        char c = flat[te];
                        // '->' arrow: skip both chars, don't change depth
                        if(c == '-' && te + 1 < flat.size() && flat[te + 1] == '>')
                        {
                            te += 2;
                            continue;
                        }
                        if(c == '(' || c == '<')  ++depth;
                        else if(c == ')') { if(depth > 0) --depth; else break; }
                        else if(c == '>') { if(depth > 0) --depth; }
                        else if(depth == 0 && c == '%') break;
                        else if(depth == 0 &&
                                (flat.compare(te, 12, "dxgml.return") == 0 ||
                                 flat.compare(te, 11, "func.return") == 0)) break;
                        ++te;
                    }
                    type_sig = trim(flat.substr(ts_start, te - ts_start));
                    pos = te;
                }
                // else: no type sig — pos stays at after_paren (already set above)
            }

            // Dispatch to op handler.
            // Multi-result ops (e.g. split) use "name:N" syntax — parse_dxgml_op
            // registers all N results into value_map itself and returns the first.
            auto colon_pos = result_name.find(':');
            int num_results = 1;
            std::string base_name = result_name;
            if(colon_pos != std::string::npos)
            {
                num_results = std::stoi(result_name.substr(colon_pos + 1));
                base_name   = result_name.substr(0, colon_pos);
            }

            instruction_ref result =
                parse_dxgml_op(op_name, operands_raw, attrs_block, type_sig,
                               base_name, num_results);
            // Single-result: register under the result name (or base_name for :N ops
            // where parse_dxgml_op already registered the individual results).
            if(num_results == 1)
                value_map[result_name] = result;
            continue;
        }

        // Not a recognized statement — advance past it
        ++pos;
    }
}

// ---------------------------------------------------------------------------
// Top-level parse
// ---------------------------------------------------------------------------

void dxgml_parser::parse_from_string(const std::string& mlir_text)
{
    mm = prog.get_main_module();

    auto ep = extract_entry_point(mlir_text);
    if(!ep.found)
        MIGRAPHX_THROW("DxGML: no dxgml.entry_point found in input");

    parse_entry_point(ep.arg_list, ep.body);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
