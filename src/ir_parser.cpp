#include <migraphx/ir_parser.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/json.hpp>
#include <migraphx/value.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/generate.hpp>
#include <string_view>
#include <regex>


namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct simple_parser
{
    std::string_view buffer{};
    std::size_t pos = 0;
    bool skip_whitespace = true;

    // Remaining text as a view
    std::string_view peek() const
    {
        if(pos >= buffer.size()) return {};
        return buffer.substr(pos);
    }

    void advance(std::size_t n)
    {
        pos += n;
        if(pos > buffer.size())
            MIGRAPHX_THROW("Parser advanced past end of buffer");
        if(skip_whitespace)
        {
            auto it = std::find_if(buffer.begin() + pos, buffer.end(), [](char c) { return !std::isspace(c); });
            pos = std::distance(buffer.begin(), it);
        }
    }

    template <class Pred>
    std::string_view parse_while(Pred p)
    {
        // stop at first char that does NOT satisfy p
        auto tail = peek();
        auto it   = std::find_if(tail.begin(), tail.end(), [&](char c) { return !p(c); });
        auto n = std::distance(tail.begin(), it);
        advance(n);
        return tail.substr(0, n);
    }

    // Tries to match a regex at current position
    std::match_results<std::string_view::const_iterator>
    parse_match(const std::regex& re)
    {
        auto rem = peek();
        std::match_results<std::string_view::const_iterator> m;
        if(std::regex_search(rem.begin(), rem.end(), m, re,
                             std::regex_constants::match_continuous))
            advance(m[0].length());
        return m;
    }

    std::string_view parse_pairs(char start, char end)
    {
        auto tail = peek();
        if(not starts_with(std::string_view{&start, 1}))
            return {};
        auto it = find_matching_delimiter(tail.begin(), tail.end(), start, end);
        if(it == tail.end())
            MIGRAPHX_THROW(error_message("'" + std::string{end} + "'"));
        auto n = std::distance(tail.begin(), it);
        advance(n);
        if(n < 1)
            return {};
        return tail.substr(1, n - 2);
    }

    static bool is_ident_char(char c)
    {
        return std::isalnum(static_cast<unsigned char>(c)) or contains({'_', '@', ':', '-', '.'}, c);
    }
  
    bool starts_with(const std::string_view& str) const
    {
        return migraphx::starts_with(peek(), str);
    }
    
    void expect(const std::string_view& str)
    {
        if(not starts_with(str))
            MIGRAPHX_THROW(error_message("'" + std::string{str} + "'"));
        advance(str.size());
    }

    std::string_view expect_identifier()
    {
        auto id = parse_while(is_ident_char);
        if(id.empty())
            MIGRAPHX_THROW(error_message("identifier"));
        return id;
    }

    std::string error_message(std::string_view expected) const
    {
        return "Expected " + std::string(expected) + " at position " + std::to_string(pos) + " in '" + std::string(buffer) + "'";
    }
};

struct ir_mod_parser
{
    module_ref mod;
    std::unordered_map<std::string, instruction_ref> instructions{};

    static value parse_list(const std::string_view& str)
    {
        return from_json_string("[" + std::string(str) + "]");
    }

    static value parse_attr(std::string str)
    {
        replace_string_inplace(str, "=", ":");
        replace_string_inplace(str, "{", "[");
        replace_string_inplace(str, "}", "]");
        return from_json_string(convert_to_json("{" + str + "}"));
    }

    static shape parse_shape(simple_parser& parser)
    {
        auto t = parser.expect_identifier();
        parser.expect(",");
        auto dims_str = parser.parse_pairs('{', '}');
        auto dims = parse_list(dims_str);
        parser.expect(",");
        auto stride_str = parser.parse_pairs('{', '}');
        auto strides = parse_list(stride_str);
        return shape{shape::parse_type(t), dims.to_vector<std::size_t>(), strides.to_vector<std::size_t>()};
    }

    std::vector<instruction_ref> parse_args(simple_parser& parser)
    {
        std::vector<instruction_ref> args;
        if(parser.starts_with("("))
        {
            auto args_str = std::string{parser.parse_pairs('(', ')')};
            auto sargs = split_string(args_str, ',');
            std::transform(sargs.begin(), sargs.end(), std::back_inserter(args),
                           [&](const std::string& s) { return instructions.at(trim(s)); });
        }
        return args;
    }

    std::pair<std::string, instruction_ref> parse_instruction_line(const std::string& line) {
        simple_parser parser{line};
        auto id = std::string{parser.expect_identifier()};
        if(id == "@return")
        {
            auto args = parse_args(parser);
            if(args.empty())
                MIGRAPHX_THROW("Return instruction must have at least one argument");
            return std::make_pair(id, mod->add_return(args));
        }
        parser.expect("=");
        auto op_name = parser.expect_identifier();
        if(op_name == "@literal")
        {
            auto data = parser.parse_pairs('{', '}');
            parser.expect("->");
            auto s = parse_shape(parser);
            if(data == "...")
                return std::make_pair(id, mod->add_literal(generate_literal(s)));
            if(shape::is_integral(s.type()))
            {
                if(shape::is_unsigned(s.type()))
                    return std::make_pair(id, mod->add_literal(literal{s, parse_list(data).to_vector<uint64_t>()}));
                else
                    return std::make_pair(id, mod->add_literal(literal{s, parse_list(data).to_vector<int64_t>()}));
            }
            return std::make_pair(id, mod->add_literal(literal{s, parse_list(data).to_vector<double>()})); 
        }
        else if(op_name == "@param")
        {
            parser.expect(":");
            auto param_name = std::string{parser.expect_identifier()};
            parser.expect("->");
            auto s = parse_shape(parser);
            return std::make_pair(id, mod->add_parameter(param_name, s));

        }
        else
        {
            std::string_view attributes{};
            if(parser.starts_with("["))
            {
                attributes = parser.parse_pairs('[', ']');
            }
            auto args = parse_args(parser);
            auto op = make_op(std::string{op_name}, parse_attr(std::string{attributes}));
            return std::make_pair(id, mod->add_instruction(op, args));
        }
    }

    void parse_ir(const std::string& ir_text) {
        std::stringstream ss(ir_text);
        std::string line;
        
        while (std::getline(ss, line)) {
            line = trim(line);
            if (line.empty() or line[0] == '#')
                continue;
            auto [id, ins] = parse_instruction_line(line);
            instructions[id] = ins;
            if (id == "@return") 
                break;
        }
    }
};

program parse_text_ir(const std::string& ir_text) {
    program prog;
    auto* main_module = prog.get_main_module();
    
    ir_mod_parser parser{main_module};
    parser.parse_ir(ir_text);
    
    return prog;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
