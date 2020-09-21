#include "jsonize_attr_string.hpp"
#include <vector>
#include <stack>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <migraphx/errors.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// get all elements of an array or an object, including '[]' and '{}'
std::string get_elements_string(const std::string& str, const std::size_t start, const char brkt)
{
    if(str.empty())
    {
        return {};
    }

    const std::vector<std::pair<char, char>> brackets = {{'[', ']'}, {'{', '}'}};
    auto bit =
        std::find_if(brackets.begin(), brackets.end(), [=](auto p) { return p.first == brkt; });
    if(bit == brackets.end())
    {
        return {};
    }
    assert(str[start] == brkt);

    std::stack<char> sc;
    std::size_t i;
    for(i = start; i < str.length(); ++i)
    {
        auto c = str[i];
        if(c == bit->second)
        {
            while(!sc.empty())
            {
                auto c_out = sc.top();
                sc.pop();
                if(c_out == bit->first)
                {
                    break;
                }
            }

            if(sc.empty())
            {
                break;
            }
        }
        else
        {
            sc.push(c);
        }
    }

    if(i == str.length())
    {
        return {};
    }

    return str.substr(start, i - start + 1);
}

bool is_value_a_string(const std::string& op_name, const std::string& key)
{
    static std::unordered_map<std::string, std::unordered_set<std::string>> attr_str_value = {
        std::pair<std::string, std::unordered_set<std::string>>("pooling", {"mode"}),
        std::pair<std::string, std::unordered_set<std::string>>("rnn_val_sl_shift_output",
                                                                {"output_name"})};

    if(attr_str_value.count(op_name) > 0 and attr_str_value[op_name].count(key) > 0)
    {
        return true;
    }

    return false;
}

std::string add_quote(const std::string& input_str)
{
    if(input_str.empty())
    {
        MIGRAPHX_THROW("ADD_QUOTE: input string cannot be empty!");
    }

    std::string str = input_str;

    auto pp_start = str.find_first_not_of(' ');
    str.insert(pp_start, 1, '"');
    auto pp_end = str.find_last_not_of(' ');
    if(pp_end == str.size() - 1)
    {
        str.append(1, '"');
    }
    else
    {
        str.insert(pp_end + 1, 1, '"');
    }

    return str;
}

std::string get_key(std::string key_str)
{
    if(key_str.empty())
    {
        MIGRAPHX_THROW("GET_KEY: key string cannot be empty!");
    }

    auto pp_start = key_str.find_first_not_of(' ');
    key_str.erase(0, pp_start);
    auto pp_end = key_str.find_last_not_of(' ');
    if(pp_end < key_str.size())
    {
        key_str.erase(pp_end + 1);
    }

    return key_str;
}

std::string jsonize_attribute_string(const std::string& op_name, const std::string& str)
{
    if(str.empty())
    {
        return {};
    }

    assert(str.front() == '{');
    assert(str.back() == '}');

    std::string result = "{";
    std::size_t pos    = 1;
    while(pos < str.length())
    {
        // processing key, add quote before and after the key
        auto c_pos = str.find(':', pos);
        if(c_pos == std::string::npos)
        {
            MIGRAPHX_THROW("JSONIZE_ATTRIBUTE_STRING: cannot find end of key");
        }

        auto key_str = str.substr(pos, c_pos - pos);
        result.append(add_quote(key_str));
        result.append(": ");

        auto key = get_key(key_str);

        // if the value must be a string
        pos = c_pos + 1;
        if(is_value_a_string(op_name, key))
        {
            // get the next ',' charact and the substring
            // as the value
            auto v_end = str.find_first_of(',', pos);
            std::string val_str;
            if(v_end == std::string::npos)
            {
                val_str = str.substr(pos);
            }
            else
            {
                val_str = str.substr(pos, v_end - pos + 1);
            }
            result.append(add_quote(val_str));

            // reach string end, return
            if(v_end == std::string::npos)
            {
                result.append(1, '}');
                return result;
            }
            pos = v_end + 1;
        }
        // value can be another object or array or single element (not a string)
        // object need recursive processing, but array and single element donot need
        else
        {
            auto sp = str.find_first_not_of(' ', pos);
            if(sp == std::string::npos)
            {
                // no value, must be wrong
                MIGRAPHX_THROW("JSON_ATTRIBUTE_STRING: no value for key " + key + "!");
            }

            // value is another object, need recursive call
            if(str[sp] == '{')
            {
                auto obj_str      = get_elements_string(str, sp, '{');
                auto json_obj_str = jsonize_attribute_string(op_name, obj_str);
                if(!json_obj_str.empty())
                {
                    result.append(json_obj_str);
                }
                pos = sp + obj_str.length();
            }
            // value is an array
            else if(str[sp] == '[')
            {
                auto array_str = get_elements_string(str, sp, '[');
                // no array element are string, no additional processing
                result.append(array_str);
                pos = sp + array_str.length();
            }
            else
            {
                auto cp = str.find_first_of(',', pos);
                if(cp == std::string::npos)
                {
                    result.append(str.substr(pos));
                }
                else
                {
                    result.append(str.substr(pos, cp - pos));
                }
            }

            pos = str.find_first_of(',', pos);
            if(pos == std::string::npos)
            {
                result.append(1, '}');
                return result;
            }
            result.append(1, ',');

            ++pos;
        }
    }

    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
