#include <migraphx/serialize.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/literal.hpp>
#include <nlohmann/json.hpp>
#include <migraphx/json.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using json = nlohmann::json;

void value_to_json(const value& val, json& j);

template <class T>
void value_to_json(const T& x, json& j)
{
    j = x;
}

void value_to_json(const std::vector<value>& x, json& j)
{
    for(auto& v : x)
    {
        if(v.get_key().empty())
        {
            json jj;
            value_to_json(v, jj);
            j.push_back(jj);
        }
        // corresponding to an object
        else
        {
            value_to_json(v, j);
        }
    }
}

template <class T, class U>
void value_to_json(const std::pair<T, U>& x, json& j)
{
    json jj;
    value_to_json(x.second, jj);
    j[x.first] = jj;
}

void value_to_json(std::nullptr_t&, json& j) { j = {}; }

void value_to_json(const value& val, json& j)
{
    val.visit([&](auto v) { value_to_json(v, j); });
}

std::string to_json_string(const value& val)
{
    json j;
    value_to_json(val, j);
    std::string str = j.dump();
    return str;
}

bool to_value(const json& j, const std::string& key, migraphx::value& val)
{

    auto type = j.type();
    switch(type)
    {

    case json::value_t::null: val[key] = migraphx::value(nullptr); break;

#define CASE_TYPE(vt, cpp_type) \
    case json::value_t::vt: val[key] = j.get<cpp_type>(); break;

        CASE_TYPE(boolean, bool)
        CASE_TYPE(number_float, double)
        CASE_TYPE(number_integer, int64_t)
        CASE_TYPE(number_unsigned, uint64_t)
        CASE_TYPE(string, std::string)
#undef CASE_TYPE

    case json::value_t::array:
    case json::value_t::object:
    case json::value_t::binary:
    case json::value_t::discarded: return false;
    }

    return true;
}

bool to_value(const json& j, migraphx::value& val)
{
    auto type = j.type();
    switch(type)
    {
    case json::value_t::null: val.push_back(migraphx::value(nullptr)); break;

#define CASE_TYPE(vt, cpp_type) \
    case json::value_t::vt: val.push_back(j.get<cpp_type>()); break;

        CASE_TYPE(boolean, bool)
        CASE_TYPE(number_float, double)
        CASE_TYPE(number_integer, int64_t)
        CASE_TYPE(number_unsigned, uint64_t)
        CASE_TYPE(string, std::string)
#undef CASE_TYPE

    case json::value_t::array:
    case json::value_t::object:
    case json::value_t::binary:
    case json::value_t::discarded: return false;
    }

    return true;
}

void value_from_json(const json& j, migraphx::value& val)
{
    json::value_t type = j.type();
    switch(type)
    {
    case json::value_t::null: val = migraphx::value(nullptr); break;

    case json::value_t::object:
        for(auto item : j.items())
        {
            auto key = item.key();
            json v   = item.value();
            if(!to_value(v, key, val))
            {
                migraphx::value mv;
                value_from_json(v, mv);
                val[key] = mv;
            }
        }
        break;

    case json::value_t::array:
        for(auto& v : j)
        {
            if(!to_value(v, val))
            {
                migraphx::value mv;
                value_from_json(v, mv);
                val.push_back(mv);
            }
        }
        break;

#define CASE_TYPE(vt, cpp_type) \
    case json::value_t::vt: val = j.get<cpp_type>(); break;

        CASE_TYPE(boolean, bool)
        CASE_TYPE(number_float, double)
        CASE_TYPE(number_integer, int64_t)
        CASE_TYPE(number_unsigned, uint64_t)
        CASE_TYPE(string, std::string)
#undef CASE_TYPE

    case json::value_t::binary:
    case json::value_t::discarded: MIGRAPHX_THROW("Convert JSON to Value: type not supported!");
    }
}

migraphx::value from_json_string(const std::string& str)
{
    migraphx::value val;
    json j = json::parse(str);
    value_from_json(j, val);

    return val;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
