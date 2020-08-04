#include <migraphx/serialize.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/literal.hpp>
#include <nlohmann/json.hpp>
#include <migraphx/json.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using json = nlohmann::json;

void value_to_json(const value& val, json& j);
void to_json(json& j, const value& val);
void from_json(const json& j, value& val);

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
            j.push_back(v);
        }
        // in the case of an object
        else
        {
            j[v.get_key()] = v.without_key();
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

void to_json(json& j, const value& val) { 
    val.visit([&](auto v) { value_to_json(v, j); });    
}

std::string to_json_string(const value& val)
{
    json j = val;
    return j.dump();
}

migraphx::value value_from_json(const json& j)
{
    migraphx::value val;
    json::value_t type = j.type();
    switch(type)
    {
    case json::value_t::null: val = migraphx::value(); break;

    case json::value_t::boolean: val = j.get<bool>(); break;

    case json::value_t::number_float: val = j.get<double>(); break;

    case json::value_t::number_integer: val = j.get<int64_t>(); break;

    case json::value_t::number_unsigned: val = j.get<uint64_t>(); break;

    case json::value_t::string: val = j.get<std::string>(); break;

    case json::value_t::array:
        std::transform(j.begin(), j.end(), std::back_inserter(val), [&](const auto& jj) {
            return value_from_json(jj);
        });
        break;

    case json::value_t::object:
        for(const auto& item : j.items())
        {
            const auto& key = item.key();
            const json& v   = item.value();
            val[key]        = value_from_json(v);
        }
        break;

    case json::value_t::binary: MIGRAPHX_THROW("Convert JSON to Value: binary type not supported!");
    case json::value_t::discarded: MIGRAPHX_THROW("Convert JSON to Value: discarded type not supported!");
    }

    return val;
}

void from_json(const json& j, value& val) { val = value_from_json(j); }

migraphx::value from_json_string(const std::string& str)
{
    migraphx::value val;
    json j = json::parse(str);
    val    = value_from_json(j);

    return val;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
