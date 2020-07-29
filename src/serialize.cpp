#include <migraphx/serialize.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/literal.hpp>
#include <nlohmann/json.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class RawData>
void raw_data_to_value(value& v, const RawData& rd)
{
    value result;
    result["shape"] = migraphx::to_value(rd.get_shape());
    rd.visit([&](auto x) { result["data"] = std::vector<value>(x.begin(), x.end()); });
    v = result;
}

void migraphx_to_value(value& v, const literal& l) { raw_data_to_value(v, l); }
void migraphx_from_value(const value& v, literal& l)
{
    auto s = migraphx::from_value<shape>(v.at("shape"));
    s.visit_type([&](auto as) {
        using type = typename decltype(as)::type;
        l          = literal{s, v.at("data").to_vector<type>()};
    });
}

void migraphx_to_value(value& v, const argument& a) { raw_data_to_value(v, a); }
void migraphx_from_value(const value& v, argument& a)
{
    literal l = migraphx::from_value<literal>(v);
    a         = l.get_argument();
}

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

void value_to_json_string(const value& val, std::string& str)
{
    json j;
    value_to_json(val, j);
    str = j.dump();
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

void value_from_json_string(const std::string& str, value& val)
{
    json j = json::parse(str);
    value_from_json(j, val);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
