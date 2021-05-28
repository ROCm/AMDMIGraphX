#include <migraphx/json.hpp>
#include <migraphx/convert_to_json.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/gpu/compile_pointwise.hpp>

#include <unordered_map>
#include <functional>
#include <vector>
#include <string>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

[[noreturn]] void error(const std::string& msg)
{
    std::cout << msg << std::endl;
    std::abort();
}

struct jit_driver
{
    value settings = value::object{};
    std::unordered_map<std::string, std::function<void(const jit_driver&, const value&)>> actions =
        {};

    template <class F>
    void add_action(std::string name, F f)
    {
        actions[name] = std::mem_fn(f);
    }

    jit_driver() { add_action("compile_pointwise", &jit_driver::compile_pointwise_action); }

    void compile_pointwise_action(const value& v) const
    {
        auto op = compile_pointwise(parse_shapes(v.at("inputs")), v.at("lambda").to<std::string>());
        std::cout << op << std::endl;
    }

    template <class T>
    T get(const value& v, const std::string& key, const T& default_value) const
    {
        return v.get(key, settings.get(key, default_value));
    }

    shape parse_shape(const value& v) const
    {
        auto lens    = get(v, "lens", std::vector<std::size_t>{});
        auto strides = get(v, "strides", std::vector<std::size_t>{});
        auto type    = shape::parse_type(get<std::string>(v, "type", "float"));
        if(strides.empty())
            return shape{type, lens};
        else
            return shape{type, lens, strides};
    }

    std::vector<shape> parse_shapes(const value& v) const
    {
        std::vector<shape> result;
        std::transform(v.begin(), v.end(), std::back_inserter(result), [&](auto&& x) {
            return parse_shape(x);
        });
        return result;
    }

    void load_settings(const value& v)
    {
        if(v.contains("settings"))
            settings = v.at("settings");
    }

    static void process(const std::string& filename)
    {
        auto v = from_json_string(convert_to_json(read_string(filename)));
        if(not v.is_object())
            error("Input is not an object");
        jit_driver d{};
        d.load_settings(v);
        for(auto&& p : v)
        {
            if(p.get_key() == "settings")
                continue;
            d.actions.at(p.get_key())(d, p.without_key());
        }
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

int main(int argc, char const* argv[])
{
    std::vector<std::string> args(argv, argv + argc);
    if(args.size() < 2)
    {
        std::cout << "Usage: gpu-jit-driver <input-file>" << std::endl;
        std::abort();
    }
    migraphx::gpu::jit_driver::process(args[1]);
}
