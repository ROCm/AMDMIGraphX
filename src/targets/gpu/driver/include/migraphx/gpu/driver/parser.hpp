#ifndef MIGRAPHX_GUARD_GPU_DRIVER_PARSER_HPP
#define MIGRAPHX_GUARD_GPU_DRIVER_PARSER_HPP

#include <migraphx/value.hpp>
#include <migraphx/shape.hpp>

#include <unordered_map>
#include <functional>
#include <vector>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace driver {

[[noreturn]] void error(const std::string& msg);

struct parser
{
    parser() = default;

    template <class T>
    T get(const value& v, const std::string& key, const T& default_value) const
    {
        return v.get(key, settings.get(key, default_value));
    }

    shape parse_shape(const value& v) const;

    std::vector<shape> parse_shapes(const value& v) const;

    void load_settings(const value& v);

    static void process(const value& v);

    private:
    value settings = value::object{};
};

} // namespace driver
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_DRIVER_PARSER_HPP
