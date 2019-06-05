#ifndef MIGRAPHX_GUARD_RTGLIB_COMMAND_HPP
#define MIGRAPHX_GUARD_RTGLIB_COMMAND_HPP

#include "argument_parser.hpp"

#include <migraphx/config.hpp>
#include <migraphx/type_name.hpp>

#include <unordered_map>
#include <vector>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

inline auto& get_commands()
{
    static std::unordered_map<std::string, std::function<void(std::vector<std::string> args)>> m;
    return m;
}

template<class T>
std::string command_name()
{
    static const std::string& name = get_type_name<T>();
    return name.substr(name.rfind("::") + 2);
}

template <class T>
int auto_register_command()
{
    auto& m = get_commands();
    m[command_name<T>()] = [](std::vector<std::string> args) {
        T x;
        argument_parser ap;
        x.parse(ap);
        ap.parse(args);
        x.run();
    };
    return 0;
}

template <class T>
struct command
{
    static int static_register;
    // This typedef ensures that the static member will be instantiated if
    // the class itself is instantiated
    using static_register_type =
        std::integral_constant<decltype(&static_register), &static_register>;
};

template <class T>
int command<T>::static_register = auto_register_command<T>(); // NOLINT

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx

#endif
