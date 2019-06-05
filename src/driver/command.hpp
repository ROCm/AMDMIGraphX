#ifndef MIGRAPHX_GUARD_RTGLIB_COMMAND_HPP
#define MIGRAPHX_GUARD_RTGLIB_COMMAND_HPP

#include "argument_parser.hpp"

#include <migraphx/config.hpp>
#include <migraphx/type_name.hpp>
#include <migraphx/stringutils.hpp>

#include <unordered_map>
#include <utility>
#include <vector>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

inline auto& get_commands()
{
    static std::unordered_map<std::string, std::function<void(std::vector<std::string> args)>> m;
    return m;
}

template <class T>
std::string compute_command_name()
{
    static const std::string& tname = get_type_name<T>();
    auto name                       = tname.substr(tname.rfind("::") + 2);
    if(ends_with(name, "_command"))
        name = name.substr(0, name.size() - 8);
    if(ends_with(name, "_cmd"))
        name = name.substr(0, name.size() - 4);
    return name;
}

template <class T>
const std::string& command_name()
{
    static const std::string& name = compute_command_name<T>();
    return name;
}

template <class T>
void run_command(std::vector<std::string> args, bool add_help = false)
{
    T x;
    argument_parser ap;
    if(add_help)
        ap(nullptr, {"-h", "--help"}, ap.help("Show help"), ap.show_help());
    x.parse(ap);
    if(ap.parse(std::move(args)))
        return;
    x.run();
}

template <class T>
int auto_register_command()
{
    auto& m              = get_commands();
    m[command_name<T>()] = [](std::vector<std::string> args) { run_command<T>(args, true); };
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

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif

template <class T>
int command<T>::static_register = auto_register_command<T>(); // NOLINT

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx

#endif
