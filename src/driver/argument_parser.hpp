#ifndef MIGRAPHX_GUARD_RTGLIB_ARGUMENT_PARSER_HPP
#define MIGRAPHX_GUARD_RTGLIB_ARGUMENT_PARSER_HPP

#include <algorithm>
#include <functional>
#include <iostream>
#include <list>
#include <set>
#include <string>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <migraphx/config.hpp>
#include <migraphx/requires.hpp>
#include <migraphx/type_name.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/rank.hpp>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

#ifdef MIGRAPHX_USE_CLANG_TIDY
#define MIGRAPHX_DRIVER_STATIC
#else
#define MIGRAPHX_DRIVER_STATIC static
#endif

template <class T>
using bare = std::remove_cv_t<std::remove_reference_t<T>>;

namespace detail {

template <class T>
auto is_container(int, T&& x) -> decltype(x.insert(x.end(), *x.begin()), std::true_type{});

template <class T>
std::false_type is_container(float, T&&);

} // namespace detail

template <class T>
struct is_container : decltype(detail::is_container(int(0), std::declval<T>()))
{
};

template <class T>
using is_multi_value =
    std::integral_constant<bool, (is_container<T>{} and not std::is_convertible<T, std::string>{})>;

enum class color
{
    reset      = 0,
    bold       = 1,
    underlined = 4,
    fg_red     = 31,
    fg_green   = 32,
    fg_yellow  = 33,
    fg_blue    = 34,
    fg_default = 39,
    bg_red     = 41,
    bg_green   = 42,
    bg_yellow  = 43,
    bg_blue    = 44,
    bg_default = 49
};
inline std::ostream& operator<<(std::ostream& os, const color& c)
{
#ifndef _WIN32
    static const bool use_color = isatty(STDOUT_FILENO) != 0;
    if(use_color)
        return os << "\033[" << static_cast<std::size_t>(c) << "m";
#endif
    return os;
}

inline std::string colorize(color c, const std::string& s)
{
    std::stringstream ss;
    ss << c << s << color::reset;
    return ss.str();
}

template <class T>
struct type_name
{
    static const std::string& apply() { return migraphx::get_type_name<T>(); }
};

template <>
struct type_name<std::string>
{
    static const std::string& apply()
    {
        static const std::string name = "std::string";
        return name;
    }
};

template <class T>
struct type_name<std::vector<T>>
{
    static const std::string& apply()
    {
        static const std::string name = "std::vector<" + type_name<T>::apply() + ">";
        return name;
    }
};

template <class T>
struct value_parser
{
    template <MIGRAPHX_REQUIRES(not std::is_enum<T>{} and not is_multi_value<T>{})>
    static T apply(const std::string& x)
    {
        T result;
        std::stringstream ss;
        ss.str(x);
        ss >> result;
        if(ss.fail())
            throw std::runtime_error("Failed to parse '" + x + "' as " + type_name<T>::apply());
        return result;
    }

    template <MIGRAPHX_REQUIRES(std::is_enum<T>{} and not is_multi_value<T>{})>
    static T apply(const std::string& x)
    {
        std::ptrdiff_t i;
        std::stringstream ss;
        ss.str(x);
        ss >> i;
        if(ss.fail())
            throw std::runtime_error("Failed to parse '" + x + "' as " + type_name<T>::apply());
        return static_cast<T>(i);
    }

    template <MIGRAPHX_REQUIRES(is_multi_value<T>{} and not std::is_enum<T>{})>
    static T apply(const std::string& x)
    {
        T result;
        using value_type = typename T::value_type;
        result.insert(result.end(), value_parser<value_type>::apply(x));
        return result;
    }
};

struct argument_parser
{
    struct argument
    {
        std::vector<std::string> flags;
        std::function<bool(argument_parser&, const std::vector<std::string>&)> action{};
        std::string type          = "";
        std::string help          = "";
        std::string metavar       = "";
        std::string default_value = "";
        unsigned nargs            = 1;
    };

    template <class T, MIGRAPHX_REQUIRES(is_multi_value<T>{})>
    std::string as_string_value(const T& x)
    {
        return to_string_range(x);
    }

    template <class T>
    auto as_string_value(rank<1>, const T& x) -> decltype(to_string(x))
    {
        return to_string(x);
    }

    template <class T>
    std::string as_string_value(rank<0>, const T&)
    {
        throw std::runtime_error("Can't convert to string");
    }

    template <class T, MIGRAPHX_REQUIRES(not is_multi_value<T>{})>
    std::string as_string_value(const T& x)
    {
        return as_string_value(rank<1>{}, x);
    }

    template <class T, class... Fs>
    void operator()(T& x, const std::vector<std::string>& flags, Fs... fs)
    {
        arguments.push_back({flags, [&](auto&&, const std::vector<std::string>& params) {
                                 if(params.empty())
                                     throw std::runtime_error("Flag with no value.");
                                 x = value_parser<T>::apply(params.back());
                                 return false;
                             }});

        argument& arg = arguments.back();
        arg.type      = type_name<T>::apply();
        migraphx::each_args([&](auto f) { f(x, arg); }, fs...);
        if(not arg.default_value.empty() and arg.nargs > 0)
            arg.default_value = as_string_value(x);
    }

    template <class... Fs>
    void operator()(std::nullptr_t x, std::vector<std::string> flags, Fs... fs)
    {
        arguments.push_back({std::move(flags)});

        argument& arg = arguments.back();
        arg.type      = "";
        arg.nargs     = 0;
        migraphx::each_args([&](auto f) { f(x, arg); }, fs...);
    }

    MIGRAPHX_DRIVER_STATIC auto nargs(unsigned n = 1)
    {
        return [=](auto&&, auto& arg) { arg.nargs = n; };
    }

    template <class F>
    MIGRAPHX_DRIVER_STATIC auto write_action(F f)
    {
        return [=](auto& x, auto& arg) {
            arg.action = [&, f](auto& self, const std::vector<std::string>& params) {
                f(self, x, params);
                return false;
            };
        };
    }

    template <class F>
    MIGRAPHX_DRIVER_STATIC auto do_action(F f)
    {
        return [=](auto&, auto& arg) {
            arg.nargs  = 0;
            arg.action = [&, f](auto& self, const std::vector<std::string>&) {
                f(self);
                return true;
            };
        };
    }

    MIGRAPHX_DRIVER_STATIC auto append()
    {
        return write_action([](auto&, auto& x, auto& params) {
            using type = typename bare<decltype(params)>::value_type;
            std::transform(params.begin(),
                           params.end(),
                           std::inserter(x, x.end()),
                           [](std::string y) { return value_parser<type>::apply(y); });
        });
    }

    template <class F>
    argument* find_argument(F f)
    {
        auto it = std::find_if(arguments.begin(), arguments.end(), f);
        if(it == arguments.end())
            return nullptr;
        return std::addressof(*it);
    }

    MIGRAPHX_DRIVER_STATIC auto show_help(const std::string& msg = "")
    {
        return do_action([=](auto& self) {
            argument* input_argument =
                self.find_argument([](const auto& arg) { return arg.flags.empty(); });
            std::cout << color::fg_yellow << "USAGE:" << color::reset << std::endl;
            std::cout << "    " << self.exe_name << " <options> ";
            if(input_argument)
                std::cout << input_argument->metavar;
            std::cout << std::endl;
            std::cout << std::endl;
            if(self.find_argument([](const auto& arg) { return arg.nargs == 0; }))
            {
                std::cout << color::fg_yellow << "FLAGS:" << color::reset << std::endl;
                std::cout << std::endl;
                for(auto&& arg : self.arguments)
                {
                    if(arg.nargs != 0)
                        continue;
                    const int col_align = 35;
                    std::string prefix  = "    ";
                    int len             = 0;
                    std::cout << color::fg_green;
                    for(const std::string& a : arg.flags)
                    {
                        len += prefix.length() + a.length();
                        std::cout << prefix;
                        std::cout << a;
                        prefix = ", ";
                    }
                    std::cout << color::reset;
                    int spaces = col_align - len;
                    if(spaces < 0)
                    {
                        std::cout << std::endl;
                    }
                    else
                    {
                        for(int i = 0; i < spaces; i++)
                            std::cout << " ";
                    }
                    std::cout << arg.help << std::endl;
                }
                std::cout << std::endl;
            }
            if(self.find_argument([](const auto& arg) { return arg.nargs != 0; }))
            {
                std::cout << color::fg_yellow << "OPTIONS:" << color::reset << std::endl;
                for(auto&& arg : self.arguments)
                {
                    if(arg.nargs == 0)
                        continue;
                    std::cout << std::endl;
                    std::string prefix = "    ";
                    std::cout << color::fg_green;
                    if(arg.flags.empty())
                    {
                        std::cout << prefix;
                        std::cout << arg.metavar;
                    }
                    for(const std::string& a : arg.flags)
                    {
                        std::cout << prefix;
                        std::cout << a;
                        prefix = ", ";
                    }
                    std::cout << color::reset;
                    if(not arg.type.empty())
                    {
                        std::cout << " [" << color::fg_blue << arg.type << color::reset << "]";
                        if(not arg.default_value.empty())
                            std::cout << " (Default: " << arg.default_value << ")";
                    }
                    std::cout << std::endl;
                    std::cout << "        " << arg.help << std::endl;
                }
                std::cout << std::endl;
            }
            if(not msg.empty())
                std::cout << msg << std::endl;
        });
    }

    MIGRAPHX_DRIVER_STATIC auto help(const std::string& help)
    {
        return [=](auto&, auto& arg) { arg.help = help; };
    }

    MIGRAPHX_DRIVER_STATIC auto metavar(const std::string& metavar)
    {
        return [=](auto&, auto& arg) { arg.metavar = metavar; };
    }

    MIGRAPHX_DRIVER_STATIC auto type(const std::string& type)
    {
        return [=](auto&, auto& arg) { arg.type = type; };
    }

    template <class T>
    MIGRAPHX_DRIVER_STATIC auto set_value(T value)
    {
        return [=](auto& x, auto& arg) {
            arg.nargs  = 0;
            arg.type   = "";
            arg.action = [&, value](auto&, const std::vector<std::string>&) {
                x = value;
                return false;
            };
        };
    }

    template <class T>
    void set_exe_name_to(T& x)
    {
        actions.push_back([&](auto& self) { x = self.exe_name; });
    }

    bool
    run_action(const argument& arg, const std::string& flag, const std::vector<std::string>& inputs)
    {
        std::string msg = "";
        try
        {
            return arg.action(*this, inputs);
        }
        catch(const std::exception& e)
        {
            msg = e.what();
        }
        catch(...)
        {
            msg = "unknown exception";
        }
        auto show_usage = [&] {
            std::cout << flag;
            if(not arg.type.empty())
                std::cout << " [" << arg.type << "]";
        };
        std::cout << color::fg_red << color::bold << "error: " << color::reset;
        std::cout << "Invalid input to '" << color::fg_yellow;
        show_usage();
        std::cout << color::reset << "'" << std::endl;
        std::cout << "       " << msg << std::endl;
        std::cout << std::endl;
        std::cout << color::fg_yellow << "USAGE:" << color::reset << std::endl;
        std::cout << "    " << exe_name << " ";
        show_usage();
        std::cout << std::endl;
        if(find_argument([](const auto& a) { return contains(a.flags, "--help"); }))
        {
            std::cout << std::endl;
            std::cout << "For more information try '" << color::fg_green << "--help" << color::reset
                      << "'" << std::endl;
        }
        return true;
    }

    bool parse(std::vector<std::string> args)
    {
        std::unordered_map<std::string, unsigned> keywords;
        for(auto&& arg : arguments)
        {
            for(auto&& flag : arg.flags)
                keywords[flag] = arg.nargs + 1;
        }
        auto arg_map =
            generic_parse(std::move(args), [&](const std::string& x) { return keywords[x]; });
        for(auto&& arg : arguments)
        {
            auto flags = arg.flags;
            if(flags.empty())
                flags = {""};
            for(auto&& flag : flags)
            {
                if(arg_map.count(flag) > 0)
                {
                    if(run_action(arg, flag, arg_map[flag]))
                        return true;
                }
            }
        }
        for(auto&& action : actions)
            action(*this);
        return false;
    }

    void set_exe_name(const std::string& s) { exe_name = s; }

    const std::string& get_exe_name() const { return exe_name; }

    using string_map = std::unordered_map<std::string, std::vector<std::string>>;
    template <class IsKeyword>
    static string_map generic_parse(std::vector<std::string> as, IsKeyword is_keyword)
    {
        string_map result;

        std::string flag;
        bool clear = false;
        for(auto&& x : as)
        {
            auto k = is_keyword(x);
            if(k > 0)
            {
                flag = x;
                result[flag]; // Ensure the flag exists
                if(k == 1)
                    flag = "";
                else if(k == 2)
                    clear = true;
                else
                    clear = false;
            }
            else
            {
                result[flag].push_back(x);
                if(clear)
                    flag = "";
                clear = false;
            }
        }
        return result;
    }

    private:
    std::list<argument> arguments;
    std::string exe_name = "";
    std::vector<std::function<void(argument_parser&)>> actions;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx

#endif
