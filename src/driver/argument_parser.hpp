#ifndef MIGRAPHX_GUARD_RTGLIB_ARGUMENT_PARSER_HPP
#define MIGRAPHX_GUARD_RTGLIB_ARGUMENT_PARSER_HPP

#include <algorithm>
#include <functional>
#include <iostream>
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
            throw std::runtime_error("Failed to parse: " + x);
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
            throw std::runtime_error("Failed to parse: " + x);
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

template<class T>
struct type_name
{
    static const std::string& apply()
    {
        return migraphx::get_type_name<T>();
    }
};

template<>
struct type_name<std::string>
{
    static const std::string& apply()
    {
        static const std::string name = "std::string";
        return name;
    }
};

template<class T>
struct type_name<std::vector<T>>
{
    static const std::string& apply()
    {
        static const std::string name = "std::vector<" + type_name<T>::apply() + ">";
        return name;
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

    template <class T, MIGRAPHX_REQUIRES(not is_multi_value<T>{})>
    std::string as_string_value(const T& x)
    {
        return to_string(x);
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

        argument& arg     = arguments.back();
        arg.type          = type_name<T>::apply();
        arg.default_value = as_string_value(x);
        migraphx::each_args([&](auto f) { f(x, arg); }, fs...);
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

    MIGRAPHX_DRIVER_STATIC auto show_help(const std::string& msg = "")
    {
        return do_action([=](auto& self) {
            for(auto&& arg : self.arguments)
            {
                std::cout << std::endl;
                std::string prefix = "    ";
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
                if(not arg.type.empty())
                {
                    std::cout << " [" << arg.type << "]";
                    if(not arg.default_value.empty())
                        std::cout << " (Default: " << arg.default_value << ")";
                }
                std::cout << std::endl;
                std::cout << "        " << arg.help << std::endl;
            }
            std::cout << std::endl;
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
                    if(arg.action(*this, arg_map[flag]))
                        return true;
                }
            }
        }
        return false;
    }

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
    std::vector<argument> arguments;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx

#endif
