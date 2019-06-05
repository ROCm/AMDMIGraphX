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

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

template <class T>
struct value_parser
{
    template <MIGRAPHX_REQUIRES(not std::is_enum<T>{})>
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

    template <MIGRAPHX_REQUIRES(std::is_enum<T>{})>
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
};

struct argument_parser
{
    struct argument
    {
        std::vector<std::string> flags;
        std::function<bool(argument_parser&, const std::vector<std::string>&)> action{};
        std::string type = "";
        std::string help = "";
    };

    template <class T, class... Fs>
    void add(T& x, std::vector<std::string> flags, Fs... fs)
    {
        arguments.emplace_back(flags, [&](auto&&, const std::vector<std::string>& params) {
            if(params.empty())
                throw std::runtime_error("Flag with no value.");
            x = value_parser<T>::apply(params.back());
            return false;
        });

        argument& arg = arguments.back();
        arg.type      = migraphx::get_type_name<T>();
        migraphx::each_args([&](auto f) { f(x, arg); }, fs...);
    }

    template <class... Fs>
    void add(std::nullptr_t x, std::vector<std::string> flags, Fs... fs)
    {
        arguments.push_back({flags});

        argument& arg = arguments.back();
        arg.type      = "";
        migraphx::each_args([&](auto f) { f(x, arg); }, fs...);
    }

    template <class F>
    static auto write_action(F f)
    {
        return [=](auto& x, auto& arg) {
            arg.action = [&, f](auto& self, const std::vector<std::string>& params)
            {
                f(self, x, params);
                return false;
            };
        };
    }

    template <class F>
    static auto do_action(F f)
    {
        return [=](auto&, auto& arg) {
            arg.action = [&, f](auto& self, const std::vector<std::string>&)
            {
                f(self);
                return true;
            };
        };
    }

    static auto write_range()
    {
        return write_action([](auto&, auto& x, auto& params) {
            using type = typename decltype(params)::value_type;
            std::transform(params.begin(),
                           params.end(),
                           std::inserter(x, x.end()),
                           [](std::string y) { return value_parser<type>::apply(y); });
        });
    }

    static auto show_help(std::string msg = "")
    {
        return do_action([=](auto& self) {
            for(auto&& arg : self.arguments)
            {
                std::cout << std::endl;
                std::string prefix = "    ";
                for(const std::string& a : arg.flags)
                {
                    std::cout << prefix;
                    std::cout << a;
                    prefix = ", ";
                }
                if(not arg.type.empty())
                    std::cout << " [" << arg.type << "]";
                std::cout << std::endl;
                std::cout << "        " << arg.help << std::endl;
            }
            std::cout << std::endl;
            if (not msg.empty())
                std::cout << msg << std::endl;
        });
    }

    static auto help(std::string help)
    {
        return [=](auto&, auto& arg) { arg.help = help; };
    }

    template <class T>
    static auto set_value(T value)
    {
        return [=](auto& x, auto& arg) {
            arg.type   = "";
            arg.action = [&, value](auto&, const std::vector<std::string>&)
            {
                x = value;
                return false;
            };
        };
    }

    void parse(std::vector<std::string> args)
    {
        std::set<std::string> keywords;
        for(auto&& arg : arguments)
        {
            keywords.insert(arg.flags.begin(), arg.flags.end());
        }
        auto arg_map = generic_parse(args, [&](std::string x) { return (keywords.count(x) > 0); });
        for(auto&& arg : arguments)
        {
            for(auto&& flag : arg.flags)
            {
                if(arg_map.count(flag) > 0)
                {
                    if(arg.action(*this, arg_map[flag]))
                        return;
                }
            }
        }
    }

    using string_map = std::unordered_map<std::string, std::vector<std::string>>;
    template <class IsKeyword>
    static string_map generic_parse(std::vector<std::string> as, IsKeyword is_keyword)
    {
        string_map result;

        std::string flag;
        for(auto&& x : as)
        {
            if(is_keyword(x))
            {
                flag = x;
                result[flag]; // Ensure the flag exists
            }
            else
            {
                result[flag].push_back(x);
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
