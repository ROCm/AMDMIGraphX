#ifndef MIGRAPHX_GUARD_TEST_AUTO_PRINT_HPP
#define MIGRAPHX_GUARD_TEST_AUTO_PRINT_HPP

#include <migraphx/register_target.hpp>
#include <functional>

struct auto_print
{
    static std::function<void()>& get_handler(const std::string& name);
    static void set_terminate_handler(const std::string& name);
    std::string name;
    template <class T>
    auto_print(T& x, std::string s) : name(std::move(s))
    {
        get_handler(name) = [&x] { std::cout << x << std::endl; };
    }

    ~auto_print();
};

#endif
