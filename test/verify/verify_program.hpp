#ifndef MIGRAPHX_GUARD_AUTO_REGISTER_VERIFY_PROGRAM_HPP
#define MIGRAPHX_GUARD_AUTO_REGISTER_VERIFY_PROGRAM_HPP

#include <migraphx/auto_register.hpp>
#include <migraphx/program.hpp>
#include <functional>

struct program_info
{
    std::string name;
    std::string section;
    std::function<migraphx::program()> get_program;
};

void register_program_info(const program_info& pi);
const std::vector<program_info>& get_programs();

struct register_verify_program_action
{
    template <class T>
    static void apply()
    {
        T x;
        program_info pi;
        pi.name        = migraphx::get_type_name<T>();
        pi.section     = x.section();
        pi.get_program = [x] { return x.create_program(); };
        register_program_info(pi);
    }
};

template <class T>
using auto_register_verify_program = migraphx::auto_register<register_verify_program_action, T>;

template <class T>
struct verify_program : auto_register_verify_program<T>
{
    std::string section() const { return "general"; };
};

#endif
