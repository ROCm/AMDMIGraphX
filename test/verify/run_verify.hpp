#ifndef MIGRAPHX_GUARD_TEST_RUN_VERIFY_HPP
#define MIGRAPHX_GUARD_TEST_RUN_VERIFY_HPP

#include <migraphx/program.hpp>
#include <functional>
#include <map>

struct target_info
{
    bool parallel = true;
    std::function<void(const migraphx::program& p)> validate;
};

struct run_verify
{
    std::vector<migraphx::argument> run_ref(migraphx::program p, migraphx::program::parameter_map inputs) const;
    std::pair<migraphx::program, std::vector<migraphx::argument>> run_target(const migraphx::target& t, migraphx::program p, migraphx::program::parameter_map inputs) const;
    void validate(const migraphx::target& t, const migraphx::program& p) const;
    void verify(const std::string& name, const migraphx::program& p) const;
    void run(int argc, const char* argv[]) const;

    target_info get_target_info(const std::string& name) const;
    void disable_parallel_for(const std::string& name);
    void add_validation_for(const std::string& name, std::function<void(const migraphx::program& p)> v);

private:
    std::map<std::string, target_info> info{};
};

#endif
