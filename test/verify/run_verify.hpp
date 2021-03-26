#ifndef MIGRAPHX_GUARD_TEST_RUN_VERIFY_HPP
#define MIGRAPHX_GUARD_TEST_RUN_VERIFY_HPP

#include <migraphx/program.hpp>
#include <functional>
#include <map>

struct target_info
{
    using validation_function =
        std::function<void(const migraphx::program& p, const migraphx::parameter_map& m)>;
    bool parallel = true;
    validation_function validate;
    std::vector<std::string> disabled_tests;
};

struct run_verify
{
    std::vector<migraphx::argument> run_ref(migraphx::program p,
                                            migraphx::parameter_map inputs) const;
    std::pair<migraphx::program, std::vector<migraphx::argument>>
    run_target(const migraphx::target& t,
               migraphx::program p,
               const migraphx::parameter_map& inputs) const;
    void validate(const migraphx::target& t,
                  const migraphx::program& p,
                  const migraphx::parameter_map& m) const;
    void verify(const std::string& name, const migraphx::program& p) const;
    void run(int argc, const char* argv[]) const;

    target_info get_target_info(const std::string& name) const;
    void disable_parallel_for(const std::string& name);
    void add_validation_for(const std::string& name, target_info::validation_function v);
    void disable_test_for(const std::string& name, const std::vector<std::string>& tests);

    private:
    std::map<std::string, target_info> info{};
};

#endif
