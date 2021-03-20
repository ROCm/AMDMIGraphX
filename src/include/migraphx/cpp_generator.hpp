#ifndef MIGRAPHX_GUARD_MIGRAPHX_CPP_GENERATOR_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_CPP_GENERATOR_HPP

#include <migraphx/config.hpp>
#include <migraphx/instruction_ref.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct operation;
struct module;

struct cpp_generator_impl;

struct cpp_generator
{
    using string_map               = std::unordered_map<std::string, std::string>;
    using generate_module_callback = std::function<std::string(
        instruction_ref, const std::unordered_map<instruction_ref, std::string>&)>;
    struct param
    {
        std::string name;
        std::string type;
    };

    struct function
    {
        std::vector<param> params           = {};
        std::string body                    = "";
        std::string return_type             = "void";
        std::string name                    = "";
        std::vector<std::string> attributes = {};
    };

    cpp_generator();

    // move constructor
    cpp_generator(cpp_generator&& rhs) noexcept;

    // copy assignment operator
    cpp_generator& operator=(cpp_generator rhs);

    ~cpp_generator() noexcept;

    static std::string generate_point_op(const operation& op,
                                         const std::vector<std::string>& args,
                                         const string_map& fmap = {});

    std::string str() const;

    std::string generate_module(function f, const module& m, const generate_module_callback& g);

    std::string create_function(const function& f);

    private:
    std::unique_ptr<cpp_generator_impl> impl;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_CPP_GENERATOR_HPP
