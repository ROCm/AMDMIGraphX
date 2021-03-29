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
struct shape;

struct cpp_generator_impl;

struct cpp_generator
{
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
        function& set_body(const module& m, const generate_module_callback& g);
        function& set_body(const std::string& s)
        {
            body = s;
            return *this;
        }
        function& set_name(const std::string& s)
        {
            name = s;
            return *this;
        }
        function& set_attributes(std::vector<std::string> attrs)
        {
            attributes = std::move(attrs);
            return *this;
        }
        function& set_types(const module& m);
        function& set_types(const module& m, const std::function<std::string(shape)>& parse);
    };

    cpp_generator();

    // move constructor
    cpp_generator(cpp_generator&&) noexcept;

    // copy assignment operator
    cpp_generator& operator=(cpp_generator rhs);

    ~cpp_generator() noexcept;

    void fmap(const std::function<std::string(std::string)>& f);

    std::string generate_point_op(const operation& op, const std::vector<std::string>& args);

    std::string str() const;

    function generate_module(const module& m, const generate_module_callback& g);

    function generate_module(const module& m);

    std::string create_function(const function& f);

    private:
    std::unique_ptr<cpp_generator_impl> impl;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_CPP_GENERATOR_HPP
