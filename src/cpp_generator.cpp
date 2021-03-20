#include <migraphx/cpp_generator.hpp>
#include <migraphx/module.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/builtin.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/iterator_for.hpp>
#include <sstream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct cpp_generator_impl
{
    std::stringstream fs{};
    std::size_t function_count = 0;
};
cpp_generator::cpp_generator() : impl(std::make_unique<cpp_generator_impl>())
{
}

cpp_generator::cpp_generator(cpp_generator&& rhs) noexcept = default;

cpp_generator& cpp_generator::operator=(cpp_generator rhs)
{
    std::swap(impl, rhs.impl);
    return *this;
}

cpp_generator::~cpp_generator() noexcept = default;

std::string cpp_generator::generate_point_op(const operation& op, const std::vector<std::string>& args, const cpp_generator::string_map& fmap)
{
    auto v = op.to_value();
    return interpolate_string(op.attributes()["point_op"].to<std::string>(), [&](auto start, auto last) -> std::string {
        auto key = trim({start, last});
        if (key.empty())
            MIGRAPHX_THROW("Empty parameter");
        std::string fselector = "function:";
        if(starts_with(key, fselector))
        {
            auto fname = key.substr(fselector.size());
            auto it = fmap.find(fname);
            if (it == fmap.end())
                return fname;
            else
                return it->second;
        }
        else if(with_char(::isdigit)(key[0]))
        {
            auto i = std::stoul(key);
            return args.at(i);
        }
        else if (v.contains(key))
        {
            return v[key].template to<std::string>();
        }
        else
        {
            return key;
        }
    });
}

std::string cpp_generator::str() const
{
    return impl->fs.str();
}

std::string cpp_generator::generate_module(cpp_generator::function f, const module& m, const cpp_generator::generate_module_callback& g)
{
    std::unordered_map<migraphx::instruction_ref, std::string> names;
    std::stringstream ss;

    auto return_ins = std::prev(m.end());

    for(auto ins : iterator_for(m))
    {
        ss << "// " << ins->get_operator() << " -> " << ins->get_shape() << "\n";
        if(ins->name() == "@return")
        {
            assert(ins->inputs().size() == 1);
            return_ins = ins->inputs().front();
        }
        std::string name = "z" + std::to_string(names.size());
        if(ins->name() == "@param")
        {
            name = migraphx::any_cast<migraphx::builtin::param>(ins->get_operator()).parameter;
        }
        names[ins] = name;
        ss << "auto " << name << " = " << g(ins, names) << ";\n";
    }
    ss << "return " << names.at(return_ins) << ";\n";
    f.body = ss.str();
    return create_function(f);
}

std::string cpp_generator::create_function(const cpp_generator::function& f)
{
    impl->function_count++;
    std::string name = f.name.empty() ? "f" + std::to_string(impl->function_count) : f.name;
    impl->fs << join_strings(f.attributes, " ") << f.return_type << " " << name << "(";
    char delim = '(';
    for(auto&& p : f.params)
    {
        impl->fs << delim << p.type << " " << p.name;
        delim = ',';
    }
    impl->fs << ") {\n" << f.body << "\n}\n";
    return name;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx


