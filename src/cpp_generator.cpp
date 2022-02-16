#include <migraphx/cpp_generator.hpp>
#include <migraphx/module.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/builtin.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/iterator_for.hpp>
#include <map>
#include <sstream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

cpp_generator::function&
cpp_generator::function::set_body(const module& m, const cpp_generator::generate_module_callback& g)
{
    std::unordered_map<migraphx::instruction_ref, std::string> names;
    std::stringstream ss;

    auto return_ins = std::prev(m.end());

    for(auto ins : iterator_for(m))
    {
        ss << "// " << ins->get_operator() << " -> " << ins->get_shape() << "\n";
        if(ins->name() == "@param")
        {
            names[ins] =
                migraphx::any_cast<migraphx::builtin::param>(ins->get_operator()).parameter;
        }
        else if(ins->name() == "@return")
        {
            assert(ins->inputs().size() == 1);
            return_ins = ins->inputs().front();
        }
        else
        {
            std::string n = "z" + std::to_string(names.size());
            names[ins]    = n;
            ss << "auto " << n << " = " << g(ins, names) << ";\n";
        }
    }
    ss << "return " << names.at(return_ins) << ";\n";
    body = ss.str();
    return *this;
}

cpp_generator::function& cpp_generator::function::set_types(const module& m)
{
    return cpp_generator::function::set_types(m, [](auto s) { return shape::cpp_type(s.type()); });
}
cpp_generator::function&
cpp_generator::function::set_types(const module& m, const std::function<std::string(shape)>& parse)
{
    this->params.clear();
    auto pmap = m.get_parameter_shapes();
    std::map<std::string, shape> input_map(pmap.begin(), pmap.end());
    std::transform(
        input_map.begin(), input_map.end(), std::back_inserter(this->params), [&](auto&& p) {
            return param{p.first, parse(p.second)};
        });
    auto output_shapes = m.get_output_shapes();
    assert(not output_shapes.empty());
    this->return_type = parse(output_shapes.front());
    return *this;
}

cpp_generator::function& cpp_generator::function::set_generic_types(const module& m)
{
    this->params.clear();
    auto pmap = m.get_parameter_shapes();
    std::map<std::string, shape> input_map(pmap.begin(), pmap.end());
    std::transform(
        input_map.begin(), input_map.end(), std::back_inserter(this->params), [&](auto&& p) {
            return param{p.first, "T" + p.first};
        });

    std::transform(input_map.begin(),
                   input_map.end(),
                   std::back_inserter(this->tparams),
                   [&](auto&& p) { return "class T" + p.first; });
    this->return_type = "auto";
    return *this;
}

struct cpp_generator_impl
{
    std::stringstream fs{};
    std::size_t function_count                                = 0;
    std::function<std::string(std::string)> fmap              = nullptr;
    std::function<std::string(shape)> fresult                 = nullptr;
    std::unordered_map<std::string, std::string> point_op_map = {};
};
cpp_generator::cpp_generator() : impl(std::make_unique<cpp_generator_impl>()) {}

cpp_generator::cpp_generator(cpp_generator&&) noexcept = default;

cpp_generator& cpp_generator::operator=(cpp_generator rhs)
{
    std::swap(impl, rhs.impl);
    return *this;
}

cpp_generator::~cpp_generator() noexcept = default;

void cpp_generator::fmap(const std::function<std::string(std::string)>& f) { impl->fmap = f; }

void cpp_generator::fresult(const std::function<std::string(shape)>& f) { impl->fresult = f; }

void cpp_generator::add_point_op(const std::string& op_name, const std::string& code)
{
    impl->point_op_map[op_name] = code;
}

std::string cpp_generator::generate_point_op(const operation& op,
                                             const std::vector<std::string>& args)
{
    auto v = op.to_value();
    std::string code;
    if(contains(impl->point_op_map, op.name()))
    {
        code = impl->point_op_map.at(op.name());
    }
    else
    {
        auto attributes = op.attributes();
        if(not attributes.contains("point_op"))
            MIGRAPHX_THROW("op is missing point_op attribute: " + op.name());
        code = attributes["point_op"].to<std::string>();
    }
    return interpolate_string(code, [&](auto start, auto last) -> std::string {
        auto key = trim({start, last});
        if(key.empty())
            MIGRAPHX_THROW("Empty parameter");
        std::string fselector = "function:";
        if(starts_with(key, fselector))
        {
            auto fname = key.substr(fselector.size());
            if(impl->fmap == nullptr)
                return fname;
            else
                return impl->fmap(fname);
        }
        else if(with_char(::isdigit)(key[0]))
        {
            auto i = std::stoul(key);
            return args.at(i);
        }
        else if(v.contains(key))
        {
            return v[key].template to<std::string>();
        }
        else
        {
            return key;
        }
    });
}

std::string cpp_generator::str() const { return impl->fs.str(); }

cpp_generator::function cpp_generator::generate_module(const module& m)
{
    function f;
    auto name = transform_string(m.name(), [](char c) {
        if(with_char(::isalnum)(c) or c == '_')
            return c;
        return '_';
    });
    f.set_name(name).set_types(m).set_body(
        m, [&](instruction_ref ins, const auto& names) -> std::string {
            if(ins->name() == "@literal")
                return shape::cpp_type(ins->get_shape().type()) + "(" +
                       ins->get_literal().to_string() + ")";
            std::vector<std::string> args;
            std::transform(ins->inputs().begin(),
                           ins->inputs().end(),
                           std::back_inserter(args),
                           [&](auto i) { return names.at(i); });

            auto s = this->generate_point_op(ins->get_operator(), args);
            if(impl->fresult)
                return impl->fresult(ins->get_shape()) + '(' + s + ')';
            else
                return s;
        });
    return f;
}

std::string cpp_generator::create_function(const cpp_generator::function& f)
{
    impl->function_count++;
    if(not f.tparams.empty())
        impl->fs << "template<" << join_strings(f.tparams, ", ") << ">\n";
    std::string name = f.name.empty() ? "f" + std::to_string(impl->function_count) : f.name;
    impl->fs << join_strings(f.attributes, " ") << " " << f.return_type << " " << name;
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
