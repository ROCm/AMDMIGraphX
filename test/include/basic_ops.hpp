#include <migraph/program.hpp>
#include <migraph/argument.hpp>
#include <migraph/shape.hpp>

struct sum_op
{
    std::string name() const { return "sum"; }
    migraph::argument
    compute(migraph::context&, migraph::shape, std::vector<migraph::argument> args) const
    {
        migraph::argument result;
        if(args.size() != 2)
            MIGRAPH_THROW("Wrong args");
        if(args[0].get_shape() != args[1].get_shape())
            MIGRAPH_THROW("Wrong args");
        if(args[0].get_shape().lens().size() != 1)
            MIGRAPH_THROW("Wrong args");
        if(args[0].get_shape().lens().front() != 1)
            MIGRAPH_THROW("Wrong args");

        args[0].visit_at([&](auto x) {
            args[1].visit_at([&](auto y) { result = migraph::literal{x + y}.get_argument(); });
        });
        return result;
    }

    migraph::shape compute_shape(std::vector<migraph::shape> inputs) const
    {
        if(inputs.size() != 2)
            MIGRAPH_THROW("Wrong inputs");
        return inputs.front();
    }
};

struct minus_op
{
    std::string name() const { return "minus"; }
    migraph::argument
    compute(migraph::context&, migraph::shape, std::vector<migraph::argument> args) const
    {
        migraph::argument result;
        if(args.size() != 2)
            MIGRAPH_THROW("Wrong args");
        if(args[0].get_shape() != args[1].get_shape())
            MIGRAPH_THROW("Wrong args");
        if(args[0].get_shape().lens().size() != 1)
            MIGRAPH_THROW("Wrong args");
        if(args[0].get_shape().lens().front() != 1)
            MIGRAPH_THROW("Wrong args");

        args[0].visit_at([&](auto x) {
            args[1].visit_at([&](auto y) { result = migraph::literal{x - y}.get_argument(); });
        });
        return result;
    }

    migraph::shape compute_shape(std::vector<migraph::shape> inputs) const
    {
        if(inputs.size() != 2)
            MIGRAPH_THROW("Wrong inputs");
        return inputs.front();
    }
};

struct pass_op
{
    std::string name() const { return "pass"; }
    migraph::argument
    compute(migraph::context&, migraph::shape, std::vector<migraph::argument> args) const
    {
        if(args.empty())
            return {};
        return args.front();
    }

    migraph::shape compute_shape(std::vector<migraph::shape> inputs) const
    {
        if(inputs.empty())
            return {};
        return inputs.front();
    }
};
