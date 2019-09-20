#include <migraphx/program.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/shape.hpp>

struct sum_op
{
    std::string name() const { return "sum"; }
    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        migraphx::argument result;
        if(args.size() != 2)
            MIGRAPHX_THROW("Wrong args");
        if(args[0].get_shape() != args[1].get_shape())
            MIGRAPHX_THROW("Wrong args");
        if(args[0].get_shape().lens().size() != 1)
            MIGRAPHX_THROW("Wrong args");
        if(args[0].get_shape().lens().front() != 1)
            MIGRAPHX_THROW("Wrong args");

        args[0].visit_at([&](auto x) {
            args[1].visit_at([&](auto y) { result = migraphx::literal{x + y}.get_argument(); });
        });
        return result;
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.size() != 2)
            MIGRAPHX_THROW("Wrong inputs");
        return inputs.front();
    }
};

struct minus_op
{
    std::string name() const { return "minus"; }
    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        migraphx::argument result;
        if(args.size() != 2)
            MIGRAPHX_THROW("Wrong args");
        if(args[0].get_shape() != args[1].get_shape())
            MIGRAPHX_THROW("Wrong args");
        if(args[0].get_shape().lens().size() != 1)
            MIGRAPHX_THROW("Wrong args");
        if(args[0].get_shape().lens().front() != 1)
            MIGRAPHX_THROW("Wrong args");

        args[0].visit_at([&](auto x) {
            args[1].visit_at([&](auto y) { result = migraphx::literal{x - y}.get_argument(); });
        });
        return result;
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.size() != 2)
            MIGRAPHX_THROW("Wrong inputs");
        return inputs.front();
    }
};

struct pass_op
{
    std::string name() const { return "pass"; }
    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        if(args.empty())
            return {};
        return args.front();
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.empty())
            return {};
        return inputs.front();
    }
    int output_alias(const std::vector<migraphx::shape>&) const { return 0; }
};

struct unary_pass_op
{
    std::string name() const { return "unary_pass"; }
    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        if(args.empty())
            return {};
        return args.front();
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.size() != 1)
            MIGRAPHX_THROW("Wrong inputs");
        return inputs.front();
    }
    int output_alias(const std::vector<migraphx::shape>&) const { return 0; }
};

struct pass_standard_op
{
    std::string name() const { return "pass"; }
    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        if(args.empty())
            return {};
        return args.front();
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        for(auto&& input : inputs)
        {
            if(not input.standard())
                throw std::runtime_error("Not standard shape");
        }
        if(inputs.empty())
            return {};
        return inputs.front();
    }
    int output_alias(const std::vector<migraphx::shape>&) const { return 0; }
};

struct nop
{
    std::string name() const { return "nop"; }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape&,
                               const std::vector<migraphx::argument>&) const
    {
        return {};
    }

    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const { return {}; }
};

inline migraphx::literal get_2x2()
{
    return migraphx::literal{{migraphx::shape::float_type, {2, 2}}, {1, 2, 3, 4}};
}

inline migraphx::literal get_2x2_transposed()
{
    return migraphx::literal{{migraphx::shape::float_type, {2, 2}, {1, 2}}, {1, 2, 3, 4}};
}

inline migraphx::literal get_2()
{
    return migraphx::literal{{migraphx::shape::float_type, {2}}, {1, 2}};
}

inline migraphx::literal get_2_broadcasted()
{
    return migraphx::literal{{migraphx::shape::float_type, {2, 1}, {1, 0}}, {1, 2}};
}
