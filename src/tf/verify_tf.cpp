#include <migraphx/tf.hpp>

#include <migraphx/cpu/target.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/verify_args.hpp>
#include <migraphx/instruction.hpp>

template <class T>
auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

template <class F>
migraphx::argument run_cpu(F f)
{
    auto p = f();
    p.compile(migraphx::cpu::target{});
    migraphx::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        m[x.first] = migraphx::generate_argument(x.second, get_hash(x.first));
    }
    auto out = p.eval(m);
    std::cout << p << std::endl;
    return out;
}

template <class F>
migraphx::argument run_gpu(F f)
{
    auto p = f();
    p.compile(migraphx::gpu::target{});

    migraphx::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        m[x.first] =
            migraphx::gpu::to_gpu(migraphx::generate_argument(x.second, get_hash(x.first)));
    }
    auto out = migraphx::gpu::from_gpu(p.eval(m));
    std::cout << p << std::endl;
    return migraphx::gpu::from_gpu(out);
}

template <class F>
void verify_program(const std::string& name, F f, double tolerance = 100)
{
    auto x = run_cpu(f);
    auto y = run_gpu(f);
    migraphx::verify_args(name, x, y, tolerance);
    // std::cout << "cpu: " << x << std::endl;
    // std::cout << "gpu: " << y << std::endl;
}

void verify_instructions(const migraphx::program& prog, double tolerance = 80)
{
    for(auto&& ins : prog)
    {
        if(ins.name().front() == '@')
            continue;
        if(ins.name() == "broadcast")
            continue;
        if(ins.name() == "transpose")
            continue;
        if(ins.name() == "reshape")
            continue;
        auto create_program = [&] {
            migraphx::program p;
            std::vector<migraphx::instruction_ref> inputs;
            for(auto&& arg : ins.inputs())
            {
                if(arg->name() == "@literal")
                    inputs.push_back(p.add_literal(arg->get_literal()));
                else
                    inputs.push_back(
                        p.add_parameter(std::to_string(inputs.size()), arg->get_shape()));
            }
            p.add_instruction(ins.get_operator(), inputs);
            return p;
        };
        try
        {
            std::cout << "Verify: " << ins.name() << std::endl;
            std::cout << create_program() << std::endl;
            verify_program(ins.name(), create_program, tolerance);
        }
        catch(...)
        {
            std::cout << "Instruction " << ins.name() << " threw an exception." << std::endl;
            throw;
        }
    }
}

template <class F>
void verify_reduced(F f, int n, double tolerance = 80)
{

    auto create_program = [&] {
        migraphx::program p = f();
        auto last           = std::prev(p.end(), n + 1);
        p.remove_instructions(last, p.end());
        return p;
    };
    std::cout << "Verify: " << std::endl;
    std::cout << create_program() << std::endl;
    verify_program(std::to_string(n), create_program, tolerance);
}

template <class F>
void verify_reduced_program(F f, double tolerance = 80)
{
    migraphx::program p = f();
    auto n              = std::distance(p.begin(), p.end());
    for(std::size_t i = 0; i < n; i++)
    {
        verify_reduced(f, i, tolerance);
    }
}

int main(int argc, char const* argv[])
{
    std::vector<std::string> args(argv + 1, argv + argc);
    if(not args.empty())
    {
        bool is_nhwc = true;

        if(std::any_of(args.begin(), args.end(), [](const auto& s) { return s == "nchw"; }))
        {
            is_nhwc = false;
        }

        std::string file = args.front();
        auto p           = migraphx::parse_tf(file, is_nhwc);
        std::cout << p << std::endl;

        if(std::any_of(args.begin(), args.end(), [](const auto& s) { return s == "-i"; }))
        {
            verify_instructions(p);
        }
        else if(std::any_of(args.begin(), args.end(), [](const auto& s) { return s == "-r"; }))
        {
            verify_reduced_program([&] { return migraphx::parse_tf(file, is_nhwc); });
        }
        else
        {
            verify_program(file, [&] { return migraphx::parse_tf(file, is_nhwc); });
        }
    }
}
