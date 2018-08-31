
#include <migraph/onnx.hpp>

#include <migraph/cpu/cpu_target.hpp>
#include <migraph/gpu/target.hpp>
#include <migraph/gpu/hip.hpp>
#include <migraph/generate.hpp>
#include <migraph/verify_args.hpp>
#include <migraph/instruction.hpp>

template <class T>
auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

template <class F>
migraph::argument run_cpu(F f)
{
    auto p = f();
    p.compile(migraph::cpu::cpu_target{});
    migraph::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        m[x.first] = migraph::generate_argument(x.second, get_hash(x.first));
    }
    auto out = p.eval(m);
    std::cout << p << std::endl;
    return out;
}

template <class F>
migraph::argument run_gpu(F f)
{
    auto p = f();
    p.compile(migraph::gpu::target{});

    migraph::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        m[x.first] = migraph::gpu::to_gpu(migraph::generate_argument(x.second, get_hash(x.first)));
    }
    auto out = migraph::gpu::from_gpu(p.eval(m));
    std::cout << p << std::endl;
    return migraph::gpu::from_gpu(out);
}

template <class F>
void verify_program(const std::string& name, F f, double tolerance = 100)
{
    auto x = run_cpu(f);
    auto y = run_gpu(f);
    migraph::verify_args(name, x, y, tolerance);
}

void verify_instructions(const migraph::program& prog, double tolerance = 80)
{
    for(auto&& ins : prog)
    {
        if(ins.op.name().front() == '@')
            continue;
        if(ins.op.name() == "broadcast")
            continue;
        if(ins.op.name() == "transpose")
            continue;
        if(ins.op.name() == "reshape")
            continue;
        auto create_program = [&] {
            migraph::program p;
            std::vector<migraph::instruction_ref> inputs;
            for(auto&& arg : ins.arguments)
            {
                if(arg->op.name() == "@literal")
                    inputs.push_back(p.add_literal(arg->lit));
                else
                    inputs.push_back(
                        p.add_parameter(std::to_string(inputs.size()), arg->get_shape()));
            }
            p.add_instruction(ins.op, inputs);
            return p;
        };
        try
        {
            std::cout << "Verify: " << ins.op.name() << std::endl;
            std::cout << create_program() << std::endl;
            verify_program(ins.op.name(), create_program, tolerance);
        }
        catch(...)
        {
            std::cout << "Instruction " << ins.op.name() << " threw an exception." << std::endl;
            throw;
        }
    }
}

int main(int argc, char const* argv[])
{
    std::vector<std::string> args(argv + 1, argv + argc);
    if(not args.empty())
    {
        std::string file = args.front();
        auto p           = migraph::parse_onnx(file);
        std::cout << p << std::endl;

        if(std::any_of(args.begin(), args.end(), [](const auto& s) { return s == "-i"; }))
        {
            verify_instructions(p);
        }
        else
        {
            verify_program(file, [&] { return migraph::parse_onnx(file); });
        }
    }
}
