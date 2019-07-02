#include "argument_parser.hpp"
#include "command.hpp"
#include "verify.hpp"
#include "perf.hpp"

#include <migraphx/tf.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/stringutils.hpp>

#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/eliminate_pad.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/simplify_algebra.hpp>
#include <migraphx/simplify_reshapes.hpp>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

struct loader
{
    std::string file;
    std::string file_type;
    bool is_nhwc  = true;
    unsigned trim = 0;
    bool optimize = false;

    void parse(argument_parser& ap)
    {
        ap(file, {}, ap.metavar("<input file>"));
        ap(file_type, {"--onnx"}, ap.help("Load as onnx"), ap.set_value("onnx"));
        ap(file_type, {"--tf"}, ap.help("Load as tensorflow"), ap.set_value("tf"));
        ap(is_nhwc, {"--nhwc"}, ap.help("Treat tensorflow format as nhwc"), ap.set_value(true));
        ap(is_nhwc, {"--nchw"}, ap.help("Treat tensorflow format as nchw"), ap.set_value(false));
        ap(trim, {"--trim", "-t"}, ap.help("Trim instructions from the end"));
        ap(optimize, {"--optimize"}, ap.help("Optimize when reading"), ap.set_value(true));
    }

    program load()
    {
        program p;
        if(file_type.empty())
        {
            if(ends_with(file, ".onnx"))
                file_type = "onnx";
            else if(ends_with(file, ".pb"))
                file_type = "tf";
        }
        std::cout << "Reading: " << file << std::endl;
        if(file_type == "onnx")
            p = parse_onnx(file);
        else if(file_type == "tf")
            p = parse_tf(file, is_nhwc);
        if(trim > 0)
        {
            auto last = std::prev(p.end(), trim);
            p.remove_instructions(last, p.end());
        }
        if (optimize)
            migraphx::run_passes(p, {
                            migraphx::eliminate_identity{},
                            migraphx::dead_code_elimination{},
                            migraphx::simplify_algebra{},
                            migraphx::dead_code_elimination{},
                            migraphx::simplify_reshapes{},
                            migraphx::dead_code_elimination{},
                            migraphx::propagate_constant{},
                            migraphx::dead_code_elimination{},
                            migraphx::eliminate_pad{},
                            migraphx::dead_code_elimination{},
                          });
        return p;
    }
};

struct compiler
{
    loader l;
    bool gpu = true;
    void parse(argument_parser& ap)
    {
        l.parse(ap);
        ap(gpu, {"--gpu"}, ap.help("Compile on the gpu"), ap.set_value(true));
        ap(gpu, {"--cpu"}, ap.help("Compile on the cpu"), ap.set_value(false));
    }

    program compile()
    {
        auto p = l.load();
        compile_program(p, gpu);
        return p;
    }

    auto params(const program& p) { return create_param_map(p, gpu); }
};

struct read : command<read>
{
    loader l;
    void parse(argument_parser& ap) { l.parse(ap); }

    void run()
    {
        auto p = l.load();
        std::cout << p << std::endl;
    }
};

struct verify : command<verify>
{
    loader l;
    double tolerance     = 80;
    bool per_instruction = false;
    bool reduce          = false;
    void parse(argument_parser& ap)
    {
        l.parse(ap);
        ap(tolerance, {"--tolerance"}, ap.help("Tolerance for errors"));
        ap(per_instruction,
           {"-i", "--per-instruction"},
           ap.help("Verify each instruction"),
           ap.set_value(true));
        ap(reduce, {"-r", "--reduce"}, ap.help("Reduce program and verify"), ap.set_value(true));
    }

    void run()
    {
        auto p = l.load();
        std::cout << p << std::endl;

        if(per_instruction)
        {
            verify_instructions(p, tolerance);
        }
        else if(reduce)
        {
            verify_reduced_program(p, tolerance);
        }
        else
        {
            verify_program(l.file, p, tolerance);
        }
    }
};

struct compile : command<compile>
{
    compiler c;
    void parse(argument_parser& ap) { c.parse(ap); }

    void run()
    {
        std::cout << "Compiling ... " << std::endl;
        auto p = c.compile();
        std::cout << p << std::endl;
    }
};

struct run_cmd : command<run_cmd>
{
    compiler c;
    void parse(argument_parser& ap) { c.parse(ap); }

    void run()
    {
        std::cout << "Compiling ... " << std::endl;
        auto p = c.compile();
        std::cout << "Allocating params ... " << std::endl;
        auto m = c.params(p);
        p.eval(m);
        std::cout << p << std::endl;
    }
};

struct perf : command<perf>
{
    compiler c;
    unsigned n = 100;
    void parse(argument_parser& ap)
    {
        c.parse(ap);
        ap(n, {"--iterations", "-n"}, ap.help("Number of iterations to run for perf report"));
    }

    void run()
    {
        std::cout << "Compiling ... " << std::endl;
        auto p = c.compile();
        std::cout << "Allocating params ... " << std::endl;
        auto m = c.params(p);
        std::cout << "Running performance report ... " << std::endl;
        p.perf_report(std::cout, n, m);
    }
};

struct main_command
{
    static std::string get_command_help()
    {
        std::string result = "Commands:\n";
        return std::accumulate(get_commands().begin(),
                               get_commands().end(),
                               result,
                               [](auto r, auto&& p) { return r + "    " + p.first + "\n"; });
    }
    void parse(argument_parser& ap)
    {
        ap(nullptr, {"-h", "--help"}, ap.help("Show help"), ap.show_help(get_command_help()));
    }

    void run() {}
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx

using namespace migraphx::driver; // NOLINT
int main(int argc, const char* argv[])
{
    std::vector<std::string> args(argv + 1, argv + argc);
    if(args.empty())
        return 0;
    auto&& m = get_commands();
    auto cmd = args.front();
    if(m.count(cmd) > 0)
    {
        m.at(cmd)({args.begin() + 1, args.end()});
    }
    else
    {
        run_command<main_command>(args);
    }
    return 0;
}
