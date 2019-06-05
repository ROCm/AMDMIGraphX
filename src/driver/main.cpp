#include "argument_parser.hpp"
#include "command.hpp"
#include "verify.hpp"

#include <migraphx/tf.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

struct loader
{
    std::string file;
    std::string type;
    bool is_nhwc  = false;
    unsigned trim = 0;

    void parse(argument_parser& ap)
    {
        ap(file, {}, ap.metavar("<input file>"));
        ap(type, {"--onnx"}, ap.help("Load as onnx"), ap.set_value("onnx"));
        ap(type, {"--tf"}, ap.help("Load as tensorflow"), ap.set_value("tf"));
        ap(is_nhwc, {"--nhwc"}, ap.help("Treat tensorflow format as nhwc"), ap.set_value(true));
        ap(
            is_nhwc, {"--nchw"}, ap.help("Treat tensorflow format as nchw"), ap.set_value(false));
        ap(trim, {"--trim", "-t"}, ap.help("Trim instructions from the end"));
    }

    program load()
    {
        program p;
        if(type.empty())
        {
            if(ends_with(file, ".onnx"))
                type = "onnx";
            else
                type = "tf";
        }
        std::cout << "Reading: " << file << std::endl;
        if(type == "onnx")
            p = parse_onnx(file);
        else if(type == "tf")
            p = parse_tf(file, is_nhwc);
        if(trim > 0)
        {
            auto last = std::prev(p.end(), trim);
            p.remove_instructions(last, p.end());
        }
        return p;
    }
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
        ap(
            reduce, {"-r", "--reduce"}, ap.help("Reduce program and verify"), ap.set_value(true));
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

struct main_command
{
    static std::string get_command_help()
    {
        std::string result = "Commands:\n";
        for(const auto& p : get_commands())
            result += "    " + p.first + "\n";
        return result;
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

using namespace migraphx::driver;
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
