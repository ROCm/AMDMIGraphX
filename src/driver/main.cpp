#include "argument_parser.hpp"
#include "command.hpp"

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
    bool is_nhwc = false;

    void parse(argument_parser& ap)
    {
        ap.add(file, {}, ap.metavar("<input file>"));
        ap.add(type, {"--onnx"}, ap.help("Load as onnx"), ap.set_value("onnx"));
        ap.add(type, {"--tf"}, ap.help("Load as tensorflow"), ap.set_value("tf"));
        ap.add(is_nhwc, {"--nhwc"}, ap.help("Treat tensorflow format as nhwc"), ap.set_value(true));
        ap.add(is_nhwc, {"--nchw"}, ap.help("Treat tensorflow format as nchw"), ap.set_value(false));
    }

    program load() 
    {
        program p;
        if (type.empty())
        {
            if (ends_with(file, ".onnx"))
                type = "onnx";
            else
                type = "tf";
        }
        if (type == "onnx")
            p = parse_onnx(file);
        else if (type == "tf")
            p = parse_tf(file, is_nhwc);
        return p;
    }
};

struct read : command<read>
{
    loader l;
    void parse(argument_parser& ap)
    {
        l.parse(ap);
    }

    void run()
    {
        auto p = l.load();
        std::cout << p << std::endl;
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
        ap.add(nullptr, {"-h", "--help"}, ap.help("Show help"), ap.show_help(get_command_help()));
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
