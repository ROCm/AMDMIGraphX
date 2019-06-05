#include "argument_parser.hpp"
#include "command.hpp"

struct main_command
{
    static std::string get_command_help()
    {
        std::string result = "Commands:\n";
        for(const auto& p : migraphx::driver::get_commands())
            result += "    " + p.first + "\n";
        return result;
    }
    void parse(migraphx::driver::argument_parser& ap)
    {
        ap.add(nullptr, {"-h", "--help"}, ap.help("Show help"), ap.show_help(get_command_help()));
    }

    void run() {}
};

int main(int argc, const char* argv[])
{
    std::vector<std::string> args(argv + 1, argv + argc);
    if(args.empty())
        return 0;
    auto&& m = migraphx::driver::get_commands();
    auto cmd = args.front();
    if(m.count(cmd) > 0)
    {
        m.at(cmd)({args.begin() + 1, args.end()});
    }
    else
    {
        migraphx::driver::argument_parser ap;
        main_command mc;
        mc.parse(ap);
        ap.parse(args);
        mc.run();
    }
    return 0;
}
