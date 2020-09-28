#include "argument_parser.hpp"
#include "command.hpp"
#include "verify.hpp"
#include "perf.hpp"
#include "models.hpp"

#include <migraphx/tf.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/json.hpp>

#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/eliminate_pad.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/rewrite_batchnorm.hpp>
#include <migraphx/simplify_algebra.hpp>
#include <migraphx/simplify_reshapes.hpp>

#include <fstream>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

struct loader
{
    std::string model;
    std::string file;
    std::string file_type;
    unsigned batch              = 1;
    bool is_nhwc                = true;
    unsigned trim               = 0;
    bool optimize               = false;
    bool skip_unknown_operators = false;
    bool brief                  = false;
    std::string output_type;
    std::string output;

    void parse(argument_parser& ap)
    {
        ap(file, {}, ap.metavar("<input file>"));
        ap(model, {"--model"}, ap.help("Load model"), ap.type("resnet50|inceptionv3|alexnet"));
        ap(file_type, {"--onnx"}, ap.help("Load as onnx"), ap.set_value("onnx"));
        ap(file_type, {"--tf"}, ap.help("Load as tensorflow"), ap.set_value("tf"));
        ap(file_type, {"--migraphx"}, ap.help("Load as MIGraphX"), ap.set_value("migraphx"));
        ap(file_type, {"--migraphx-json"}, ap.help("Load as MIGraphX JSON"), ap.set_value("json"));
        ap(batch, {"--batch"}, ap.help("Set batch size for model"));
        ap(is_nhwc, {"--nhwc"}, ap.help("Treat tensorflow format as nhwc"), ap.set_value(true));
        ap(skip_unknown_operators,
           {"--skip-unknown-operators"},
           ap.help("Skip unknown operators when parsing and continue to parse."),
           ap.set_value(true));
        ap(is_nhwc, {"--nchw"}, ap.help("Treat tensorflow format as nchw"), ap.set_value(false));
        ap(trim, {"--trim", "-t"}, ap.help("Trim instructions from the end"));
        ap(optimize, {"--optimize", "-O"}, ap.help("Optimize when reading"), ap.set_value(true));
        ap(output_type,
           {"--graphviz", "-g"},
           ap.help("Print out a graphviz representation."),
           ap.set_value("graphviz"));
        ap(brief, {"--brief"}, ap.help("Make the output brief."), ap.set_value(true));
        ap(output_type,
           {"--cpp"},
           ap.help("Print out the program as cpp program."),
           ap.set_value("cpp"));
        ap(output_type, {"--json"}, ap.help("Print out program as json."), ap.set_value("json"));
        ap(output_type,
           {"--text"},
           ap.help("Print out program in text format."),
           ap.set_value("text"));
        ap(output_type,
           {"--binary"},
           ap.help("Print out program in binary format."),
           ap.set_value("binary"));
        ap(output, {"--output", "-o"}, ap.help("Output to file."));
    }

    program load()
    {
        program p;
        if(model.empty())
        {
            if(file_type.empty())
            {
                if(ends_with(file, ".onnx"))
                    file_type = "onnx";
                else if(ends_with(file, ".pb"))
                    file_type = "tf";
                else if(ends_with(file, ".json"))
                    file_type = "json";
                else
                    file_type = "migraphx";
            }
            std::cout << "Reading: " << file << std::endl;
            if(file_type == "onnx")
            {
                onnx_options options;
                options.default_dim_value      = batch;
                options.skip_unknown_operators = skip_unknown_operators;
                options.print_program_on_error = true;
                p                              = parse_onnx(file, options);
            }
            else if(file_type == "tf")
            {
                p = parse_tf(file, tf_options{is_nhwc, batch});
            }
            else if(file_type == "json")
            {
                file_options options;
                options.format = "json";
                p              = migraphx::load(file, options);
            }
            else if(file_type == "migraphx")
            {
                p = migraphx::load(file);
            }
        }
        else
        {
            if(model == "resnet50")
                p = resnet50(batch);
            else if(model == "inceptionv3")
                p = inceptionv3(batch);
            else if(model == "alexnet")
                p = alexnet(batch);
            else
                MIGRAPHX_THROW("Unknown model: " + model);
        }
        if(trim > 0)
        {
            auto last = std::prev(p.end(), trim);
            p.remove_instructions(last, p.end());
        }
        if(optimize)
            migraphx::run_passes(p,
                                 {
                                     migraphx::rewrite_batchnorm{},
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

    static void write(std::ostream& os, const std::vector<char>& buffer)
    {
        os.write(buffer.data(), buffer.size());
    }

    void save(const program& p)
    {
        auto* os = &std::cout;
        std::ofstream fs;
        if(not output.empty())
        {
            fs.open(output);
            os = &fs;
        }

        std::string type = output_type;
        if(type.empty())
        {
            if(output.empty())
                type = "text";
            else
                type = "binary";
        }

        if(type == "cpp")
            p.print_cpp(*os);
        else if(type == "graphviz")
            p.print_graph(*os, brief);
        else if(type == "text")
            *os << p << std::endl;
        else if(type == "json")
            *os << to_json_string(p.to_value()) << std::endl;
        else if(type == "binary")
            write(*os, save_buffer(p));
    }
};

struct program_params
{
    std::vector<std::string> fill0{};
    std::vector<std::string> fill1{};
    void parse(argument_parser& ap)
    {
        ap(fill0, {"--fill0"}, ap.help("Fill parameter with 0s"), ap.append());
        ap(fill1, {"--fill1"}, ap.help("Fill parameter with 1s"), ap.append());
    }

    auto generate(const program& p, bool use_gpu)
    {
        program::parameter_map m;
        for(auto&& s : fill0)
            m[s] = fill_argument(p.get_parameter_shape(s), 0);
        for(auto&& s : fill1)
            m[s] = fill_argument(p.get_parameter_shape(s), 1);
        fill_param_map(m, p, use_gpu);
        return m;
    }
};

struct compiler
{
    static const int q_fp16 = 1;
    static const int q_int8 = 2;
    loader l;
    program_params parameters;
    bool gpu          = true;
    bool offload_copy = false;
    int quantize      = 0;

    std::vector<std::string> fill0;
    std::vector<std::string> fill1;
    void parse(argument_parser& ap)
    {
        l.parse(ap);
        parameters.parse(ap);
        ap(gpu, {"--gpu"}, ap.help("Compile on the gpu"), ap.set_value(true));
        ap(gpu, {"--cpu"}, ap.help("Compile on the cpu"), ap.set_value(false));
        ap(offload_copy,
           {"--enable-offload-copy"},
           ap.help("Enable implicit offload copying"),
           ap.set_value(true));
        ap(quantize, {"--fp16"}, ap.help("Quantize for fp16"), ap.set_value(q_fp16));
        ap(quantize, {"--int8"}, ap.help("Quantize for int8"), ap.set_value(q_int8));
    }

    auto params(const program& p, bool use_gpu = true)
    {
        return parameters.generate(p, use_gpu && gpu && !offload_copy);
    }

    program compile()
    {
        auto p = l.load();
        // Dont compile if its already been compiled
        if(p.is_compiled())
            return p;
        auto t = get_target(gpu);
        if(quantize == q_fp16)
        {
            quantize_fp16(p);
        }
        else if(quantize == q_int8)
        {
            quantize_int8(p, t, {params(p, false)});
        }
        compile_options options;
        options.offload_copy = offload_copy;
        p.compile(t, options);
        l.save(p);
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
        l.save(p);
    }
};

struct params : command<params>
{
    loader l;
    void parse(argument_parser& ap) { l.parse(ap); }

    void run()
    {
        auto p = l.load();
        for(auto&& param : p.get_parameter_shapes())
            std::cout << param.first << ": " << param.second << std::endl;
    }
};

struct verify : command<verify>
{
    loader l;
    program_params parameters;
    double tolerance     = 80;
    bool per_instruction = false;
    bool reduce          = false;
    bool offload_copy    = false;
    void parse(argument_parser& ap)
    {
        l.parse(ap);
        parameters.parse(ap);
        ap(offload_copy,
           {"--enable-offload-copy"},
           ap.help("Enable implicit offload copying"),
           ap.set_value(true));
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
        l.save(p);
        std::cout << p << std::endl;

        compile_options options;
        options.offload_copy = offload_copy;

        auto m = parameters.generate(p, false);

        if(per_instruction)
        {
            verify_instructions(p, options, tolerance);
        }
        else if(reduce)
        {
            verify_reduced_program(p, options, m, tolerance);
        }
        else
        {
            verify_program(l.file, p, options, m, tolerance);
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
        c.compile();
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

struct op : command<op>
{
    bool show_ops = false;
    void parse(argument_parser& ap)
    {
        ap(show_ops,
           {"--list", "-l"},
           ap.help("List all the operators of MIGraphX"),
           ap.set_value(true));
    }
    void run() const
    {
        if(show_ops)
        {
            for(const auto& name : get_operators())
                std::cout << name << std::endl;
        }
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
