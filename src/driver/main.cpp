/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include "verify.hpp"
#include "argument_parser.hpp"
#include "command.hpp"
#include "precision.hpp"
#include "perf.hpp"
#include "models.hpp"
#include "marker_roctx.hpp"

#include <migraphx/tf.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/json.hpp>
#include <migraphx/version.h>

#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/eliminate_pad.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/simplify_algebra.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/register_target.hpp>

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
    std::vector<std::string> param_dims;
    std::vector<std::string> output_names;

    void parse(argument_parser& ap)
    {
        ap(file, {}, ap.metavar("<input file>"), ap.file_exist(), ap.required(), ap.group("input"));
        ap(model,
           {"--model"},
           ap.help("Load model"),
           ap.type("resnet50|inceptionv3|alexnet"),
           ap.group("input"));
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
        ap(param_dims,
           {"--input-dim"},
           ap.help("Dim of a parameter (format: \"@name d1 d2 dn\")"),
           ap.append(),
           ap.nargs(2));

        ap(output_names,
           {"--output-names"},
           ap.help("Names of node output (format: \"name_1 name_2 name_n\")"),
           ap.append(),
           ap.nargs(2));
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

    static auto parse_param_dims(const std::vector<std::string>& param_dims_info)
    {
        std::unordered_map<std::string, std::vector<std::size_t>> map_input_dims;
        std::string name = "";
        for(auto&& x : param_dims_info)
        {
            if(x[0] == '@')
            {
                name = x.substr(1);
            }
            else
            {
                map_input_dims[name].push_back(value_parser<std::size_t>::apply(x));
            }
        }

        return map_input_dims;
    }

    static auto parse_output_names(const std::vector<std::string>& output_names_info)
    {
        std::vector<std::string> output_node_names;
        std::transform(output_names_info.begin(),
                       output_names_info.end(),
                       std::back_inserter(output_node_names),
                       [&](auto x) { return value_parser<std::string>::apply(x); });

        return output_node_names;
    }

    program load()
    {
        program p;
        if(model.empty())
        {
            auto map_input_dims    = parse_param_dims(param_dims);
            auto output_node_names = parse_output_names(output_names);
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
                options.map_input_dims         = map_input_dims;
                p                              = parse_onnx(file, options);
            }
            else if(file_type == "tf")
            {
                p = parse_tf(file, tf_options{is_nhwc, batch, map_input_dims, output_node_names});
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
            auto* mm  = p.get_main_module();
            auto last = std::prev(mm->end(), trim);
            mm->remove_instructions(last, mm->end());
        }
        // Remove unused variable when exporting to cpp
        if(output_type == "cpp")
            migraphx::run_passes(*p.get_main_module(), {migraphx::dead_code_elimination{}});
        if(optimize)
        {
            migraphx::run_passes(*p.get_main_module(),
                                 {
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
        }
        return p;
    }

    static void write(std::ostream& os, const std::vector<char>& buffer)
    {
        os.write(buffer.data(), buffer.size());
    }

    void save(const program& p) const
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
        ap(fill0, {"--fill0"}, ap.help("Fill parameter with 0s"), ap.append(), ap.nargs(2));
        ap(fill1, {"--fill1"}, ap.help("Fill parameter with 1s"), ap.append(), ap.nargs(2));
    }

    auto generate(const program& p, const target& t, bool offload)
    {
        parameter_map m;
        for(auto&& s : fill0)
            m[s] = fill_argument(p.get_parameter_shape(s), 0);
        for(auto&& s : fill1)
            m[s] = fill_argument(p.get_parameter_shape(s), 1);
        fill_param_map(m, p, t, offload);
        return m;
    }
};

struct compiler_target
{
#ifdef HAVE_GPU
    std::string target_name = "gpu";
#else
    std::string target_name = "cpu";
#endif

    void parse(argument_parser& ap)
    {
        ap(target_name, {"--gpu"}, ap.help("Compile on the gpu"), ap.set_value("gpu"));
        ap(target_name, {"--cpu"}, ap.help("Compile on the cpu"), ap.set_value("cpu"));
        ap(target_name,
           {"--ref"},
           ap.help("Compile on the reference implementation"),
           ap.set_value("ref"));
    }

    target get_target() const { return make_target(target_name); }
};

struct compiler
{
    loader l;
    program_params parameters;
    compiler_target ct;
    bool offload_copy  = false;
    bool fast_math     = true;
    precision quantize = precision::fp32;

    std::vector<std::string> fill0;
    std::vector<std::string> fill1;
    void parse(argument_parser& ap)
    {
        l.parse(ap);
        parameters.parse(ap);
        ct.parse(ap);
        ap(offload_copy,
           {"--enable-offload-copy"},
           ap.help("Enable implicit offload copying"),
           ap.set_value(true));
        ap(fast_math,
           {"--disable-fast-math"},
           ap.help("Disable fast math optimization"),
           ap.set_value(false));
        ap(quantize, {"--fp16"}, ap.help("Quantize for fp16"), ap.set_value(precision::fp16));
        ap(quantize, {"--int8"}, ap.help("Quantize for int8"), ap.set_value(precision::int8));
    }

    auto params(const program& p) { return parameters.generate(p, ct.get_target(), offload_copy); }

    program compile()
    {
        auto p = l.load();
        // Dont compile if its already been compiled
        if(p.is_compiled())
            return p;
        auto t = ct.get_target();
        if(quantize == precision::fp16)
        {
            quantize_fp16(p);
        }
        else if(quantize == precision::int8)
        {
            quantize_int8(p, t, {params(p)});
        }
        compile_options options;
        options.offload_copy = offload_copy;
        options.fast_math    = fast_math;
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
    compiler_target ct;
    double tolerance     = 80;
    bool per_instruction = false;
    bool reduce          = false;
    bool offload_copy    = false;
    bool fast_math       = true;
    precision quantize   = precision::fp32;
    void parse(argument_parser& ap)
    {
        l.parse(ap);
        parameters.parse(ap);
        ct.parse(ap);
        ap(offload_copy,
           {"--enable-offload-copy"},
           ap.help("Enable implicit offload copying"),
           ap.set_value(true));
        ap(fast_math,
           {"--disable-fast-math"},
           ap.help("Disable fast math optimization"),
           ap.set_value(false));
        ap(tolerance, {"--tolerance"}, ap.help("Tolerance for errors"));
        ap(per_instruction,
           {"-i", "--per-instruction"},
           ap.help("Verify each instruction"),
           ap.set_value(true));
        ap(reduce, {"-r", "--reduce"}, ap.help("Reduce program and verify"), ap.set_value(true));
        ap(quantize, {"--fp16"}, ap.help("Quantize for fp16"), ap.set_value(precision::fp16));
    }

    void run()
    {
        auto p = l.load();
        l.save(p);
        std::cout << p << std::endl;

        compile_options options;
        options.offload_copy = offload_copy;
        options.fast_math    = fast_math;
        auto t               = ct.get_target();
        auto m               = parameters.generate(p, t, true);

        if(per_instruction)
        {
            verify_instructions(p, t, options, quantize, tolerance);
        }
        else if(reduce)
        {
            verify_reduced_program(p, t, options, quantize, m, tolerance);
        }
        else
        {
            verify_program(l.file, p, t, options, quantize, m, tolerance);
        }
    }
};

struct version : command<version>
{
    void parse(const argument_parser&) {}
    void run() const
    {
        std::cout << "MIGraphX Version: " << MIGRAPHX_VERSION_MAJOR << "." << MIGRAPHX_VERSION_MINOR
                  << std::endl;
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
        p.perf_report(std::cout, n, m, c.l.batch);
    }
};

struct roctx : command<roctx>
{
    compiler c;
    void parse(argument_parser& ap) { c.parse(ap); }

    void run()
    {
        std::cout << "Compiling ... " << std::endl;
        auto p = c.compile();
        std::cout << "Allocating params ... " << std::endl;
        auto m = c.params(p);
        std::cout << "rocTX:\tLoading rocTX library..." << std::endl;
        auto rtx = create_marker_roctx();
        p.mark(m, std::move(rtx));
    }
};

struct op : command<op>
{
    bool show_ops = false;
    std::string op_name{};
    void parse(argument_parser& ap)
    {
        ap(op_name, {}, ap.metavar("<MIGraphX operator name>"));
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
        else
        {
            auto op = load_op(op_name);
            std::cout << op_name << ": " << std::endl;
            std::cout << to_pretty_json_string(op.to_value()) << std::endl;
        }
    }
};

struct onnx : command<onnx>
{
    bool show_ops = false;
    void parse(argument_parser& ap)
    {
        ap(show_ops,
           {"--list", "-l"},
           ap.help("List all onnx operators supported by MIGraphX"),
           ap.set_value(true));
    }
    void run() const
    {
        if(show_ops)
        {
            for(const auto& name : get_onnx_operators())
                std::cout << name << std::endl;
        }
    }
};

struct main_command
{
    static std::string get_command_help(const std::string& title = colorize(color::fg_yellow,
                                                                            "COMMANDS:"))
    {
        std::string result = title + "\n";
        std::vector<std::string> commands(get_commands().size());
        std::transform(get_commands().begin(),
                       get_commands().end(),
                       commands.begin(),
                       [](const auto& p) { return colorize(color::fg_green, p.first); });
        std::sort(commands.begin(), commands.end());
        return std::accumulate(commands.begin(), commands.end(), result, [](auto r, auto&& s) {
            return r + "    " + s + "\n";
        });
    }
    void parse(argument_parser& ap)
    {
        std::string version_str = "MIGraphX Version: " + std::to_string(MIGRAPHX_VERSION_MAJOR) +
                                  "." + std::to_string(MIGRAPHX_VERSION_MINOR);
        ap(wrong_commands, {}, ap.metavar("<command>"), ap.append());
        ap(nullptr, {"-h", "--help"}, ap.help("Show help"), ap.show_help(get_command_help()));
        ap(nullptr,
           {"-v", "--version"},
           ap.help("Show MIGraphX version"),
           ap.show_help(version_str));

        // Trim command off of exe name
        ap.set_exe_name(ap.get_exe_name().substr(0, ap.get_exe_name().size() - 5));
        ap.set_exe_name_to(exe_name);
    }

    std::vector<std::string> wrong_commands{};
    std::string exe_name = "<exe>";

    void run()
    {
        std::cout << color::fg_red << color::bold << "error: " << color::reset;
        auto it = std::find_if(wrong_commands.begin(), wrong_commands.end(), [](const auto& c) {
            return get_commands().count(c) > 0;
        });
        if(it == wrong_commands.end())
        {
            std::cout << "'" << color::fg_yellow << wrong_commands.front() << color::reset
                      << "' is not a valid command." << std::endl;
            std::cout << get_command_help("Available commands:") << std::endl;
        }
        else
        {
            std::cout << "command '" << color::fg_yellow << *it << color::reset
                      << "' must be first argument" << std::endl;
            std::cout << std::endl;

            std::cout << color::fg_yellow << "USAGE:" << color::reset << std::endl;
            std::cout << "    " << exe_name << " " << *it << " <options>" << std::endl;
        }
        std::cout << std::endl;
    }
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx

using namespace migraphx::driver; // NOLINT
int main(int argc, const char* argv[])
{
    std::vector<std::string> args(argv + 1, argv + argc);

    // no argument, print the help infomration by default
    if(args.empty())
    {
        args.push_back("-h");
    }

    auto&& m = get_commands();
    auto cmd = args.front();
    if(m.count(cmd) > 0)
    {
        m.at(cmd)(argv[0], {args.begin() + 1, args.end()});
    }
    else
    {
        run_command<main_command>(argv[0], args);
    }

    return 0;
}
