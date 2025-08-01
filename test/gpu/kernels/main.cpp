
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/kernel.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <kernel_tests.hpp>
#include <test.hpp>

#include <map>
#include <regex>
#include <string_view>

static migraphx::src_file make_src_file(const std::string& name, const std::string& content)
{
    return {name, content};
}

std::vector<std::string> parse_cases(const std::string_view& content)
{
    std::regex case_re(R"(TEST_CASE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\))");
    std::match_results<std::string_view::const_iterator> m;
    std::vector<std::string> test_names;

    auto it = content.cbegin();
    while(std::regex_search(it, content.cend(), m, case_re))
    {
        test_names.push_back(m[1].str());
        it = m.suffix().first;
    }
    return test_names;
}

struct test_suite : std::enable_shared_from_this<test_suite>
{
    std::string name;
    std::string_view content;
    std::map<std::string, int> test_cases;
    migraphx::gpu::hip_compile_options options = {};
    migraphx::gpu::kernel k;

    test_suite(const std::string_view& src_name, const std::string_view& src_content)
        : name(src_name.substr(0, src_name.size() - 4)), content(src_content)
    {
        auto cases = parse_cases(src_content);
        for(std::size_t i = 0; i < cases.size(); ++i)
        {
            test_cases[name + "." + cases[i]] = i;
        }

        migraphx::gpu::context ctx;
        options.global      = 1;
        options.local       = ctx.get_current_device().get_wavefront_size();
        options.kernel_name = "gpu_test_kernel";
    }

    std::string generate_source() const
    {
        std::ostringstream out;
        out << content << '\n';
        out << "extern \"C\" __global__ void " << options.kernel_name << "(int id) {\n";
        out << "    switch(id) {\n";
        for(const auto& [case_name, i] : test_cases)
        {
            auto fname = case_name.substr(name.size() + 1);
            out << "        case " << i << ": " << fname << "(); break;\n";
        }
        out << "        default: abort();\n";
        out << "    }\n";
        out << "}\n";
        return out.str();
    }

    void compile()
    {
        if(not k.empty())
            return;
        migraphx::gpu::context ctx;
        auto binary = migraphx::gpu::compile_hip_raw(ctx, generate_source(), options);

        k = {binary, options.kernel_name};
    }

    void run(const std::string& case_name)
    {
        compile();
        k.launch(nullptr, options.global, options.local)(test_cases.at(case_name));
        CHECK(hipDeviceSynchronize() == hipSuccess);
    }
};

int main(int argc, const char* argv[])
{
    test::driver d{};
    for(auto [name, content] : ::kernel_tests())
    {
        auto ts = std::make_shared<test_suite>(name, content);
        for(auto&& p : ts->test_cases)
        {
            auto case_name = p.first;
            test::add_test_case(case_name, [ts, case_name] { ts->run(case_name); });
        }
    }
    d.run(argc, argv);
}
