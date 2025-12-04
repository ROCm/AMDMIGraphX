/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 */

#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/kernel.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/par_for.hpp>
#include <kernel_tests.hpp>
#include <test.hpp>

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <string_view>

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
        out << "extern \"C\" __global__ void " << options.kernel_name
            << "(int id, migraphx::int32_t* failures) {\n";
        out << "    migraphx::test::test_manager tm{failures};\n";
        out << "    switch(id) {\n";
        for(const auto& [case_name, i] : test_cases)
        {
            auto fname = case_name.substr(name.size() + 1);
            out << "        case " << i << ": " << fname << "(tm); break;\n";
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
        auto failures = migraphx::gpu::write_to_gpu(int32_t{0}, true);
        compile();
        k.launch(nullptr, options.global, options.local)(test_cases.at(case_name), failures.get());
        CHECK(hipDeviceSynchronize() == hipSuccess);
        test::report_failure(*failures);
    }
};

int main(int argc, const char* argv[])
{
    test::driver d{};
    std::unordered_map<std::string, std::shared_ptr<test_suite>> suites;
    for(auto [name, content] : ::kernel_tests())
    {
        auto ts = std::make_shared<test_suite>(name, content);
        for(auto&& p : ts->test_cases)
        {
            auto case_name    = p.first;
            suites[case_name] = ts;
            test::add_test_case(case_name, [ts, case_name] { ts->run(case_name); });
        }
    }
    d.on_selected_cases = [&](const std::vector<std::string>& cases) {
        std::unordered_set<std::shared_ptr<test_suite>> run_suites;
        for(auto&& name : cases)
        {
            if(suites.count(name) == 0)
                continue;
            run_suites.insert(suites.at(name));
        }
        migraphx::par_for(run_suites.size(), 1, [&](auto i) {
            auto it = run_suites.begin();
            std::advance(it, i);
            (*it)->compile();
        });
    };
    d.run(argc, argv);
}
