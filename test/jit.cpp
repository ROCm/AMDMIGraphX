#include <migraphx/compile_src.hpp>
#include <migraphx/dynamic_loader.hpp>
#include <migraphx/cpp_generator.hpp>
#include <test.hpp>

// NOLINTNEXTLINE
const std::string add_42_src = R"migraphx(
extern "C" int add(int x)
{
    return x+42;
}
)migraphx";

template <class F>
std::function<F>
compile_function(const std::string& src, const std::string& flags, const std::string& fname)
{
    migraphx::src_compiler compiler;
    compiler.flags  = flags + "-fPIC -shared";
    compiler.output = "libsimple.so";
    migraphx::src_file f;
    f.path     = "main.cpp";
    f.content  = std::make_pair(src.data(), src.data() + src.size());
    auto image = compiler.compile({f});
    return migraphx::dynamic_loader{image}.get_function<F>(fname);
}

TEST_CASE(simple_run)
{
    auto f = compile_function<int(int)>(add_42_src, "", "add");
    EXPECT(f(8) == 50);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
