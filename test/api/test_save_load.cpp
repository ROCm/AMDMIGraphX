#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(load_save_default)
{
    std::string filename = "migraphx_api_load_save.dat";
    auto p1              = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    auto s1              = p1.get_output_shapes();

    migraphx::save(p1, filename.c_str());
    auto p2 = migraphx::load(filename.c_str());
    auto s2 = p2.get_output_shapes();
    EXPECT(s1.size() == s2.size());
    EXPECT(bool{s1.front() == s2.front()});
    EXPECT(bool{p1.sort() == p2.sort()});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
