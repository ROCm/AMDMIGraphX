#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(load_and_run) { auto p = migraphx::parse_onnx("conv_relu_maxpool_test.onnx"); }

int main(int argc, const char* argv[]) { test::run(argc, argv); }
