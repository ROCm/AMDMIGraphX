
#include <migraph/onnx.hpp>

int main(int argc, char const* argv[])
{
    if(argc > 1)
    {
        std::string file = argv[1];
        auto prog        = migraph::parse_onnx(file);
        std::cout << prog << std::endl;
    }
}
