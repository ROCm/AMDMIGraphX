#include <migraphx/tf.hpp>

int main(int argc, char const* argv[])
{
    if(argc > 1)
    {
        bool is_nhwc = true;
        if(argc > 2)
        {
            if(strcmp(argv[2], "nchw") == 0)
                is_nhwc = false;
        }
        std::string file = argv[1];
        auto prog        = migraphx::parse_tf(file, is_nhwc);
        std::cout << prog << std::endl;
    }
}
