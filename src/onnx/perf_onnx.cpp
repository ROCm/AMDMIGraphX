
#include <migraphx/onnx.hpp>

#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/verify.hpp>

migraphx::program::parameter_map create_param_map(const migraphx::program& p, bool gpu = true)
{
    migraphx::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        if(gpu)
            m[x.first] = migraphx::gpu::to_gpu(migraphx::generate_argument(x.second));
        else
            m[x.first] = migraphx::generate_argument(x.second);
    }
    return m;
}

int main(int argc, char const* argv[])
{
    if(argc > 1)
    {
        std::string file = argv[1];
        std::size_t n    = argc > 2 ? std::stoul(argv[2]) : 50;
        auto p           = migraphx::parse_onnx(file);
        std::cout << "Compiling ... " << std::endl;
        p.compile(migraphx::gpu::target{});
        std::cout << "Allocating params ... " << std::endl;
        auto m = create_param_map(p);
        std::cout << "Running performance report ... " << std::endl;

        p.perf_report(std::cout, n, m);
        p.destroy();

    }
}
