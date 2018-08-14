#include <cstdio>
#include <string>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include <migraph/onnx.hpp>

#include <migraph/cpu/cpu_target.hpp>
#include <migraph/gpu/target.hpp>
#include <migraph/gpu/hip.hpp>
#include <migraph/generate.hpp>

auto read_cifar10_images(std::string full_path)
{

    std::ifstream file(full_path, std::ios::binary);

    const size_t nimages          = 10;
    const size_t nbytes_per_image = 3072;
    std::vector<int8_t> raw_data(nimages * (nbytes_per_image + 1));
    std::vector<int8_t> labels(nimages);
    std::vector<float> data(nimages * nbytes_per_image);
    if(file.is_open())
    {
        file.read(reinterpret_cast<char*>(raw_data.data()),
                  (nbytes_per_image + 1) * nimages * sizeof(int8_t));
        int8_t* pimage = raw_data.data();
        for(size_t i = 0; i < nimages; i++, pimage += nbytes_per_image)
        {
            labels[i] = *pimage++;
            for(size_t j = 0; j < nbytes_per_image; j++)
            {
                float v                        = *(pimage + j) / 255.0f;
                data[i * nbytes_per_image + j] = v;
            }
        }
        return std::make_pair(labels, data);
    }
    else
    {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}

std::vector<float> softmax(std::vector<float> p)
{
    size_t n = p.size();
    std::vector<float> result(n);
    std::transform(p.begin(), p.end(), result.begin(), [](auto x) { return std::exp(x); });
    float s = std::accumulate(result.begin(), result.end(), 0.0f, std::plus<float>());
    std::transform(result.begin(), result.end(), result.begin(), [=](auto x) { return x / s; });
    return result;
}

int main(int argc, char const* argv[])
{
    std::string file     = argv[1];
    std::string datafile = argv[2];
    auto prog            = migraph::parse_onnx(file);

    auto imageset = read_cifar10_images(datafile);

    // // GPU target
    // prog.compile(migraph::gpu::target{});
    // migraph::program::parameter_map m;
    // auto s = migraph::shape{migraph::shape::float_type, {1, 3, 32, 32}};
    // m["output"] =
    //     migraph::gpu::to_gpu(migraph::generate_argument(prog.get_parameter_shape("output")));
    // auto labels = imageset.first;
    // auto input  = imageset.second;
    // auto ptr    = input.data();
    // for(int i = 0; i < 10; i++)
    // {
    //     std::cout << "label: " << (uint32_t)labels[i] << "  ---->  ";
    //     m["0"]      = migraph::gpu::to_gpu(migraph::argument{s, &ptr[3072 * i]});
    //     auto result = migraph::gpu::from_gpu(prog.eval(m));
    //     std::vector<float> logits;
    //     result.visit([&](auto output) { logits.assign(output.begin(), output.end()); });
    //     std::vector<float> probs = softmax(logits);
    //     for(auto x : logits)
    //         std::cout << x << "  ";
    //     std::cout << std::endl;
    // }

    // // CPU target
    // prog.compile(migraph::cpu::cpu_target{});
    // auto s = migraph::shape{migraph::shape::float_type, {1, 3, 32, 32}};
    // auto input3 = migraph::generate_argument(s, 12345);
    // auto result = prog.eval({{"0", input3}});
    prog.compile(migraph::cpu::cpu_target{});
    auto s = migraph::shape{migraph::shape::float_type, {1, 3, 32, 32}};
    auto labels = imageset.first;
    auto input  = imageset.second;
    auto ptr    = input.data();
    for(int i = 0; i < 10; i++)
    {
        std::cout << "label: " << (uint32_t)labels[i] << "  ---->  ";
        auto input3 = migraph::argument{s, &ptr[3072 * i]};
        auto result = prog.eval({{"0", input3}});
        std::vector<float> logits;
        result.visit([&](auto output) { logits.assign(output.begin(), output.end()); });
        std::vector<float> probs = softmax(logits);
        for(auto x : logits)
            std::cout << x << "  ";
        std::cout << std::endl;
    }

}
