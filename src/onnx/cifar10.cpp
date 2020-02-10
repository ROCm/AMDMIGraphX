#include <cstdio>
#include <string>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include <migraphx/onnx.hpp>

#include <migraphx/cpu/target.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/generate.hpp>

#include "softmax.hpp"

auto read_cifar10_images(const std::string& full_path)
{
    std::ifstream file(full_path, std::ios::binary);

    const size_t nimages          = 10;
    const size_t nbytes_per_image = 3072;
    std::vector<uint8_t> raw_data(nimages * (nbytes_per_image + 1));
    std::vector<uint8_t> labels(nimages);
    std::vector<float> data(nimages * nbytes_per_image);
    if(file.is_open())
    {
        file.read(reinterpret_cast<char*>(raw_data.data()),
                  (nbytes_per_image + 1) * nimages * sizeof(uint8_t));
        uint8_t* pimage = raw_data.data();
        for(size_t i = 0; i < nimages; i++, pimage += nbytes_per_image)
        {
            labels[i] = *pimage++;
            for(size_t j = 0; j < nbytes_per_image; j++)
            {
                float v                        = float(*(pimage + j)) / 255.0f;
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

int main(int argc, char const* argv[])
{
    if(argc < 4)
    {
        throw std::runtime_error("Usage:  cifar10 [gpu | cpu] <onnx file> <cifar10 data file>");
    }
    std::string gpu_cpu  = argv[1];
    std::string file     = argv[2];
    std::string datafile = argv[3];
    auto prog            = migraphx::parse_onnx(file);
    std::cout << prog << std::endl;
    auto imageset = read_cifar10_images(datafile);

    if(gpu_cpu == "gpu")
    {
        // GPU target
        prog.compile(migraphx::gpu::target{});
        migraphx::program::parameter_map m;
        auto s = migraphx::shape{migraphx::shape::float_type, {1, 3, 32, 32}};
        for(auto&& x : prog.get_parameter_shapes())
        {
            m[x.first] = migraphx::gpu::to_gpu(migraphx::generate_argument(x.second));
        }
        auto labels = imageset.first;
        auto input  = imageset.second;
        auto ptr    = input.data();
        for(int i = 0; i < 10; i++)
        {
            std::cout << "label: " << static_cast<uint32_t>(labels[i]) << "  ---->  ";
            m["0"]          = migraphx::gpu::to_gpu(migraphx::argument{s, &ptr[3072 * i]});
            auto gpu_result = prog.eval(m).back();
            auto result     = migraphx::gpu::from_gpu(gpu_result);
            std::vector<float> logits;
            result.visit([&](auto output) { logits.assign(output.begin(), output.end()); });
            std::vector<float> probs = softmax<float>(logits);
            for(auto x : probs)
                std::cout << x << "    ";
            std::cout << std::endl << std::endl;
        }
    }
    else
    {
        // CPU target
        prog.compile(migraphx::cpu::target{});
        auto s      = migraphx::shape{migraphx::shape::float_type, {1, 3, 32, 32}};
        auto labels = imageset.first;
        auto input  = imageset.second;
        auto ptr    = input.data();
        for(int i = 0; i < 10; i++)
        {
            std::cout << "label: " << static_cast<uint32_t>(labels[i]) << "  ---->  ";
            auto input3 = migraphx::argument{s, &ptr[3072 * i]};
            auto result = prog.eval({{"0", input3}}).back();
            std::vector<float> logits;
            result.visit([&](auto output) { logits.assign(output.begin(), output.end()); });
            std::vector<float> probs = softmax<float>(logits);
            for(auto x : probs)
                std::cout << x << "    ";
            std::cout << std::endl;
        }
    }
}
