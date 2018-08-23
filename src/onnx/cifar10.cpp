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
    if (argc < 4)
    {
        throw std::runtime_error("Usage:  cifar10 [gpu | cpu] <onnx file> <cifar10 data file>");
    }
    std::string gpu_cpu  = argv[1];
    std::string file     = argv[2];
    std::string datafile = argv[3];
    auto prog            = migraph::parse_onnx(file);
    std::cout << prog << std::endl;
    auto imageset = read_cifar10_images(datafile);

    if (gpu_cpu == "gpu")
    {
        // GPU target
        prog.compile(migraph::gpu::target{});
        migraph::program::parameter_map m;
        auto s = migraph::shape{migraph::shape::float_type, {1, 3, 32, 32}};
        for(auto&& x : prog.get_parameter_shapes())
        {
            m[x.first] = migraph::gpu::to_gpu(migraph::generate_argument(x.second));
        }
        auto labels = imageset.first;
        auto input  = imageset.second;
        auto ptr    = input.data();
        for(int i = 0; i < 10; i++)
        {
            std::cout << "label: " << static_cast<uint32_t>(labels[i]) << "  ---->  ";
            m["0"]      = migraph::gpu::to_gpu(migraph::argument{s, &ptr[3072 * i]});
            auto result = migraph::gpu::from_gpu(prog.eval(m));
            std::vector<float> logits;
            result.visit([&](auto output) { logits.assign(output.begin(), output.end()); });
            std::vector<float> probs = softmax(logits);
            for(auto x : probs)
                std::cout << x << "    ";
            std::cout << std::endl << std::endl;
        }
    }
    else
    {
        // CPU target
        prog.compile(migraph::cpu::cpu_target{});
        auto s      = migraph::shape{migraph::shape::float_type, {1, 3, 32, 32}};
        auto labels = imageset.first;
        auto input  = imageset.second;
        auto ptr    = input.data();
        for(int i = 0; i < 10; i++)
        {
            std::cout << "label: " << static_cast<uint32_t>(labels[i]) << "  ---->  ";
            auto input3 = migraph::argument{s, &ptr[3072 * i]};
            auto result = prog.eval({{"0", input3}});
            std::vector<float> logits;
            result.visit([&](auto output) { logits.assign(output.begin(), output.end()); });
            std::vector<float> probs = softmax(logits);
            for(auto x : probs)
                std::cout << x << "    ";
            std::cout << std::endl;
        }
    }
}
