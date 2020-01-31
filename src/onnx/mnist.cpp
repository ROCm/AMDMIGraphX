#include <cstdio>
#include <string>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include <migraphx/onnx.hpp>

#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/generate.hpp>

#include "softmax.hpp"

auto reverse_int(unsigned int i)
{
    unsigned char c1;
    unsigned char c2;
    unsigned char c3;
    unsigned char c4;
    c1 = i & 255u;
    c2 = (i >> 8u) & 255u;
    c3 = (i >> 16u) & 255u;
    c4 = (i >> 24u) & 255u;
    return (static_cast<unsigned int>(c1) << 24u) + (static_cast<unsigned int>(c2) << 16u) +
           (static_cast<unsigned int>(c3) << 8u) + c4;
};

std::vector<float>
read_mnist_images(const std::string& full_path, int& number_of_images, int& image_size)
{
    using uchar = unsigned char;

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open())
    {
        int magic_number = 0;
        int n_rows       = 0;
        int n_cols       = 0;

        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = reverse_int(magic_number);

        if(magic_number != 2051)
            throw std::runtime_error("Invalid MNIST image file!");

        file.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
        file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
        n_rows = reverse_int(n_rows);
        file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));
        n_cols = reverse_int(n_cols);

        image_size = n_rows * n_cols;

        std::vector<float> result(number_of_images * image_size);
        for(int i = 0; i < number_of_images; i++)
        {
            for(int j = 0; j < image_size; j++)
            {
                uchar tmp;
                file.read(reinterpret_cast<char*>(&tmp), 1);
                result[i * image_size + j] = tmp / 255.0;
            }
        }
        return result;
    }
    else
    {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}

std::vector<int32_t> read_mnist_labels(const std::string& full_path, int& number_of_labels)
{
    using uchar = unsigned char;

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open())
    {
        int magic_number = 0;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = reverse_int(magic_number);

        if(magic_number != 2049)
            throw std::runtime_error("Invalid MNIST label file!");

        file.read(reinterpret_cast<char*>(&number_of_labels), sizeof(number_of_labels));
        number_of_labels = reverse_int(number_of_labels);

        std::vector<int32_t> result(number_of_labels);
        for(int i = 0; i < number_of_labels; i++)
        {
            uchar tmp;
            file.read(reinterpret_cast<char*>(&tmp), 1);
            result[i] = tmp;
        }
        return result;
    }
    else
    {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

int main(int argc, char const* argv[])
{
    if(argc > 3)
    {
        std::string datafile        = argv[2];
        std::string labelfile       = argv[3];
        int nimages                 = -1;
        int image_size              = -1;
        int nlabels                 = -1;
        std::vector<float> input    = read_mnist_images(datafile, nimages, image_size);
        std::vector<int32_t> labels = read_mnist_labels(labelfile, nlabels);

        std::string file = argv[1];
        auto prog        = migraphx::parse_onnx(file);
        std::cout << prog << std::endl << std::endl;
        prog.compile(migraphx::gpu::target{});
        auto s = migraphx::shape{migraphx::shape::float_type, {1, 1, 28, 28}};
        std::cout << s << std::endl;
        auto ptr = input.data();
        migraphx::program::parameter_map m;
        m["output"] =
            migraphx::gpu::to_gpu(migraphx::generate_argument(prog.get_parameter_shape("output")));
        for(int i = 0; i < 20; i++)
        {
            std::cout << "label: " << labels[i] << "  ---->  ";
            m["0"]      = migraphx::gpu::to_gpu(migraphx::argument{s, &ptr[784 * i]});
            auto results = prog.eval(m);
            auto result = migraphx::gpu::from_gpu(results.back());
            std::vector<float> logits;
            result.visit([&](auto output) { logits.assign(output.begin(), output.end()); });
            std::vector<float> probs = softmax(logits);
            for(auto x : probs)
                std::cout << x << "  ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}
