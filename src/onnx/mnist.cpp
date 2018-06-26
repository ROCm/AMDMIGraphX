#include <cstdio>
#include <string>
#include <fstream>
#include <stdexcept>

#include <rtg/onnx.hpp>

#include <rtg/cpu/cpu_target.hpp>
#include <rtg/generate.hpp>

std::vector<float> read_mnist_images(std::string full_path, int& number_of_images, int& image_size)
{
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
        return (static_cast<int>(c1) << 24) + (static_cast<int>(c2) << 16) +
               (static_cast<int>(c3) << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open())
    {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051)
            throw std::runtime_error("Invalid MNIST image file!");

        file.read((char*)&number_of_images, sizeof(number_of_images)),
            number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        printf("n_rows: %d    n_cols: %d    image_size: %d\n\n", n_rows, n_cols, image_size);

        // uchar** _dataset = new uchar*[number_of_images];
        // for(int i = 0; i < number_of_images; i++) {
        //     _dataset[i] = new uchar[image_size];
        //     file.read((char *)_dataset[i], image_size);
        // }

        std::vector<float> result(number_of_images * image_size);
        for(int i = 0; i < number_of_images; i++)
        {
            for(int j = 0; j < image_size; j++)
            {
                uchar tmp;
                file.read((char*)&tmp, 1);
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

std::vector<int32_t> read_mnist_labels(std::string full_path, int& number_of_labels)
{
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
        return (static_cast<int>(c1) << 24) + (static_cast<int>(c2) << 16) +
               (static_cast<int>(c3) << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open())
    {
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049)
            throw std::runtime_error("Invalid MNIST label file!");

        file.read((char*)&number_of_labels, sizeof(number_of_labels)),
            number_of_labels = reverseInt(number_of_labels);

        std::vector<int32_t> result(number_of_labels);
        for(int i = 0; i < number_of_labels; i++)
        {
            uchar tmp;
            file.read((char*)&tmp, 1);
            result[i] = tmp;
        }
        return result;
    }
    else
    {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

std::vector<float> softmax(std::vector<float> p) {
    size_t n = p.size();
    std::vector<float> result(n);
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) {
        result[i] = std::exp(p[i]);
        s += result[i];
    }
    for (size_t i = 0; i < n; i++) {
        result[i] = result[i]/s;
    }
    return result;
}

int main(int argc, char const* argv[])
{
    if(argc > 1)
    {
        std::string datafile        = argv[2];
        std::string labelfile       = argv[3];
        int nimages                 = -1;
        int image_size              = -1;
        int nlabels                 = -1;
        std::vector<float> input    = read_mnist_images(datafile, nimages, image_size);
        std::vector<int32_t> labels = read_mnist_labels(labelfile, nlabels);

        std::string file = argv[1];
        auto prog        = rtg::parse_onnx(file);
        prog.compile(rtg::cpu::cpu_target{});
        // auto s = prog.get_parameter_shape("Input3");
        auto s = rtg::shape{rtg::shape::float_type, {1, 1, 28, 28}};
        std::cout << s << std::endl;
        auto ptr = input.data();
        for (int i = 0; i < 20; i++)
        {
            printf("label: %d  ---->  ", labels[i]);
            auto input3 = rtg::argument{s, &ptr[784*i]};
            auto result = prog.eval({{"Input3", input3}});
            std::vector<float> logits;
            result.visit([&](auto output) { logits.assign(output.begin(), output.end()); });
            std::vector<float> probs = softmax(logits);
            for (auto x : probs) printf("%8.4f    ", x);
            printf("\n");
        }
        printf("\n");
    }
}
