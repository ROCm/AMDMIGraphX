#include <test.hpp>
#include <migraphx/gpu/pack_args.hpp>

template <class T>
std::size_t packed_sizes()
{
    return sizeof(T);
}

template <class T, class U, class... Ts>
std::size_t packed_sizes()
{
    return sizeof(T) + packed_sizes<U, Ts...>();
}

template <class... Ts>
std::size_t sizes()
{
    return migraphx::gpu::pack_args({Ts{}...}).size();
}

template <class... Ts>
std::size_t padding()
{
    EXPECT(sizes<Ts...>() >= packed_sizes<Ts...>());
    return sizes<Ts...>() - packed_sizes<Ts...>();
}

struct float_struct
{
    float x, y;
};

TEST_CASE(alignment_padding)
{
    EXPECT(padding<short, short>() == 0);
    EXPECT(padding<float, float_struct>() == 0);
    EXPECT(padding<short, float_struct>() == 2);
    EXPECT(padding<short, int>() == 2);
    EXPECT(padding<char, short, int, char>() == 1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
