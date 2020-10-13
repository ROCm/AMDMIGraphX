#include <test.hpp>
#include <migraphx/gpu/pack_args.hpp>

template<class T>
std::pair<std::size_t, void*> make_arg(T&& x)
{
    return {sizeof(T), &x};
}

template<class T>
std::size_t packed_sizes()
{
    return sizeof(T);
}

template<class T, class U, class... Ts>
std::size_t packed_sizes()
{
    return sizeof(T) + packed_sizes<U, Ts...>();
}

template<class... Ts>
std::size_t sizes()
{
    return migraphx::gpu::pack_args({make_arg(Ts{})...}).size();
}

template<class... Ts>
std::size_t padding()
{
    return sizes<Ts...>() - packed_sizes<Ts...>();
}

TEST_CASE(alignment)
{
    EXPECT(padding<short, short>() == 0);
    EXPECT(padding<short, int>() == 2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
