#include <migraphx/bit_signal.hpp>
#include <test.hpp>

TEST_CASE(triggered)
{
    migraphx::bit_signal<64> signals;
    auto slot = signals.subscribe();
    signals.notify();
    EXPECT(slot.triggered());
}

TEST_CASE(not_triggered)
{
    migraphx::bit_signal<64> signals;
    auto slot = signals.subscribe();
    EXPECT(not slot.triggered());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
