#include <migraphx/analyze_streams.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/context.hpp>
#include "test.hpp"
#include "basic_ops.hpp"

struct record_event
{
    std::size_t event = 0;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.event, "event"));
    }
    std::string name() const { return "record_event"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const { return {}; }

    migraphx::argument compute(const migraphx::shape&, const std::vector<migraphx::argument>&) const
    {
        return {};
    }
};

struct wait_event
{
    std::size_t event = 0;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.event, "event"));
    }
    std::string name() const { return "wait_event"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const { return {}; }

    migraphx::argument compute(const migraphx::shape&, const std::vector<migraphx::argument>&) const
    {
        return {};
    }
};

struct set_stream
{
    std::size_t stream = 0;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.stream, "stream"));
    }
    std::string name() const { return "set_stream"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const { return {}; }

    migraphx::argument compute(const migraphx::shape&, const std::vector<migraphx::argument>&) const
    {
        return {};
    }
};

struct test_stream_model
{
    std::size_t nstreams = 0;
    std::unordered_map<migraphx::instruction_ref, std::size_t> ins2stream{};
    std::size_t get_nstream() const
    {
        return nstreams;
    }
    std::size_t get_stream(migraphx::instruction_ref ins) const
    {
        return ins2stream.at(ins);
    }
    std::size_t get_event_id(migraphx::instruction_ref ins) const
    {
        auto v = ins->get_operator().to_value();
        return v["event"].to<std::size_t>();
    }
    bool has_stream(migraphx::instruction_ref ins) const
    {
        return ins2stream.count(ins) > 0;
    }
    bool is_record(migraphx::instruction_ref ins) const
    {
        return ins->name() == "record_event";
    }
    bool is_wait(migraphx::instruction_ref ins) const
    {
        return ins->name() == "wait_event";
    }
};

struct program_model
{
    migraphx::program p;
    std::unordered_map<migraphx::instruction_ref, std::size_t> ins2stream{};
    std::size_t nstreams = 0;

    template<class... Ts>
    migraphx::instruction_ref add_literal(Ts... xs)
    {
        return p.add_literal(xs...);
    }

    template<class... Ts>
    migraphx::instruction_ref add_instruction(Ts... xs)
    {
        return p.add_instruction(xs...);
    }

    template<class... Ts>
    migraphx::instruction_ref add_instruction_stream(std::size_t n, Ts... xs)
    {
        nstreams = std::max(nstreams, n);
        auto ins = p.add_instruction(xs...);
        ins2stream[ins] = n;
        return ins;
    }

    test_stream_model get_stream_model() const
    {
        return {nstreams, ins2stream};
    }

    std::vector<migraphx::stream_race> analyze() const
    {
        return migraphx::analyze_streams(p, get_stream_model());
    }
};

TEST_CASE(simple_race)
{
    program_model pm;
    auto one = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    auto pass3 = pm.add_instruction_stream(0, pass_op{}, pass1, pass2);
    auto races = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass3});
    EXPECT(bool{races.front().before == pass2});
}

TEST_CASE(simple_race_record_wait_wrong_stream)
{
    program_model pm;
    auto one = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    pm.add_instruction_stream(0, record_event{1});
    pm.add_instruction_stream(1, wait_event{1});
    auto pass3 = pm.add_instruction_stream(0, pass_op{}, pass1, pass2);
    auto races = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass3});
    EXPECT(bool{races.front().before == pass2});
}

TEST_CASE(simple_race_record_wait_same_stream1)
{
    program_model pm;
    auto one = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    pm.add_instruction_stream(1, record_event{1});
    pm.add_instruction_stream(1, wait_event{1});
    auto pass3 = pm.add_instruction_stream(0, pass_op{}, pass1, pass2);
    auto races = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass3});
    EXPECT(bool{races.front().before == pass2});
}

TEST_CASE(simple_race_record_wait_same_stream2)
{
    program_model pm;
    auto one = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    pm.add_instruction_stream(0, record_event{1});
    pm.add_instruction_stream(0, wait_event{1});
    auto pass3 = pm.add_instruction_stream(0, pass_op{}, pass1, pass2);
    auto races = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass3});
    EXPECT(bool{races.front().before == pass2});
}

TEST_CASE(simple_race_sync)
{
    program_model pm;
    auto one = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    pm.add_instruction_stream(1, record_event{1});
    pm.add_instruction_stream(0, wait_event{1});
    pm.add_instruction_stream(0, pass_op{}, pass1, pass2);
    auto races = pm.analyze();

    EXPECT(races.empty());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }

