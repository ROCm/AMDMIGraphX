#include <migraphx/pre_scheduling.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct set_stream
{
    int stream = -1;
    std::string name() const { return "gpu::set_stream"; }

    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        if(inputs.empty())
            return {};
        else
            return inputs.front();
    }
};

struct create_events
{
    int num_of_events = 0;
    std::string name() const { return "gpu::create_events"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        if(inputs.empty())
            return {};
        else
            return inputs.front();
    }
};

struct weight_func
{
    weight_func()
    {
        weight_map["@param"]   = std::make_pair(1, 1);
        weight_map["@literal"] = std::make_pair(1, 1);
    };
    std::pair<int, int> operator()(const migraphx::operation& op)
    {
        if(weight_map.find(op.name()) != weight_map.end())
            return weight_map[op.name()];
        else
            return std::make_pair(1, 0);
    }
    std::unordered_map<std::string, std::pair<int, int>> weight_map;
};

struct insert_instruction
{
    void insert_stream(migraphx::program* p, migraphx::instruction_ref ins, int stream)
    {

        p->insert_instruction(ins, set_stream{stream});
    }

    void insert_create_events(migraphx::program*, migraphx::instruction_ref, int) {}
    void insert_record_event(migraphx::program*, migraphx::instruction_ref, int) {}

    void insert_wait_event(migraphx::program*, migraphx::instruction_ref, int) {}
};

struct stream_execution_target
{
    struct context
    {
        void finish() const {}
        void set_stream(int) {}
        void create_events(int) {}
        void record_event(int) {}
        void wait_event(int) {}
    };
    migraphx::context ctx = context{};
    std::string name() const { return "stream_execution"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&) const
    {
        return {migraphx::pre_scheduling{weight_func(), 2, insert_instruction{}}};
    }
    migraphx::context get_context() const { return {ctx}; }
};

TEST_CASE(test1)
{
    migraphx::program p;
    auto in1 =
        p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {32, 256, 35, 35}});
    auto l1 =
        p.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {64, 256, 1, 1}}));
    auto p1 = p.add_instruction(migraphx::op::convolution{}, in1, l1);
    auto l2 =
        p.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {48, 256, 1, 1}}));
    auto p2 = p.add_instruction(migraphx::op::convolution{}, in1, l2);
    p.add_instruction(migraphx::op::concat{1}, p1, p2);
    p.compile(stream_execution_target{});
    CHECK(std::count_if(
              p.begin(), p.end(), [](auto&& ins) { return ins.name() == "gpu::set_stream"; }) == 3);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
