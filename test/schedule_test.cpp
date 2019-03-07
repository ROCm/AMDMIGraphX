#include <migraphx/schedule.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/dfor.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct unary_op
{
    std::string name() const { return "unary"; }
    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        if(args.empty())
            return {};
        return args.front();
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.empty())
            return {};
        return inputs.front();
    }
    int output_alias(const std::vector<migraphx::shape>&) const { return 0; }
};

struct nary_op
{
    std::string name() const { return "nary"; }
    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        if(args.empty())
            return {};
        return args.front();
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.empty())
            return {};
        return inputs.front();
    }
};

struct wait_event
{
    std::shared_ptr<std::vector<std::size_t>> wait_for =
        std::make_shared<std::vector<std::size_t>>();
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(*self.wait_for, "wait_for"));
    }
    std::string name() const { return "wait_event"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const { return {}; }

    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape&,
                               const std::vector<migraphx::argument>&) const
    {
        assert(wait_for != nullptr);
        assert(not wait_for->empty());
        return {};
    }
};

using instruction_map = std::unordered_map<migraphx::instruction_ref, std::size_t>;
using wait_map =
    std::unordered_map<migraphx::instruction_ref, std::shared_ptr<std::vector<std::size_t>>>;

struct schedule_model_test
{
    std::shared_ptr<instruction_map> ins2stream = std::make_shared<instruction_map>();
    std::shared_ptr<std::unordered_map<std::size_t, std::size_t>> wait2stream =
        std::make_shared<std::unordered_map<std::size_t, std::size_t>>();
    std::shared_ptr<wait_map> ins2wait_for = std::make_shared<wait_map>();
    std::size_t concurrency() const { return 4; }
    void sched(migraphx::program&, migraphx::instruction_ref ins, std::size_t n) const
    {
        (*ins2stream)[ins] = n;
    }
    void wait(migraphx::program& p, migraphx::instruction_ref ins, std::size_t wait_id) const
    {
        if(ins2wait_for->count(ins) == 0)
        {
            auto event = wait_event{};
            p.insert_instruction(ins, event);
            (*ins2wait_for)[ins] = event.wait_for;
        }
        (*ins2wait_for)[ins]->push_back(wait2stream->at(wait_id));
    }
    void record(migraphx::program& p, migraphx::instruction_ref ins, std::size_t wait_id) const
    {
        (*wait2stream)[wait_id] = ins2stream->at(ins);
    }
    std::size_t weight(const migraphx::operation& op) const
    {
        if(op.name() == "binary" or op.name() == "unary")
            return 4;
        else
            return 1;
    }
};

struct schedule_target
{
    schedule_model_test model{};
    std::string name() const { return "schedule"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&) const
    {
        return {migraphx::schedule{model}};
    }
    migraphx::context get_context() const { return {}; }

    std::size_t get_stream(migraphx::instruction_ref ins) { return model.ins2stream->at(ins); }

    bool has_stream(migraphx::instruction_ref ins) { return model.ins2stream->count(ins) > 0; }
};

bool check_conflicts(migraphx::program& p, migraphx::instruction_ref x, migraphx::instruction_ref y)
{
    for(auto ins : migraphx::iterator_for(p))
    {
        if(ins->name() != "identity")
            continue;
        if(not migraphx::contains(ins->inputs(), x))
            continue;
        if(not migraphx::contains(ins->inputs(), y))
            continue;
        return true;
    }
    return false;
}

void check_conflicts(migraphx::program& p,
                     std::vector<std::vector<migraphx::instruction_ref>> conflicts,
                     bool result = true)
{
    migraphx::dfor(conflicts.size(), conflicts.size())([&](auto i, auto j) {
        if(i == j)
            return;
        for(auto ins1 : conflicts[i])
            for(auto ins2 : conflicts[j])
                CHECK(check_conflicts(p, ins1, ins2) == result);
    });
}

template <class T>
std::vector<T> sorted(std::vector<T> x)
{
    std::sort(x.begin(), x.end());
    return x;
}

template <class T>
std::vector<T> unique(std::vector<T> x)
{
    std::sort(x.begin(), x.end());
    x.erase(std::unique(x.begin(), x.end()), x.end());
    return x;
}

std::vector<std::size_t> get_wait_for(std::vector<std::size_t> wait_for)
{
    std::sort(wait_for.begin(), wait_for.end());
    return wait_for;
}

std::vector<std::size_t> get_wait_for(std::size_t wait_on, std::vector<std::size_t> wait_for)
{
    wait_for.erase(std::find(wait_for.begin(), wait_for.end(), wait_on));
    std::sort(wait_for.begin(), wait_for.end());
    return wait_for;
}

std::vector<std::size_t> get_wait_for(migraphx::instruction_ref ins)
{
    auto wait_ins = std::prev(ins);
    if(wait_ins->name() != "wait_event")
        return {};
    auto wf = *migraphx::any_cast<wait_event>(wait_ins->get_operator()).wait_for;
    std::sort(wf.begin(), wf.end());
    return wf;
}

template <class T>
std::vector<migraphx::instruction_ref>
chain(migraphx::program& p, std::size_t n, T x, migraphx::instruction_ref input)
{
    std::vector<migraphx::instruction_ref> result;
    for(std::size_t i = 0; i < n; i++)
    {
        result.push_back(p.add_instruction(x, input));
        input = result.back();
    }
    return result;
}
TEST_CASE(single_entry)
{
    schedule_target t{};
    migraphx::program p;
    auto one    = p.add_literal(1);
    auto onep1  = p.add_instruction(unary_op{}, one);
    auto onep2  = p.add_instruction(unary_op{}, one);
    auto binary = p.add_instruction(nary_op{}, onep1, onep2);
    p.compile(t);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(onep1) != t.get_stream(onep2));
    EXPECT(t.get_stream(binary) == 0);
    EXPECT(get_wait_for(binary) ==
           get_wait_for(t.get_stream(binary), {t.get_stream(onep1), t.get_stream(onep2)}));
    EXPECT(check_conflicts(p, onep1, onep2));
}

TEST_CASE(zero_record)
{
    schedule_target t{};
    migraphx::program p;
    auto one    = p.add_literal(1);
    auto onep1  = p.add_instruction(unary_op{}, one);
    auto onep2  = p.add_instruction(unary_op{}, one);
    auto binary = p.add_instruction(nary_op{},
                                    p.add_instruction(migraphx::op::identity{}, onep1),
                                    p.add_instruction(migraphx::op::identity{}, onep2));
    p.compile(t);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(onep1) != t.get_stream(onep2));
    EXPECT(t.has_stream(binary));
    EXPECT(get_wait_for(binary) ==
           get_wait_for(t.get_stream(binary), {t.get_stream(onep1), t.get_stream(onep2)}));
    EXPECT(check_conflicts(p, onep1, onep2));
}

TEST_CASE(zero_merge1)
{
    schedule_target t{};
    migraphx::program p;
    auto one    = p.add_literal(1);
    auto onep1  = p.add_instruction(unary_op{}, one);
    auto onep2  = p.add_instruction(unary_op{}, one);
    auto binary = p.add_instruction(migraphx::op::identity{}, onep1, onep2);
    p.compile(t);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(onep1) != t.get_stream(onep2));
    // No stream assignment
    EXPECT(not t.has_stream(binary));
    // There is no wait
    EXPECT(get_wait_for(binary).empty());
    EXPECT(check_conflicts(p, onep1, onep2));
}

TEST_CASE(zero_merge2)
{
    schedule_target t{};
    migraphx::program p;
    auto one    = p.add_literal(1);
    auto onep1  = p.add_instruction(unary_op{}, one);
    auto onep2  = p.add_instruction(unary_op{}, one);
    auto binary = p.add_instruction(migraphx::op::identity{},
                                    p.add_instruction(migraphx::op::identity{}, onep1),
                                    p.add_instruction(migraphx::op::identity{}, onep2));
    p.compile(t);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(onep1) != t.get_stream(onep2));
    // No stream assignment
    EXPECT(not t.has_stream(binary));
    // There is no wait
    EXPECT(get_wait_for(binary).empty());
    EXPECT(check_conflicts(p, onep1, onep2));
}

TEST_CASE(double_entry)
{
    schedule_target t{};
    migraphx::program p;
    auto one    = p.add_literal(1);
    auto two    = p.add_literal(2);
    auto onep   = p.add_instruction(unary_op{}, one);
    auto twop   = p.add_instruction(unary_op{}, two);
    auto binary = p.add_instruction(nary_op{}, onep, twop);
    p.compile(t);
    EXPECT(not t.has_stream(one));
    EXPECT(not t.has_stream(two));
    EXPECT(t.get_stream(onep) != t.get_stream(twop));
    EXPECT(t.get_stream(binary) == 0);
    EXPECT(get_wait_for(binary) ==
           get_wait_for(t.get_stream(binary), {t.get_stream(onep), t.get_stream(twop)}));
    // EXPECT(check_conflicts(p, onep, twop));
}

TEST_CASE(two_branches)
{
    schedule_target t{};
    migraphx::program p;
    auto one    = p.add_literal(1);
    auto c1     = chain(p, 2, unary_op{}, one);
    auto i1     = p.add_instruction(unary_op{}, one);
    auto binary = p.add_instruction(nary_op{}, i1, c1.back());
    p.compile(t);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(i1) == 1);
    for(auto ins : c1)
        EXPECT(t.get_stream(ins) == 0);
    EXPECT(t.get_stream(binary) == 0);
    EXPECT(get_wait_for(binary) ==
           get_wait_for(t.get_stream(binary), {t.get_stream(c1.back()), t.get_stream(i1)}));
    check_conflicts(p, {c1, {i1}});
}

TEST_CASE(four_branches)
{
    schedule_target t{};
    migraphx::program p;
    auto one    = p.add_literal(1);
    auto c1     = chain(p, 4, unary_op{}, one);
    auto c2     = chain(p, 3, unary_op{}, one);
    auto c3     = chain(p, 2, unary_op{}, one);
    auto i1     = p.add_instruction(unary_op{}, one);
    auto binary = p.add_instruction(nary_op{}, i1, c1.back(), c2.back(), c3.back());
    p.compile(t);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(i1) == 3);
    for(auto ins : c1)
        EXPECT(t.get_stream(ins) == 0);
    for(auto ins : c2)
        EXPECT(t.get_stream(ins) == 1);
    for(auto ins : c3)
        EXPECT(t.get_stream(ins) == 2);
    EXPECT(t.get_stream(binary) == 0);
    EXPECT(get_wait_for(binary) == get_wait_for(t.get_stream(binary),
                                                {t.get_stream(c1.back()),
                                                 t.get_stream(c2.back()),
                                                 t.get_stream(c3.back()),
                                                 t.get_stream(i1)}));
    check_conflicts(p, {c1, c2, c3, {i1}});
}

TEST_CASE(five_branches)
{
    schedule_target t{};
    migraphx::program p;
    auto one    = p.add_literal(1);
    auto c1     = chain(p, 5, unary_op{}, one);
    auto c2     = chain(p, 4, unary_op{}, one);
    auto c3     = chain(p, 3, unary_op{}, one);
    auto c4     = chain(p, 2, unary_op{}, one);
    auto i1     = p.add_instruction(unary_op{}, one);
    auto binary = p.add_instruction(nary_op{}, i1, c1.back(), c2.back(), c3.back(), c4.back());
    p.compile(t);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(i1) == 3);
    for(auto ins : c1)
        EXPECT(t.get_stream(ins) == 0);
    for(auto ins : c2)
        EXPECT(t.get_stream(ins) == 1);
    for(auto ins : c3)
        EXPECT(t.get_stream(ins) == 2);
    for(auto ins : c4)
        EXPECT(t.get_stream(ins) == 3);
    EXPECT(t.get_stream(binary) == 0);
    EXPECT(get_wait_for(binary) == get_wait_for(t.get_stream(binary),
                                                {t.get_stream(c1.back()),
                                                 t.get_stream(c2.back()),
                                                 t.get_stream(c3.back()),
                                                 t.get_stream(i1)}));
    check_conflicts(p, {c1, c2, c3, c4});
    check_conflicts(p, {c1, c2, c3, {i1}});
}

TEST_CASE(four_branches_eq)
{
    schedule_target t{};
    migraphx::program p;
    auto one    = p.add_literal(1);
    auto onep1  = p.add_instruction(unary_op{}, one);
    auto onep2  = p.add_instruction(unary_op{}, one);
    auto onep3  = p.add_instruction(unary_op{}, one);
    auto onep4  = p.add_instruction(unary_op{}, one);
    auto binary = p.add_instruction(nary_op{}, onep1, onep2, onep3, onep4);
    p.compile(t);
    EXPECT(not t.has_stream(one));
    EXPECT(
        sorted<std::size_t>(
            {t.get_stream(onep1), t.get_stream(onep2), t.get_stream(onep3), t.get_stream(onep4)}) ==
        unique<std::size_t>(
            {t.get_stream(onep1), t.get_stream(onep2), t.get_stream(onep3), t.get_stream(onep4)}));
    EXPECT(t.get_stream(binary) == 0);
    EXPECT(
        get_wait_for(binary) ==
        get_wait_for(
            t.get_stream(binary),
            {t.get_stream(onep1), t.get_stream(onep2), t.get_stream(onep3), t.get_stream(onep4)}));
    check_conflicts(p, {{onep1}, {onep2}, {onep3}, {onep4}});
}

TEST_CASE(seq_merge)
{
    schedule_target t{};
    migraphx::program p;
    auto one     = p.add_literal(1);
    auto c1      = chain(p, 2, unary_op{}, one);
    auto i1      = p.add_instruction(unary_op{}, one);
    auto binary1 = p.add_instruction(nary_op{}, i1, c1.back());

    auto c2      = chain(p, 2, unary_op{}, binary1);
    auto i2      = p.add_instruction(unary_op{}, binary1);
    auto binary2 = p.add_instruction(nary_op{}, i2, c2.back());

    p.compile(t);
    EXPECT(not t.has_stream(one));

    EXPECT(t.get_stream(i1) == 2);
    for(auto ins : c1)
        EXPECT(t.get_stream(ins) == 3);
    EXPECT(t.get_stream(binary1) == 3);
    EXPECT(get_wait_for(binary1) ==
           get_wait_for(t.get_stream(binary1), {t.get_stream(c1.back()), t.get_stream(i1)}));
    check_conflicts(p, {c1, {i1}});

    EXPECT(t.get_stream(i2) == 3);
    for(auto ins : c2)
        EXPECT(t.get_stream(ins) == 0);
    EXPECT(t.get_stream(binary2) == 0);
    EXPECT(get_wait_for(binary2) ==
           get_wait_for(t.get_stream(binary2), {t.get_stream(c2.back()), t.get_stream(i2)}));
    check_conflicts(p, {c2, {i2}});
}

TEST_CASE(par_merge)
{
    schedule_target t{};
    migraphx::program p;
    auto one     = p.add_literal(1);
    auto start1  = p.add_instruction(unary_op{}, one);
    auto c1      = chain(p, 3, unary_op{}, start1);
    auto i1      = p.add_instruction(unary_op{}, start1);
    auto binary1 = p.add_instruction(nary_op{}, i1, c1.back());

    auto start2  = p.add_instruction(unary_op{}, one);
    auto c2      = chain(p, 2, unary_op{}, start2);
    auto i2      = p.add_instruction(unary_op{}, start2);
    auto binary2 = p.add_instruction(nary_op{}, i2, c2.back());

    auto binary3 = p.add_instruction(nary_op{}, binary1, binary2);

    p.compile(t);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(binary3) == 0);

    EXPECT(t.get_stream(i1) == 2);
    for(auto ins : c1)
        EXPECT(t.get_stream(ins) == 0);
    EXPECT(t.get_stream(binary1) == 0);
    EXPECT(get_wait_for(binary1) ==
           get_wait_for(t.get_stream(binary1), {t.get_stream(c1.back()), t.get_stream(i1)}));
    check_conflicts(p, {c1, {i1}});

    EXPECT(t.get_stream(i2) == 1);
    for(auto ins : c2)
        EXPECT(t.get_stream(ins) == 3);
    EXPECT(t.get_stream(binary2) == 3);
    EXPECT(get_wait_for(binary2) ==
           get_wait_for(t.get_stream(binary2), {t.get_stream(c2.back()), t.get_stream(i2)}));
    check_conflicts(p, {c2, {i2}});

    EXPECT(check_conflicts(p, binary1, binary2));
}
int main(int argc, const char* argv[]) { test::run(argc, argv); }
