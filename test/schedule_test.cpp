#include <migraphx/schedule.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/op/identity.hpp>
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
    std::string comment;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.comment, "comment"));
    }
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

struct stream_free_op
{
    std::string comment;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.comment, "comment"));
    }
    std::string name() const { return "stream_free"; }
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
using int_map         = std::unordered_map<std::size_t, std::size_t>;
using wait_map =
    std::unordered_map<migraphx::instruction_ref, std::shared_ptr<std::vector<std::size_t>>>;

struct schedule_model_test
{
    std::shared_ptr<instruction_map> ins2stream = std::make_shared<instruction_map>();
    std::shared_ptr<int_map> wait2stream        = std::make_shared<int_map>();
    std::shared_ptr<wait_map> ins2wait_for      = std::make_shared<wait_map>();
    std::size_t concurrency() const { return 4; }
    void sched(migraphx::module&, migraphx::instruction_ref ins, std::size_t n) const
    {
        (*ins2stream)[ins] = n;
    }
    void wait(migraphx::module& p, migraphx::instruction_ref ins, std::size_t wait_id) const
    {
        if(ins2wait_for->count(ins) == 0)
        {
            auto event = wait_event{};
            p.insert_instruction(ins, event);
            (*ins2wait_for)[ins] = event.wait_for;
        }
        (*ins2wait_for)[ins]->push_back(wait2stream->at(wait_id));
    }
    void record(migraphx::module&, migraphx::instruction_ref ins, std::size_t wait_id) const
    {
        (*wait2stream)[wait_id] = ins2stream->at(ins);
    }
    std::size_t weight(const migraphx::operation& op) const
    {
        if(op.name() == "stream_free")
            return 0;
        else if(op.name() == "binary" or op.name() == "unary")
            return 4;
        else
            return 1;
    }
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

struct scheduler
{
    schedule_model_test model{};

    std::size_t get_stream(migraphx::instruction_ref ins) { return model.ins2stream->at(ins); }

    std::vector<std::size_t> get_streams(std::vector<migraphx::instruction_ref> inss)
    {
        std::vector<std::size_t> result;
        std::transform(inss.begin(), inss.end(), std::back_inserter(result), [&](auto ins) {
            return this->get_stream(ins);
        });
        return result;
    }

    void run_pass(migraphx::program& p) { 
        auto* mm = p.get_main_module();
        migraphx::run_passes(*mm, {migraphx::schedule{model}}); 
    }

    bool has_stream(migraphx::instruction_ref ins) { return model.ins2stream->count(ins) > 0; }

    void check_conflicts(migraphx::program& p,
                         std::vector<std::vector<migraphx::instruction_ref>> conflicts,
                         bool result = true)
    {
        migraphx::dfor(conflicts.size(), conflicts.size())([&](auto i, auto j) {
            if(i == j)
                return;
            for(auto ins1 : conflicts[i])
            {
                for(auto ins2 : conflicts[j])
                {
                    // If both instructions are on the same stream then dont check for a conflict
                    if(this->has_stream(ins1) and this->has_stream(ins2) and
                       this->get_stream(ins1) == this->get_stream(ins2))
                        continue;
                    CHECK(::check_conflicts(p, ins1, ins2) == result);
                }
            }
        });
    }
};

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
    return unique(std::move(wait_for));
}

std::vector<std::size_t> get_wait_for(std::size_t wait_on, std::vector<std::size_t> wait_for)
{
    wait_for.erase(std::find(wait_for.begin(), wait_for.end(), wait_on));
    return unique(wait_for);
}

std::vector<std::size_t> get_wait_for(migraphx::instruction_ref ins)
{
    auto wait_ins = std::prev(ins);
    // Skip identity operators
    while(wait_ins->name() == "identity")
        wait_ins = std::prev(wait_ins);
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
        result.push_back(p.get_main_module()->add_instruction(x, input));
        input = result.back();
    }
    return result;
}
TEST_CASE(single_entry)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one    = mm->add_literal(1);
    auto onep1  = mm->add_instruction(unary_op{}, one);
    auto onep2  = mm->add_instruction(unary_op{}, one);
    auto binary = mm->add_instruction(nary_op{}, onep1, onep2);
    t.run_pass(p);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(onep1) != t.get_stream(onep2));
    EXPECT(t.get_stream(binary) == 0);
    EXPECT(get_wait_for(binary) ==
           get_wait_for(t.get_stream(binary), {t.get_stream(onep1), t.get_stream(onep2)}));
    EXPECT(check_conflicts(p, onep1, onep2));
}

TEST_CASE(stream_free)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one    = mm->add_literal(1);
    auto onep1  = mm->add_instruction(stream_free_op{}, one);
    auto onep2  = mm->add_instruction(stream_free_op{}, one);
    auto binary = mm->add_instruction(nary_op{}, onep1, onep2);
    t.run_pass(p);
    EXPECT(not t.has_stream(one));
    EXPECT(not t.has_stream(onep1));
    EXPECT(not t.has_stream(onep2));
    EXPECT(not t.has_stream(binary));
}

TEST_CASE(zero_record)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one    = mm->add_literal(1);
    auto onep1  = mm->add_instruction(unary_op{}, one);
    auto onep2  = mm->add_instruction(unary_op{}, one);
    auto onei1  = mm->add_instruction(migraphx::op::identity{}, onep1);
    auto onei2  = mm->add_instruction(migraphx::op::identity{}, onep2);
    auto binary = mm->add_instruction(nary_op{}, onei1, onei2);
    t.run_pass(p);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(onep1) != t.get_stream(onep2));
    EXPECT(t.has_stream(binary));
    EXPECT(get_wait_for(binary) ==
           get_wait_for(t.get_stream(binary), {t.get_stream(onep1), t.get_stream(onep2)}));
    EXPECT(check_conflicts(p, onep1, onep2));
    t.check_conflicts(p, {{onep1, onei1}, {onep2, onei2}});
}

TEST_CASE(zero_merge1)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one    = mm->add_literal(1);
    auto onep1  = mm->add_instruction(unary_op{}, one);
    auto onep2  = mm->add_instruction(unary_op{}, one);
    auto binary = mm->add_instruction(migraphx::op::identity{}, onep1, onep2);
    t.run_pass(p);
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
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one    = mm->add_literal(1);
    auto onep1  = mm->add_instruction(unary_op{}, one);
    auto onep2  = mm->add_instruction(unary_op{}, one);
    auto binary = mm->add_instruction(migraphx::op::identity{},
                                    mm->add_instruction(migraphx::op::identity{}, onep1),
                                    mm->add_instruction(migraphx::op::identity{}, onep2));
    t.run_pass(p);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(onep1) != t.get_stream(onep2));
    // No stream assignment
    EXPECT(not t.has_stream(binary));
    // There is no wait
    EXPECT(get_wait_for(binary).empty());
    EXPECT(check_conflicts(p, onep1, onep2));
}

TEST_CASE(zero_merge3)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto onep1 = mm->add_instruction(unary_op{}, one);
    auto onep2 = mm->add_instruction(unary_op{}, one);
    auto id    = mm->add_instruction(migraphx::op::identity{}, onep1, onep2);
    auto final = mm->add_instruction(unary_op{}, id);
    t.run_pass(p);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(onep1) != t.get_stream(onep2));
    // No stream assignment
    EXPECT(not t.has_stream(id));
    // There is no wait
    EXPECT(get_wait_for(id).empty());
    // Stream assignment for final op
    EXPECT(t.get_stream(final) == 0);
    EXPECT(get_wait_for(final) ==
           get_wait_for(t.get_stream(final), {t.get_stream(onep1), t.get_stream(onep2)}));
    EXPECT(check_conflicts(p, onep1, onep2));
}

TEST_CASE(zero_merge4)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto onep1 = mm->add_instruction(unary_op{}, one);
    auto onep2 = mm->add_instruction(unary_op{}, one);
    auto id    = mm->add_instruction(migraphx::op::identity{},
                                mm->add_instruction(migraphx::op::identity{}, onep1),
                                mm->add_instruction(migraphx::op::identity{}, onep2));
    auto final = mm->add_instruction(unary_op{}, id);
    t.run_pass(p);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(onep1) != t.get_stream(onep2));
    // No stream assignment
    EXPECT(not t.has_stream(id));
    // There is no wait
    EXPECT(get_wait_for(id).empty());
    // Stream assignment for final op
    EXPECT(t.get_stream(final) == 0);
    EXPECT(get_wait_for(final) ==
           get_wait_for(t.get_stream(final), {t.get_stream(onep1), t.get_stream(onep2)}));
    EXPECT(check_conflicts(p, onep1, onep2));
}

TEST_CASE(double_entry)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one    = mm->add_instruction(stream_free_op{}, mm->add_literal(1));
    auto two    = mm->add_instruction(stream_free_op{}, mm->add_literal(2));
    auto onep   = mm->add_instruction(unary_op{}, one);
    auto twop   = mm->add_instruction(unary_op{}, two);
    auto binary = mm->add_instruction(nary_op{}, onep, twop);
    t.run_pass(p);
    EXPECT(not t.has_stream(one));
    EXPECT(not t.has_stream(two));
    EXPECT(t.get_stream(onep) != t.get_stream(twop));
    EXPECT(t.get_stream(binary) == 0);
    EXPECT(get_wait_for(binary) ==
           get_wait_for(t.get_stream(binary), {t.get_stream(onep), t.get_stream(twop)}));
    t.check_conflicts(p, {{onep, one}, {twop, two}});
}

TEST_CASE(two_branches)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one    = mm->add_literal(1);
    auto c1     = chain(p, 2, unary_op{}, one);
    auto i1     = mm->add_instruction(unary_op{}, one);
    auto binary = mm->add_instruction(nary_op{}, i1, c1.back());
    t.run_pass(p);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(i1) == 1);
    for(auto ins : c1)
        EXPECT(t.get_stream(ins) == 0);
    EXPECT(t.get_stream(binary) == 0);
    EXPECT(get_wait_for(binary) ==
           get_wait_for(t.get_stream(binary), {t.get_stream(c1.back()), t.get_stream(i1)}));
    t.check_conflicts(p, {c1, {i1}});
}

TEST_CASE(four_branches)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one    = mm->add_literal(1);
    auto c1     = chain(p, 4, unary_op{}, one);
    auto c2     = chain(p, 3, unary_op{}, one);
    auto c3     = chain(p, 2, unary_op{}, one);
    auto i1     = mm->add_instruction(unary_op{}, one);
    auto binary = mm->add_instruction(nary_op{}, i1, c1.back(), c2.back(), c3.back());
    t.run_pass(p);
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
    t.check_conflicts(p, {c1, c2, c3, {i1}});
}

TEST_CASE(five_branches)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one    = mm->add_literal(1);
    auto c1     = chain(p, 5, unary_op{}, one);
    auto c2     = chain(p, 4, unary_op{}, one);
    auto c3     = chain(p, 3, unary_op{}, one);
    auto c4     = chain(p, 2, unary_op{}, one);
    auto i1     = mm->add_instruction(unary_op{}, one);
    auto binary = mm->add_instruction(nary_op{}, i1, c1.back(), c2.back(), c3.back(), c4.back());
    t.run_pass(p);
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
    t.check_conflicts(p, {c1, c2, c3, c4});
    t.check_conflicts(p, {c1, c2, c3, {i1}});
}

TEST_CASE(four_branches_eq)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one    = mm->add_literal(1);
    auto onep1  = mm->add_instruction(unary_op{}, one);
    auto onep2  = mm->add_instruction(unary_op{}, one);
    auto onep3  = mm->add_instruction(unary_op{}, one);
    auto onep4  = mm->add_instruction(unary_op{}, one);
    auto binary = mm->add_instruction(nary_op{}, onep1, onep2, onep3, onep4);
    t.run_pass(p);
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
    t.check_conflicts(p, {{onep1}, {onep2}, {onep3}, {onep4}});
}

TEST_CASE(seq_merge)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one     = mm->add_literal(1);
    auto c1      = chain(p, 2, unary_op{}, one);
    auto i1      = mm->add_instruction(unary_op{}, one);
    auto binary1 = mm->add_instruction(nary_op{}, i1, c1.back());

    auto c2      = chain(p, 2, unary_op{}, binary1);
    auto i2      = mm->add_instruction(unary_op{}, binary1);
    auto binary2 = mm->add_instruction(nary_op{}, i2, c2.back());

    t.run_pass(p);
    EXPECT(not t.has_stream(one));

    EXPECT(t.get_stream(i1) != t.get_stream(c1.back()));
    for(auto ins : c1)
        EXPECT(t.get_stream(ins) == t.get_stream(c1.back()));
    EXPECT(t.get_stream(binary1) == t.get_stream(c1.back()));
    EXPECT(get_wait_for(binary1) ==
           get_wait_for(t.get_stream(binary1), {t.get_stream(c1.back()), t.get_stream(i1)}));
    t.check_conflicts(p, {c1, {i1}});

    EXPECT(t.get_stream(i2) != t.get_stream(c2.back()));
    for(auto ins : c2)
        EXPECT(t.get_stream(ins) == t.get_stream(c2.back()));
    EXPECT(t.get_stream(binary2) == 0);
    EXPECT(get_wait_for(binary2) ==
           get_wait_for(t.get_stream(binary2), {t.get_stream(c2.back()), t.get_stream(i2)}));
    t.check_conflicts(p, {c2, {i2}});
}

TEST_CASE(par_merge)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one     = mm->add_literal(1);
    auto start1  = mm->add_instruction(unary_op{}, one);
    auto c1      = chain(p, 3, unary_op{}, start1);
    auto i1      = mm->add_instruction(unary_op{}, start1);
    auto binary1 = mm->add_instruction(nary_op{}, i1, c1.back());

    auto start2  = mm->add_instruction(unary_op{}, one);
    auto c2      = chain(p, 2, unary_op{}, start2);
    auto i2      = mm->add_instruction(unary_op{}, start2);
    auto binary2 = mm->add_instruction(nary_op{}, i2, c2.back());

    auto binary3 = mm->add_instruction(nary_op{}, binary1, binary2);

    t.run_pass(p);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(binary3) == 0);

    EXPECT(t.get_stream(i1) != t.get_stream(i2));
    for(auto ins : c1)
        EXPECT(t.get_stream(ins) == 0);
    EXPECT(t.get_stream(binary1) == 0);
    EXPECT(get_wait_for(binary1) ==
           get_wait_for(t.get_stream(binary1), {t.get_stream(c1.back()), t.get_stream(i1)}));
    t.check_conflicts(p, {c1, {i1}});

    for(auto ins : c2)
        EXPECT(t.get_stream(ins) == t.get_stream(binary2));
    EXPECT(t.get_stream(binary2) != t.get_stream(i1));
    EXPECT(t.get_stream(binary2) != t.get_stream(i2));
    EXPECT(get_wait_for(binary2) ==
           get_wait_for(t.get_stream(binary2), {t.get_stream(c2.back()), t.get_stream(i2)}));
    t.check_conflicts(p, {c2, {i2}});

    EXPECT(check_conflicts(p, binary1, binary2));
    t.check_conflicts(p, {c1, {i1}, c2, {i2}});
}

TEST_CASE(inner_par_merge)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one     = mm->add_literal(1);
    auto start1  = mm->add_instruction(unary_op{}, one);
    auto c1      = chain(p, 3, unary_op{}, start1);
    auto i1      = mm->add_instruction(unary_op{}, start1);
    auto binary1 = mm->add_instruction(nary_op{}, i1, c1.back());

    auto start2  = mm->add_instruction(unary_op{}, one);
    auto c2      = chain(p, 2, unary_op{}, start2);
    auto i2      = mm->add_instruction(unary_op{}, start2);
    auto binary2 = mm->add_instruction(nary_op{}, i2, c2.back());

    auto outer1 = mm->add_instruction(unary_op{}, one);
    auto outer2 = mm->add_instruction(unary_op{}, one);

    auto output = mm->add_instruction(nary_op{}, binary1, binary2, outer1, outer2);

    t.run_pass(p);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(output) == 0);
    EXPECT(get_wait_for(output) == get_wait_for(t.get_stream(output),
                                                {t.get_stream(binary1),
                                                 t.get_stream(binary2),
                                                 t.get_stream(outer1),
                                                 t.get_stream(outer2)}));

    EXPECT(t.get_stream(outer1) != t.get_stream(outer2));
    EXPECT(migraphx::contains({1, 2}, t.get_stream(outer1)));
    EXPECT(migraphx::contains({1, 2}, t.get_stream(outer2)));

    EXPECT(t.get_stream(i1) != t.get_stream(i2));
    for(auto ins : c1)
        EXPECT(t.get_stream(ins) == 0);
    EXPECT(t.get_stream(binary1) == 0);
    EXPECT(get_wait_for(binary1) ==
           get_wait_for(t.get_stream(binary1), {t.get_stream(c1.back()), t.get_stream(i1)}));
    t.check_conflicts(p, {c1, {i1}});

    for(auto ins : c2)
        EXPECT(t.get_stream(ins) == t.get_stream(binary2));
    EXPECT(t.get_stream(binary2) != t.get_stream(i1));
    EXPECT(t.get_stream(binary2) != t.get_stream(i2));
    EXPECT(get_wait_for(binary2) ==
           get_wait_for(t.get_stream(binary2), {t.get_stream(c2.back()), t.get_stream(i2)}));
    t.check_conflicts(p, {c2, {i2}});

    EXPECT(check_conflicts(p, binary1, binary2));
    t.check_conflicts(p, {c1, {i1}, c2, {i2}, {outer1}, {outer2}});
}

TEST_CASE(par_merge_multi_entry)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one     = mm->add_literal(1);
    auto start1  = mm->add_instruction(unary_op{}, one);
    auto c1      = chain(p, 3, unary_op{}, start1);
    auto i1      = mm->add_instruction(unary_op{}, start1);
    auto binary1 = mm->add_instruction(nary_op{}, i1, c1.back());

    auto two     = mm->add_literal(1);
    auto start2  = mm->add_instruction(unary_op{}, two);
    auto c2      = chain(p, 2, unary_op{}, start2);
    auto i2      = mm->add_instruction(unary_op{}, start2);
    auto binary2 = mm->add_instruction(nary_op{}, i2, c2.back());

    auto binary3 = mm->add_instruction(nary_op{}, binary1, binary2);

    t.run_pass(p);
    EXPECT(not t.has_stream(one));
    EXPECT(not t.has_stream(two));
    EXPECT(t.get_stream(binary3) == 0);

    EXPECT(t.get_stream(i1) != t.get_stream(i2));
    for(auto ins : c1)
        EXPECT(t.get_stream(ins) == 0);
    EXPECT(t.get_stream(binary1) == 0);
    EXPECT(get_wait_for(binary1) ==
           get_wait_for(t.get_stream(binary1), {t.get_stream(c1.back()), t.get_stream(i1)}));
    t.check_conflicts(p, {c1, {i1}});

    for(auto ins : c2)
        EXPECT(t.get_stream(ins) == t.get_stream(binary2));
    EXPECT(t.get_stream(binary2) != t.get_stream(i1));
    EXPECT(t.get_stream(binary2) != t.get_stream(i2));
    EXPECT(get_wait_for(binary2) ==
           get_wait_for(t.get_stream(binary2), {t.get_stream(c2.back()), t.get_stream(i2)}));
    t.check_conflicts(p, {c2, {i2}});

    EXPECT(check_conflicts(p, binary1, binary2));
    t.check_conflicts(p, {c1, {i1}, c2, {i2}});
}

TEST_CASE(inner_split1)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one    = mm->add_literal(1);
    auto c1     = chain(p, 2, unary_op{}, one);
    auto i1     = mm->add_instruction(unary_op{}, one);
    auto s1     = mm->add_instruction(unary_op{}, c1);
    auto s2     = mm->add_instruction(unary_op{}, c1);
    auto output = mm->add_instruction(nary_op{}, i1, s1, s2);
    t.run_pass(p);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(i1) != t.get_stream(s1));
    EXPECT(t.get_stream(i1) != t.get_stream(s2));
    for(auto ins : c1)
        EXPECT(t.get_stream(ins) != t.get_stream(i1));
    EXPECT(t.get_stream(s1) != t.get_stream(s2));

    EXPECT(t.get_stream(output) == 0);
    EXPECT(
        get_wait_for(output) ==
        get_wait_for(t.get_stream(output), {t.get_stream(i1), t.get_stream(s1), t.get_stream(s2)}));
    // Either s1 or s2 has a wait depending on the sort order but not both
    EXPECT(get_wait_for(s1).empty() xor get_wait_for(s2).empty());
    t.check_conflicts(p, {c1, {i1}, {s1}, {s2}});
}

TEST_CASE(inner_split2)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one    = mm->add_literal(1);
    auto c1     = chain(p, 2, unary_op{}, one);
    auto i1     = mm->add_instruction(unary_op{}, one);
    auto s1     = chain(p, 3, unary_op{}, c1.back());
    auto s2     = chain(p, 4, unary_op{}, c1.back());
    auto output = mm->add_instruction(nary_op{}, i1, s1.back(), s2.back());
    t.run_pass(p);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(i1) != t.get_stream(s1.back()));
    EXPECT(t.get_stream(i1) != t.get_stream(s2.back()));
    for(auto ins : c1)
        EXPECT(t.get_stream(ins) != t.get_stream(i1));
    EXPECT(t.get_stream(s1.back()) != t.get_stream(s2.back()));

    EXPECT(t.get_stream(output) == 0);
    EXPECT(get_wait_for(output) ==
           get_wait_for(t.get_stream(output),
                        {t.get_stream(i1), t.get_stream(s1.back()), t.get_stream(s2.back())}));
    EXPECT(get_wait_for(s1.front()) == get_wait_for({t.get_stream(c1.back())}));
    t.check_conflicts(p, {c1, {i1}, s1, s2});
}

TEST_CASE(inception_resnet)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto one    = mm->add_literal(1);
    auto input  = mm->add_instruction(unary_op{}, one);
    auto c1     = chain(p, 2, unary_op{}, input);
    auto i1     = mm->add_instruction(unary_op{}, input);
    auto binary = mm->add_instruction(nary_op{}, i1, c1.back());
    auto output = mm->add_instruction(nary_op{}, binary, input);
    t.run_pass(p);
    EXPECT(not t.has_stream(one));
    EXPECT(t.get_stream(i1) != 0);
    for(auto ins : c1)
        EXPECT(t.get_stream(ins) == 0);
    EXPECT(t.get_stream(binary) == 0);
    EXPECT(get_wait_for(binary) ==
           get_wait_for(t.get_stream(binary), {t.get_stream(c1.back()), t.get_stream(i1)}));
    EXPECT(t.get_stream(output) == 0);
    EXPECT(get_wait_for(output).empty());
    t.check_conflicts(p, {c1, {i1}});
}

TEST_CASE(inception1)
{
    scheduler t{};
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto i1     = mm->add_literal(0);
    auto i2     = mm->add_literal(1);
    auto i3     = mm->add_literal(1);
    auto i4     = mm->add_literal(2);
    auto i7     = mm->add_instruction(nary_op{"i7"}, i1, i4, i3, i2);
    auto i8     = mm->add_literal(2);
    auto i9     = mm->add_instruction(migraphx::op::identity{}, i8);
    auto i10    = mm->add_literal(1);
    auto i11    = mm->add_instruction(nary_op{"i11"}, i7, i9, i10);
    auto i12    = mm->add_literal(2);
    auto i13    = mm->add_instruction(migraphx::op::identity{}, i12);
    auto i14    = mm->add_literal(1);
    auto i15    = mm->add_literal(1);
    auto i16    = mm->add_literal(2);
    auto i17    = mm->add_instruction(nary_op{"i17"}, i11, i16, i15, i13, i14);
    auto i18    = mm->add_literal(2);
    auto i19    = mm->add_instruction(migraphx::op::identity{}, i18);
    auto i20    = mm->add_literal(1);
    auto i21    = mm->add_literal(1);
    auto i22    = mm->add_literal(2);
    auto i23    = mm->add_instruction(nary_op{"i23"}, i17, i22, i21, i19, i20);
    auto i24    = mm->add_literal(1);
    auto i25    = mm->add_instruction(nary_op{"i25"}, i23, i24);
    auto i26    = mm->add_literal(2);
    auto i27    = mm->add_instruction(migraphx::op::identity{}, i26);
    auto i28    = mm->add_literal(1);
    auto i29    = mm->add_literal(1);
    auto i30    = mm->add_literal(2);
    auto i31    = mm->add_instruction(nary_op{"i31"}, i25, i30, i29, i27, i28);
    auto i32    = mm->add_literal(2);
    auto i33    = mm->add_instruction(migraphx::op::identity{}, i32);
    auto i34    = mm->add_literal(1);
    auto i35    = mm->add_literal(1);
    auto i36    = mm->add_literal(2);
    auto i37    = mm->add_instruction(nary_op{"i37"}, i31, i36, i35, i33, i34);
    auto i38    = mm->add_literal(1);
    auto i39    = mm->add_instruction(nary_op{"i39"}, i37, i38);
    auto i41    = mm->add_literal(2);
    auto i42    = mm->add_instruction(migraphx::op::identity{}, i41);
    auto i43    = mm->add_literal(1);
    auto i44    = mm->add_literal(1);
    auto i45    = mm->add_literal(2);
    auto i48    = mm->add_instruction(nary_op{"i48"}, i39, i45, i44, i42, i43);
    auto i49    = mm->add_literal(2);
    auto i50    = mm->add_instruction(migraphx::op::identity{}, i49);
    auto i51    = mm->add_literal(1);
    auto i52    = mm->add_literal(1);
    auto i53    = mm->add_literal(2);
    auto i54    = mm->add_instruction(nary_op{"i54"}, i48, i53, i52, i50, i51);
    auto i55    = mm->add_literal(1);
    auto i56    = mm->add_instruction(migraphx::op::identity{}, i55);
    auto i57    = mm->add_literal(2);
    auto i58    = mm->add_instruction(migraphx::op::identity{}, i57);
    auto i59    = mm->add_literal(1);
    auto i60    = mm->add_literal(2);
    auto i61    = mm->add_instruction(nary_op{"i61"}, i54, i60, i59, i58, i56);
    auto i62    = mm->add_literal(2);
    auto i63    = mm->add_instruction(migraphx::op::identity{}, i62);
    auto i64    = mm->add_literal(1);
    auto i65    = mm->add_literal(1);
    auto i66    = mm->add_literal(2);
    auto i69    = mm->add_instruction(nary_op{"i69"}, i39, i66, i65, i63, i64);
    auto i70    = mm->add_instruction(migraphx::op::identity{}, i55);
    auto i71    = mm->add_literal(2);
    auto i72    = mm->add_instruction(migraphx::op::identity{}, i71);
    auto i73    = mm->add_literal(1);
    auto i74    = mm->add_literal(2);
    auto i75    = mm->add_instruction(nary_op{"i75"}, i69, i74, i73, i72, i70);
    auto i77    = mm->add_literal(1);
    auto i80    = mm->add_instruction(nary_op{"i80"}, i39, i77);
    auto i81    = mm->add_instruction(migraphx::op::identity{}, i55);
    auto i82    = mm->add_literal(2);
    auto i83    = mm->add_instruction(migraphx::op::identity{}, i82);
    auto i84    = mm->add_literal(1);
    auto i85    = mm->add_literal(2);
    auto i86    = mm->add_instruction(nary_op{"i86"}, i80, i85, i84, i83, i81);
    auto i88    = mm->add_instruction(migraphx::op::identity{}, i55);
    auto i89    = mm->add_literal(2);
    auto i90    = mm->add_instruction(migraphx::op::identity{}, i89);
    auto i91    = mm->add_literal(1);
    auto i92    = mm->add_literal(2);
    auto i94    = mm->add_instruction(nary_op{"i94"}, i39, i92, i91, i90, i88);
    auto i96    = mm->add_instruction(migraphx::op::identity{}, i55, i94, i75, i61, i86);
    auto i97    = mm->add_literal(2);
    auto i98    = mm->add_instruction(migraphx::op::identity{}, i97);
    auto i99    = mm->add_literal(3);
    auto i100   = mm->add_literal(1);
    auto i101   = mm->add_literal(2);
    auto output = mm->add_instruction(nary_op{"output"}, i96, i101, i100, i98, i99);

    t.run_pass(p);

    EXPECT(t.get_streams({i7, i11, i17, i23, i25, i31, i37, i39}) ==
           t.get_streams({i7, i7, i7, i7, i7, i7, i7, i7}));
    EXPECT(t.get_streams({i48, i54, i61, output}) ==
           t.get_streams({output, output, output, output}));
    EXPECT(t.get_streams({i80, i86}) == t.get_streams({i80, i80}));
    EXPECT(t.get_streams({i69, i75}) == t.get_streams({i69, i69}));

    EXPECT(t.get_stream(i7) != t.get_stream(i80));
    EXPECT(t.get_stream(i69) != t.get_stream(i80));
    EXPECT(t.get_stream(i69) != t.get_stream(i7));
    EXPECT(t.get_stream(output) != t.get_stream(i69));
    EXPECT(t.get_stream(output) != t.get_stream(i80));

    EXPECT(get_wait_for(i80) == get_wait_for({t.get_stream(i39)}));
    EXPECT(get_wait_for(i69) == get_wait_for({t.get_stream(i39)}));
    EXPECT(get_wait_for(i94) == get_wait_for({t.get_stream(i39)}));
    EXPECT(
        get_wait_for(output) ==
        get_wait_for(t.get_stream(output),
                     {t.get_stream(i94), t.get_stream(i75), t.get_stream(i61), t.get_stream(i86)}));

    t.check_conflicts(p, {{i80, i86}, {i69, i75}, {i48, i54, i61}, {i94}});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
