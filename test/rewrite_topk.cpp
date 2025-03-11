#include <migraphx/rewrite_topk.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::module& m) { migraphx::run_passes(m, {migraphx::rewrite_topk{}}); }

TEST_CASE(small_topk)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {300}});
        auto r = m1.add_instruction(migraphx::make_op("topk", {{"k", 8}}), x);
        m1.add_return({r});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(large_topk_no_split)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {240000}});
        auto r = m1.add_instruction(migraphx::make_op("topk", {{"k", 120000}}), x);
        m1.add_return({r});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(split_topk_batch_1)
{
    const auto n = 240000;
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {n}});
        auto r = m1.add_instruction(migraphx::make_op("topk", {{"k", 8}}), x);
        m1.add_return({r});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        const auto group = 32;
        std::vector<std::uint32_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {n}});
        auto input_idx =
            m2.add_literal(migraphx::literal{{migraphx::shape::uint32_type, {n}}, indices});
        auto input_idxb = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {n}}}), input_idx);
        auto input_idxr = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", {group, n / group}}}), input_idxb);
        auto xr =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {group, n / group}}}), x);
        auto r1 =
            m2.add_instruction(migraphx::make_op("topk", {{"k", 8}, {"axis", 1}}), xr, input_idxr);
        auto value1 = m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), r1);
        auto idx1   = m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), r1);
        auto valuer =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {8 * group}}}), value1);
        auto idxr = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {8 * group}}}), idx1);
        auto r2 =
            m2.add_instruction(migraphx::make_op("topk", {{"k", 8}, {"axis", 0}}), valuer, idxr);
        m2.add_return({r2});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(split_topk_batch_64)
{
    const auto n     = 240000;
    const auto batch = 64;
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {batch, n}});
        auto r = m1.add_instruction(migraphx::make_op("topk", {{"k", 8}, {"axis", 1}}), x);
        m1.add_return({r});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        const auto group = 32;
        std::vector<std::uint32_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {batch, n}});
        auto input_idx =
            m2.add_literal(migraphx::literal{{migraphx::shape::uint32_type, {n}}, indices});
        auto input_idxb = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {batch, n}}}), input_idx);
        auto input_idxr = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", {batch, group, n / group}}}), input_idxb);
        auto xr = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", {batch, group, n / group}}}), x);
        auto r1 =
            m2.add_instruction(migraphx::make_op("topk", {{"k", 8}, {"axis", 2}}), xr, input_idxr);
        auto value1 = m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), r1);
        auto idx1   = m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), r1);
        auto valuer = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", {batch, 8 * group}}}), value1);
        auto idxr =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {batch, 8 * group}}}), idx1);
        auto r2 =
            m2.add_instruction(migraphx::make_op("topk", {{"k", 8}, {"axis", 1}}), valuer, idxr);
        m2.add_return({r2});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(split_topk_batch_64_last)
{
    const auto n     = 240000;
    const auto batch = 64;
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {n, batch}});
        auto r = m1.add_instruction(migraphx::make_op("topk", {{"k", 8}, {"axis", 0}}), x);
        m1.add_return({r});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        const auto group = 32;
        std::vector<std::uint32_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {n, batch}});
        auto input_idx =
            m2.add_literal(migraphx::literal{{migraphx::shape::uint32_type, {n}}, indices});
        auto input_idxb = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {n, batch}}}), input_idx);
        auto input_idxr = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", {group, n / group, batch}}}), input_idxb);
        auto xr = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", {group, n / group, batch}}}), x);
        auto r1 =
            m2.add_instruction(migraphx::make_op("topk", {{"k", 8}, {"axis", 1}}), xr, input_idxr);
        auto value1 = m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), r1);
        auto idx1   = m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), r1);
        auto valuer = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", {8 * group, batch}}}), value1);
        auto idxr =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {8 * group, batch}}}), idx1);
        auto r2 =
            m2.add_instruction(migraphx::make_op("topk", {{"k", 8}, {"axis", 0}}), valuer, idxr);
        m2.add_return({r2});
    }
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
