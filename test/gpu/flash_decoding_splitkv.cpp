/*
 * Compile flash-decoding attention programs through the GPU/MLIR pipeline and
 * report whether rocMLIR emits splitKV / rock.attention in the rock dialect dump.
 *
 * Run all scenarios (report only):
 *   ./build/bin/test_gpu_flash_decoding_splitkv 'flash_decoding_splitkv_report*'
 *
 * Run all scenarios and fail if flash-decoding expectations mismatch:
 *   ./build/bin/test_gpu_flash_decoding_splitkv 'flash_decoding_splitkv_check_expectations*'
 *
 * Or use tools/check_flash_decoding_splitkv.sh
 */
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/env.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/fuse_attention.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/tmp_dir.hpp>
#include <test.hpp>

#include <cstdlib>
#include <cctype>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct splitkv_report
{
    std::string name;
    bool compile_ok          = false;
    bool flash_decoding_ir   = false;
    bool mlir_op             = false;
    bool rock_attention      = false;
    bool splitkv             = false;
    std::string error;
};

static bool contains_file_text(const migraphx::fs::path& file, std::string_view needle)
{
    std::ifstream in(file);
    if(not in)
        return false;
    std::string line;
    while(std::getline(in, line))
    {
        if(line.find(needle) != std::string::npos)
            return true;
    }
    return false;
}

static bool is_hl_mlir_dump(const migraphx::fs::path& path)
{
    const auto filename = path.filename().string();
    constexpr std::string_view suffix = ".hl.mlir";
    return filename.size() >= suffix.size() and
           filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool contains_rock_attention_splitkv_gt_one(const migraphx::fs::path& file)
{
    std::ifstream in(file);
    if(not in)
        return false;
    std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if(contents.find("rock.attention") == std::string::npos)
        return false;

    constexpr std::string_view needle = "splitKV =";
    std::size_t pos                   = 0;
    while((pos = contents.find(needle, pos)) != std::string::npos)
    {
        pos += needle.size();
        while(pos < contents.size() and std::isspace(static_cast<unsigned char>(contents[pos])))
            ++pos;
        const std::size_t start = pos;
        while(pos < contents.size() and std::isdigit(static_cast<unsigned char>(contents[pos])))
            ++pos;
        if(pos > start and std::stoi(contents.substr(start, pos - start)) > 1)
            return true;
    }
    return false;
}

static void scan_dump_dir(const migraphx::fs::path& dir, splitkv_report& report)
{
    if(not migraphx::fs::exists(dir))
        return;
    for(const auto& entry : migraphx::fs::recursive_directory_iterator(dir))
    {
        if(not entry.is_regular_file())
            continue;
        const auto& path = entry.path();
        if(not is_hl_mlir_dump(path))
            continue;
        report.rock_attention |= contains_file_text(path, "rock.attention");
        report.splitkv |= contains_rock_attention_splitkv_gt_one(path);
    }
}

static const migraphx::fs::path& mlir_dump_root()
{
    static migraphx::fs::path root;
    static bool initialized = false;
    if(not initialized)
    {
        initialized = true;
        const char* root_env = std::getenv("MIGRAPHX_MLIR_DUMP_ROOT");
        if(root_env != nullptr and root_env[0] != '\0')
        {
            root = migraphx::fs::path{root_env};
            return root;
        }
        const char* dump_env = std::getenv("MIGRAPHX_MLIR_DUMP");
        if(dump_env != nullptr and dump_env[0] != '\0')
            root = migraphx::fs::path{dump_env};
    }
    return root;
}

static migraphx::fs::path scenario_dump_path(std::string_view scenario_name,
                                             const migraphx::fs::path& owned_dump_dir)
{
    const auto& root = mlir_dump_root();
    if(root.empty())
        return owned_dump_dir;
    return root / std::string(scenario_name);
}

static bool module_has_flash_decoding(const migraphx::module& m)
{
    for(auto ins : migraphx::iterator_for(m))
    {
        if(ins->name() != "group")
            continue;
        if(ins->module_inputs().empty())
            continue;
        const auto& mod_name = ins->module_inputs().front()->name();
        if(mod_name.find("flash_decoding") != std::string::npos)
            return true;
    }
    return false;
}

static bool module_has_mlir_compile(const migraphx::module& m)
{
    for(auto ins : migraphx::iterator_for(m))
    {
        if(ins->name() == "gpu::mlir_op")
            return true;
        if(ins->name() == "gpu::code_object")
        {
            const auto* sym = ins->get_operator().to_value()["symbol_name"].if_string();
            if(sym != nullptr and sym->find("mlir") != std::string::npos)
                return true;
        }
        if(ins->name() == "gpu::precompile_op")
        {
            const auto& op = ins->get_operator();
            if(op.name() == "gpu::precompile_op")
            {
                const auto* inner = op.to_value()["op"].if_string();
                if(inner != nullptr and inner->find("mlir_op") != std::string::npos)
                    return true;
            }
        }
    }
    return false;
}

static void configure_flash_decoding_env(const migraphx::fuse_attention& fa)
{
    if(not fa.flash_decoding_enabled and fa.flash_decoding_num_splits == 0)
    {
        unsetenv("MIGRAPHX_FLASH_DECODING_NUM_SPLITS");
        return;
    }
    if(fa.flash_decoding_num_splits > 0)
        setenv("MIGRAPHX_FLASH_DECODING_NUM_SPLITS",
               std::to_string(fa.flash_decoding_num_splits).c_str(),
               1);
    else
        unsetenv("MIGRAPHX_FLASH_DECODING_NUM_SPLITS");
}

static migraphx::program make_gemm_softmax_gemm_4d(const migraphx::shape& s1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto a   = mm->add_parameter("q", s1);
    auto k   = mm->add_parameter("k", s1);
    auto v   = mm->add_parameter("v", s1);
    k        = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                          k);
    v        = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                          v);
    auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), a, k);
    auto rmax  = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), gemm1);
    rmax       = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rmax);
    auto sub  = mm->add_instruction(migraphx::make_op("sub"), gemm1, rmax);
    auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);
    auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
    rsum      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
    auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
    auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, v);
    mm->add_return({gemm2});
    return p;
}

static migraphx::program make_gemm_softmax_gemm_4d_causal(const migraphx::shape& s_scores,
                                                            const migraphx::shape& s_mask)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto q   = mm->add_parameter("q", s_scores);
    auto k   = mm->add_parameter("k", s_scores);
    auto v   = mm->add_parameter("v", s_scores);
    k        = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                          k);
    v        = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                          v);
    auto scores = mm->add_instruction(migraphx::make_op("dot"), q, k);

    std::vector<uint8_t> mask_data(s_mask.elements());
    const std::size_t seq = s_mask.lens().back();
    for(std::size_t i = 0; i < seq; ++i)
    {
        for(std::size_t j = 0; j < seq; ++j)
            mask_data[i * seq + j] = j > i;
    }
    auto mask = mm->add_literal(migraphx::literal{s_mask, mask_data});
    mask      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s_scores.lens()}}), mask);
    auto ninf = mm->add_literal(migraphx::literal{migraphx::shape{s_scores.type(), {1}}, {-10000.0f}});
    ninf      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s_scores.lens()}}), ninf);
    auto masked = mm->add_instruction(migraphx::make_op("where"), mask, ninf, scores);

    auto rmax = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), masked);
    rmax      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s_scores.lens()}}), rmax);
    auto sub  = mm->add_instruction(migraphx::make_op("sub"), masked, rmax);
    auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);
    auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
    rsum      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s_scores.lens()}}), rsum);
    auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
    auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, v);
    mm->add_return({gemm2});
    return p;
}

static migraphx::program make_gemm_softmax_gemm_4d_rectangular()
{
    migraphx::shape s_qv{migraphx::shape::half_type, {1, 12, 384, 64}};
    migraphx::shape s_k{migraphx::shape::half_type, {1, 12, 64, 384}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto q   = mm->add_parameter("q", s_qv);
    auto k   = mm->add_parameter("k", s_k);
    auto v   = mm->add_parameter("v", s_qv);
    auto scores = mm->add_instruction(migraphx::make_op("dot"), q, k);
    auto rmax   = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), scores);
    rmax        = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 384, 384}}}), rmax);
    auto sub  = mm->add_instruction(migraphx::make_op("sub"), scores, rmax);
    auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);
    auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
    rsum      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 384, 384}}}), rsum);
    auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
    auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, v);
    mm->add_return({gemm2});
    return p;
}

static migraphx::program make_gemm_softmax_gemm_3d()
{
    migraphx::shape s_3d{migraphx::shape::half_type, {1, 256, 256}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto q   = mm->add_parameter("q", s_3d);
    auto k   = mm->add_parameter("k", s_3d);
    auto v   = mm->add_parameter("v", s_3d);
    k        = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), k);
    v        = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), v);
    auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), q, k);
    auto rmax  = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}), gemm1);
    rmax       = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}), rmax);
    auto sub  = mm->add_instruction(migraphx::make_op("sub"), gemm1, rmax);
    auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);
    auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), exp);
    rsum      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}), rsum);
    auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
    auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, v);
    mm->add_return({gemm2});
    return p;
}

static migraphx::program make_gemm_softmax_gemm_3d_rectangular()
{
    migraphx::shape s_qk{migraphx::shape::half_type, {1, 256, 240}};
    migraphx::shape s_v{migraphx::shape::half_type, {1, 240, 256}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto q   = mm->add_parameter("q", s_qk);
    auto k   = mm->add_parameter("k", s_qk);
    auto v   = mm->add_parameter("v", s_v);
    q        = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), q);
    auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), q, k);
    auto rmax  = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}), gemm1);
    rmax       = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}), rmax);
    auto sub  = mm->add_instruction(migraphx::make_op("sub"), gemm1, rmax);
    auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);
    auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), exp);
    rsum      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}), rsum);
    auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
    auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, v);
    mm->add_return({gemm2});
    return p;
}

static migraphx::instruction_ref softmax_dot_3d(migraphx::module* mm,
                                                 migraphx::instruction_ref scores,
                                                 migraphx::instruction_ref v,
                                                 const migraphx::shape& s3d)
{
    auto rmax = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}), scores);
    rmax      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s3d.lens()}}), rmax);
    auto sub  = mm->add_instruction(migraphx::make_op("sub"), scores, rmax);
    auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);
    auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), exp);
    rsum      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s3d.lens()}}), rsum);
    auto div = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
    return mm->add_instruction(migraphx::make_op("dot"), div, v);
}

static migraphx::instruction_ref softmax_dot_4d(migraphx::module* mm,
                                                migraphx::instruction_ref scores,
                                                migraphx::instruction_ref v,
                                                const std::vector<std::size_t>& score_lens)
{
    auto rmax = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), scores);
    rmax      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", score_lens}}), rmax);
    auto sub  = mm->add_instruction(migraphx::make_op("sub"), scores, rmax);
    auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);
    auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
    rsum      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", score_lens}}), rsum);
    auto div = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
    return mm->add_instruction(migraphx::make_op("dot"), div, v);
}

static migraphx::program make_verify_3d_rectangular()
{
    migraphx::shape q_shape{migraphx::shape::half_type, {1, 64, 16}};
    migraphx::shape k_shape{migraphx::shape::half_type, {1, 16, 256}};
    migraphx::shape v_shape{migraphx::shape::half_type, {1, 256, 32}};
    migraphx::shape score_shape{migraphx::shape::half_type, {1, 64, 256}};

    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto q       = mm->add_parameter("q", q_shape);
    auto k       = mm->add_parameter("k", k_shape);
    auto v       = mm->add_parameter("v", v_shape);
    auto scores  = mm->add_instruction(migraphx::make_op("dot"), q, k);
    auto rmax    = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}), scores);
    rmax         = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", score_shape.lens()}}), rmax);
    auto sub  = mm->add_instruction(migraphx::make_op("sub"), scores, rmax);
    auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);
    auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), exp);
    rsum      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", score_shape.lens()}}), rsum);
    auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
    auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, v);
    mm->add_return({gemm2});
    return p;
}

static migraphx::program make_verify_causal_rectangular()
{
    const std::size_t seq   = 384;
    const std::size_t heads = 12;
    const std::size_t dim   = 64;

    migraphx::shape s_qv{migraphx::shape::half_type, {1, heads, seq, dim}};
    migraphx::shape s_k{migraphx::shape::half_type, {1, heads, dim, seq}};
    migraphx::shape s_scores{migraphx::shape::half_type, {1, heads, seq, seq}};
    migraphx::shape s_mask{migraphx::shape::bool_type, {1, 1, seq, seq}};
    migraphx::shape s_scalar{migraphx::shape::half_type, {1}};

    std::vector<uint8_t> mask_data(s_mask.elements());
    for(std::size_t i = 0; i < seq; ++i)
        for(std::size_t j = 0; j < seq; ++j)
            mask_data[i * seq + j] = j > i;

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto q   = mm->add_parameter("q", s_qv);
    auto k   = mm->add_parameter("k", s_k);
    auto v   = mm->add_parameter("v", s_qv);
    auto kt  = mm->add_instruction(migraphx::make_op("dot"), q, k);

    auto scale = mm->add_literal(migraphx::literal{s_scalar, {0.125f}});
    scale      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s_scores.lens()}}), scale);
    auto scaled = mm->add_instruction(migraphx::make_op("mul"), kt, scale);

    auto mask = mm->add_literal(migraphx::literal{s_mask, mask_data});
    mask      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s_scores.lens()}}), mask);
    auto ninf = mm->add_literal(migraphx::literal{s_scalar, {-10000.0f}});
    ninf      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s_scores.lens()}}), ninf);
    auto masked = mm->add_instruction(migraphx::make_op("where"), mask, ninf, scaled);

    auto out = softmax_dot_4d(mm, masked, v, s_scores.lens());
    mm->add_return({out});
    return p;
}

static migraphx::program make_verify_rectangular_padding_mask()
{
    const std::size_t seq   = 384;
    const std::size_t heads = 12;
    const std::size_t dim   = 64;

    migraphx::shape s_qv{migraphx::shape::half_type, {1, heads, seq, dim}};
    migraphx::shape s_k{migraphx::shape::half_type, {1, heads, dim, seq}};
    migraphx::shape s_scores{migraphx::shape::half_type, {1, heads, seq, seq}};
    migraphx::shape s_mask{migraphx::shape::bool_type, {1, 1, seq, seq}};
    migraphx::shape s_scalar{migraphx::shape::half_type, {1}};

    std::vector<uint8_t> mask_data(s_mask.elements(), 0);

    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto q      = mm->add_parameter("q", s_qv);
    auto k      = mm->add_parameter("k", s_k);
    auto v      = mm->add_parameter("v", s_qv);
    auto scores = mm->add_instruction(migraphx::make_op("dot"), q, k);

    auto mask = mm->add_literal(migraphx::literal{s_mask, mask_data});
    mask      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s_scores.lens()}}), mask);
    auto ninf = mm->add_literal(migraphx::literal{s_scalar, {-10000.0f}});
    ninf      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s_scores.lens()}}), ninf);
    auto masked = mm->add_instruction(migraphx::make_op("where"), mask, ninf, scores);

    auto out = softmax_dot_4d(mm, masked, v, s_scores.lens());
    mm->add_return({out});
    return p;
}

static migraphx::program make_flash_decoding_3d_with_attention_literal()
{
    migraphx::shape s_3d{migraphx::shape::half_type, {1, 256, 256}};
    migraphx::shape s_scalar{migraphx::shape::half_type, {1}};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto q   = mm->add_parameter("q", s_3d);
    auto k   = mm->add_parameter("k", s_3d);
    auto v   = mm->add_parameter("v", s_3d);
    k        = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), k);
    v        = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), v);
    auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), q, k);

    auto scale = mm->add_literal(migraphx::literal{s_scalar, {0.125f}});
    scale      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s_3d.lens()}}), scale);
    auto scaled = mm->add_instruction(migraphx::make_op("mul"), gemm1, scale);

    auto out = softmax_dot_3d(mm, scaled, v, s_3d);
    mm->add_return({out});
    return p;
}

static migraphx::program make_flash_decoding_4d_with_broadcastable_mask_literal()
{
    migraphx::shape s1{migraphx::shape::half_type, {1, 12, 384, 384}};
    migraphx::shape s_mask{migraphx::shape::bool_type, {1, 1, 384, 384}};
    std::vector<uint8_t> mask_data(s_mask.elements(), 1);

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto q    = mm->add_parameter("q", s1);
    auto k    = mm->add_parameter("k", s1);
    auto v    = mm->add_parameter("v", s1);
    auto mask = mm->add_literal(migraphx::literal{s_mask, mask_data});
    auto ninf = mm->add_literal(-std::numeric_limits<float>::infinity());
    ninf      = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", s1.type()}}), ninf);
    k = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);
    v = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), v);
    auto gemm1   = mm->add_instruction(migraphx::make_op("dot"), q, k);
    auto mask_bc = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), mask);
    auto ninf_bc = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), ninf);
    auto masked = mm->add_instruction(migraphx::make_op("where"), mask_bc, gemm1, ninf_bc);
    auto out    = softmax_dot_4d(mm, masked, v, s1.lens());
    mm->add_return({out});
    return p;
}

static migraphx::program make_flash_decoding_4d_with_broadcastable_mask_param()
{
    migraphx::shape s1{migraphx::shape::half_type, {1, 12, 256, 256}};
    migraphx::shape s_mask{migraphx::shape::bool_type, {1, 1, 256, 256}};

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto q    = mm->add_parameter("q", s1);
    auto k    = mm->add_parameter("k", s1);
    auto v    = mm->add_parameter("v", s1);
    auto mask = mm->add_parameter("mask", s_mask);
    auto ninf = mm->add_literal(-std::numeric_limits<float>::infinity());
    ninf      = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", s1.type()}}), ninf);
    k = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);
    v = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), v);
    auto gemm1   = mm->add_instruction(migraphx::make_op("dot"), q, k);
    auto mask_bc = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), mask);
    auto ninf_bc = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), ninf);
    auto masked = mm->add_instruction(migraphx::make_op("where"), mask_bc, gemm1, ninf_bc);
    auto out    = softmax_dot_4d(mm, masked, v, s1.lens());
    mm->add_return({out});
    return p;
}

static migraphx::program make_flash_decoding_4d_with_unary_on_softmax_broadcast()
{
    migraphx::shape s1{migraphx::shape::half_type, {1, 12, 384, 384}};
    migraphx::shape s_qv{migraphx::shape::half_type, {1, 12, 384, 64}};
    migraphx::shape s_k{migraphx::shape::half_type, {1, 12, 64, 384}};

    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto q     = mm->add_parameter("q", s_qv);
    auto k     = mm->add_parameter("k", s_k);
    auto v     = mm->add_parameter("v", s_qv);
    auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), q, k);
    auto rmax  = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), gemm1);
    rmax       = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rmax);
    auto rmax_cont = mm->add_instruction(migraphx::make_op("contiguous"), rmax);
    auto sub       = mm->add_instruction(migraphx::make_op("sub"), gemm1, rmax_cont);
    auto exp       = mm->add_instruction(migraphx::make_op("exp"), sub);
    auto rsum      = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
    rsum           = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
    auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
    auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, v);
    mm->add_return({gemm2});
    return p;
}

static migraphx::program make_verify_3d_input_fusion()
{
    migraphx::shape s_3d{migraphx::shape::half_type, {1, 256, 256}};
    migraphx::shape s_scalar{migraphx::shape::half_type, {1}};

    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto q_input = mm->add_parameter("q", s_3d);
    auto k_input = mm->add_parameter("k", s_3d);
    auto v_input = mm->add_parameter("v", s_3d);
    auto q_bias  = mm->add_parameter("q_bias", s_3d);
    auto k_bias  = mm->add_parameter("k_bias", s_3d);
    auto v_bias  = mm->add_parameter("v_bias", s_3d);
    auto scale   = mm->add_parameter("scale", s_scalar);

    auto q_with_bias = mm->add_instruction(migraphx::make_op("add"), q_input, q_bias);
    auto k_with_bias = mm->add_instruction(migraphx::make_op("add"), k_input, k_bias);
    auto v_with_bias = mm->add_instruction(migraphx::make_op("add"), v_input, v_bias);
    scale            = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s_3d.lens()}}), scale);
    auto q_scaled    = mm->add_instruction(migraphx::make_op("mul"), q_with_bias, scale);
    auto q_activated = mm->add_instruction(migraphx::make_op("tanh"), q_scaled);
    auto k_activated = mm->add_instruction(migraphx::make_op("tanh"), k_with_bias);
    auto v_activated = mm->add_instruction(migraphx::make_op("tanh"), v_with_bias);
    auto k_transposed =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), k_activated);
    auto v_transposed =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), v_activated);
    auto scores = mm->add_instruction(migraphx::make_op("dot"), q_activated, k_transposed);
    auto out    = softmax_dot_3d(mm, scores, v_transposed, s_3d);
    mm->add_return({out});
    return p;
}

static migraphx::program make_verify_3d_output_fusion()
{
    migraphx::shape s_3d{migraphx::shape::half_type, {1, 256, 256}};
    migraphx::shape bias_shape{migraphx::shape::half_type, {256}};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto q   = mm->add_parameter("q", s_3d);
    auto k   = mm->add_parameter("k", s_3d);
    auto v   = mm->add_parameter("v", s_3d);
    auto k_transposed =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), k);
    auto v_transposed =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), v);
    auto scores           = mm->add_instruction(migraphx::make_op("dot"), q, k_transposed);
    auto attention_output = softmax_dot_3d(mm, scores, v_transposed, s_3d);

    auto output_proj_weight =
        mm->add_parameter("output_proj_weight", migraphx::shape{s_3d.type(), {256, 256}});
    auto output_bias = mm->add_parameter("output_bias", bias_shape);
    auto attn_reshaped =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {256, 256}}}), attention_output);
    auto projected =
        mm->add_instruction(migraphx::make_op("dot"), attn_reshaped, output_proj_weight);
    projected = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s_3d.lens()}}), projected);
    output_bias = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s_3d.lens()}}), output_bias);
    auto out = mm->add_instruction(migraphx::make_op("add"), projected, output_bias);
    mm->add_return({out});
    return p;
}

static migraphx::program make_gemm_softmax_gemm_3d_sized(migraphx::shape s_3d)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto q   = mm->add_parameter("q", s_3d);
    auto k   = mm->add_parameter("k", s_3d);
    auto v   = mm->add_parameter("v", s_3d);
    k        = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), k);
    v        = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), v);
    auto scores = mm->add_instruction(migraphx::make_op("dot"), q, k);
    auto out    = softmax_dot_3d(mm, scores, v, s_3d);
    mm->add_return({out});
    return p;
}

static migraphx::fuse_attention default_flash_decoding_fa()
{
    migraphx::fuse_attention fa{};
    fa.attn_enabled                  = true;
    fa.flash_decoding_enabled        = true;
    fa.flash_decoding_num_splits     = 2;
    fa.flash_decoding_threshold      = 32;
    fa.flash_decoding_max_splits     = 16;
    fa.flash_decoding_min_chunk_size = 32;
    return fa;
}

static migraphx::fuse_attention auto_split_fa(std::size_t threshold      = 32,
                                              std::size_t max_splits     = 16,
                                              std::size_t min_chunk_size = 32)
{
    migraphx::fuse_attention fa{};
    fa.attn_enabled                  = true;
    fa.flash_decoding_enabled        = true;
    fa.flash_decoding_num_splits     = 0;
    fa.flash_decoding_threshold      = threshold;
    fa.flash_decoding_max_splits     = max_splits;
    fa.flash_decoding_min_chunk_size = min_chunk_size;
    return fa;
}

static splitkv_report analyze_scenario(std::string_view name,
                                       const std::function<migraphx::program()>& build,
                                       migraphx::fuse_attention fa)
{
    splitkv_report report;
    report.name = std::string(name);

    migraphx::tmp_dir owned_dump_dir{"flash_decoding_splitkv_" + report.name};
    const migraphx::fs::path dump_path = scenario_dump_path(name, owned_dump_dir.path);
    if(migraphx::fs::exists(dump_path))
        migraphx::fs::remove_all(dump_path);
    migraphx::fs::create_directories(dump_path);

    try
    {
        configure_flash_decoding_env(fa);
        setenv("MIGRAPHX_MLIR_USE_SPECIFIC_OPS", "attention", 1);

        auto p = build();
        migraphx::run_passes(p, {fa, migraphx::dead_code_elimination{}});
        report.flash_decoding_ir = module_has_flash_decoding(*p.get_main_module());

        setenv("MIGRAPHX_MLIR_DUMP", dump_path.string().c_str(), 1);

        p.compile(migraphx::make_target("gpu"), migraphx::compile_options{});
        report.compile_ok = true;
        report.mlir_op    = module_has_mlir_compile(*p.get_main_module());
        scan_dump_dir(dump_path, report);
        if(report.mlir_op and not report.rock_attention)
            report.error = "dump: " + dump_path.string();
    }
    catch(const std::exception& ex)
    {
        report.error = ex.what();
    }

    return report;
}

static void print_report_header()
{
    std::cout << std::left << std::setw(48) << "scenario" << std::setw(8) << "fd_ir"
              << std::setw(8) << "mlir" << std::setw(10) << "rock.attn" << std::setw(10)
              << "splitKV" << "notes\n";
    std::cout << std::string(90, '-') << "\n";
}

static void print_report_row(const splitkv_report& report)
{
    std::cout << std::left << std::setw(48) << report.name << std::setw(8)
              << (report.flash_decoding_ir ? "yes" : "no") << std::setw(8)
              << (report.mlir_op ? "yes" : "no") << std::setw(10)
              << (report.rock_attention ? "yes" : "no") << std::setw(10)
              << (report.splitkv ? "yes" : "no") << report.error << "\n";
}

struct scenario
{
    std::string name;
    std::function<migraphx::program()> build;
    migraphx::fuse_attention fa;
    bool expect_flash_decoding = true;
};

static std::vector<scenario> all_scenarios()
{
    migraphx::shape s4{migraphx::shape::half_type, {1, 12, 256, 256}};
    migraphx::shape s_mask{migraphx::shape::bool_type, {1, 1, 256, 256}};
    auto fa = default_flash_decoding_fa();

    return {
        // test/fuse_attention.cpp
        {"gemm_softmax_gemm_flash_decoding",
         [=] { return make_gemm_softmax_gemm_4d(s4); },
         fa,
         true},
        {"flash_decoding_3d", [] { return make_gemm_softmax_gemm_3d(); }, fa, true},
        {"flash_decoding_3d_rectangular",
         [] { return make_gemm_softmax_gemm_3d_rectangular(); },
         fa,
         true},
        {"flash_decoding_4d_rectangular",
         [] { return make_gemm_softmax_gemm_4d_rectangular(); },
         fa,
         true},
        {"gemm_causal_mask_softmax_gemm_flash_decoding",
         [=] { return make_gemm_softmax_gemm_4d_causal(s4, s_mask); },
         fa,
         true},
        {"flash_decoding_3d_with_attention_literal",
         [] { return make_flash_decoding_3d_with_attention_literal(); },
         fa,
         true},
        {"flash_decoding_4d_with_broadcastable_mask_literal",
         [] { return make_flash_decoding_4d_with_broadcastable_mask_literal(); },
         fa,
         true},
        {"flash_decoding_4d_with_broadcastable_mask_param",
         [] { return make_flash_decoding_4d_with_broadcastable_mask_param(); },
         fa,
         true},
        {"flash_decoding_4d_with_unary_on_softmax_broadcast",
         [] { return make_flash_decoding_4d_with_unary_on_softmax_broadcast(); },
         fa,
         true},
        {"flash_decoding_3d_auto_split_large_sequence",
         [] {
             return make_gemm_softmax_gemm_3d_sized(
                 migraphx::shape{migraphx::shape::half_type, {1, 512, 512}});
         },
         auto_split_fa(),
         true},
        {"flash_decoding_3d_auto_split_small_sequence",
         [] {
             return make_gemm_softmax_gemm_3d_sized(
                 migraphx::shape{migraphx::shape::half_type, {1, 16, 16}});
         },
         auto_split_fa(),
         false},
        {"flash_decoding_4d_auto_split_custom_params",
         [=] { return make_gemm_softmax_gemm_4d(s4); },
         auto_split_fa(32, 4, 64),
         true},
        {"flash_decoding_auto_split_threshold_below",
         [] {
             return make_gemm_softmax_gemm_3d_sized(
                 migraphx::shape{migraphx::shape::half_type, {1, 127, 127}});
         },
         auto_split_fa(128, 8, 32),
         false},
        {"flash_decoding_auto_split_threshold_at",
         [] {
             return make_gemm_softmax_gemm_3d_sized(
                 migraphx::shape{migraphx::shape::half_type, {1, 128, 128}});
         },
         auto_split_fa(128, 8, 32),
         true},
        {"gemm_softmax_gemm_no_flash_decoding",
         [=] { return make_gemm_softmax_gemm_4d(s4); },
         [] {
             migraphx::fuse_attention no_fd{};
             no_fd.attn_enabled              = true;
             no_fd.flash_decoding_enabled    = false;
             no_fd.flash_decoding_num_splits = 0;
             return no_fd;
         }(),
         false},
        // test/verify/test_attention_flash_decoding_*.cpp
        {"verify_causal_rectangular",
         [] { return make_verify_causal_rectangular(); },
         fa,
         true},
        {"verify_rectangular_no_mask",
         [] { return make_gemm_softmax_gemm_4d_rectangular(); },
         fa,
         true},
        {"verify_rectangular_padding_mask",
         [] { return make_verify_rectangular_padding_mask(); },
         fa,
         true},
        {"verify_3d_rectangular", [] { return make_verify_3d_rectangular(); }, fa, true},
        {"verify_3d_input_fusion", [] { return make_verify_3d_input_fusion(); }, fa, true},
        {"verify_3d_output_fusion", [] { return make_verify_3d_output_fusion(); }, fa, true},
    };
}

static bool scenario_matches_expectation(const splitkv_report& report, bool expect_flash_decoding)
{
    const bool has_flash_decoding = report.flash_decoding_ir and report.splitkv;
    return has_flash_decoding == expect_flash_decoding;
}

static void run_report(const std::vector<scenario>& scenarios, bool check_expectations)
{
    print_report_header();
    std::size_t splitkv_hits     = 0;
    std::size_t expectation_miss = 0;
    std::vector<std::string> missed;
    std::vector<std::pair<std::string, splitkv_report>> results;
    results.reserve(scenarios.size());

    for(const auto& s : scenarios)
    {
        auto report = analyze_scenario(s.name, s.build, s.fa);
        results.emplace_back(s.name, report);
        print_report_row(report);
        if(report.splitkv)
            ++splitkv_hits;
        if(not scenario_matches_expectation(report, s.expect_flash_decoding))
        {
            ++expectation_miss;
            const char* detail =
                s.expect_flash_decoding ? " (expected flash decoding)" : " (expected no flash decoding)";
            missed.push_back(s.name + detail);
        }
    }

    std::cout << std::string(90, '-') << "\n";
    std::cout << "splitKV > 1 in rocMLIR: " << splitkv_hits << " / " << scenarios.size() << "\n";

    std::cout << "\nFlash decoding (fd_ir + splitKV > 1):\n";
    for(std::size_t i = 0; i < scenarios.size(); ++i)
    {
        const auto& report = results[i].second;
        if(report.flash_decoding_ir and report.splitkv)
            std::cout << "  + " << results[i].first << "\n";
    }
    std::cout << "No flash decoding (no fd_ir or splitKV = 1):\n";
    for(std::size_t i = 0; i < scenarios.size(); ++i)
    {
        const auto& report = results[i].second;
        if(not report.flash_decoding_ir or not report.splitkv)
            std::cout << "  - " << results[i].first;
        if(report.flash_decoding_ir and not report.splitkv)
            std::cout << "  [fd_ir without splitKV>1]";
        if(not report.flash_decoding_ir and report.splitkv)
            std::cout << "  [splitKV>1 without fd_ir]";
        if(not report.flash_decoding_ir or not report.splitkv)
            std::cout << "\n";
    }

    if(not missed.empty())
    {
        std::cout << "expectation mismatches: " << expectation_miss << "\n";
        for(const auto& name : missed)
            std::cout << "  - " << name << "\n";
    }
    std::cout << "\nColumns:\n"
              << "  fd_ir     = fuse_attention created *_flash_decoding submodule\n"
              << "  mlir      = gpu::mlir_op or gpu::code_object(mlir_*) after compile\n"
              << "  rock.attn = rock.attention in *.hl.mlir dump (post high-level pipeline)\n"
              << "  splitKV   = splitKV > 1 in *.hl.mlir dump (flash decoding in rocMLIR)\n"
              << "\nPass: fd_ir=yes and splitKV=yes for flash-decoding scenarios;\n"
              << "      fd_ir=no and splitKV=no for non-flash-decoding scenarios.\n";

    if(check_expectations)
        EXPECT(expectation_miss == 0);
}

static void run_single_scenario(std::string_view scenario_name)
{
    for(const auto& s : all_scenarios())
    {
        if(s.name != scenario_name)
            continue;
        print_report_header();
        print_report_row(analyze_scenario(s.name, s.build, s.fa));
        return;
    }
    EXPECT(false);
}

#define FLASH_DECODING_SPLITKV_SCENARIO(scenario_name) \
    TEST_CASE(flash_decoding_splitkv_##scenario_name)  \
    {                                                  \
        run_single_scenario(#scenario_name);           \
    }

} // namespace

TEST_CASE(flash_decoding_splitkv_report)
{
    run_report(all_scenarios(), false);
}

TEST_CASE(flash_decoding_splitkv_check_expectations)
{
    run_report(all_scenarios(), true);
}

FLASH_DECODING_SPLITKV_SCENARIO(gemm_softmax_gemm_flash_decoding)
FLASH_DECODING_SPLITKV_SCENARIO(flash_decoding_3d)
FLASH_DECODING_SPLITKV_SCENARIO(flash_decoding_3d_rectangular)
FLASH_DECODING_SPLITKV_SCENARIO(flash_decoding_4d_rectangular)
FLASH_DECODING_SPLITKV_SCENARIO(gemm_causal_mask_softmax_gemm_flash_decoding)
FLASH_DECODING_SPLITKV_SCENARIO(gemm_softmax_gemm_no_flash_decoding)

int main(int argc, const char* argv[]) { test::run(argc, argv); }
