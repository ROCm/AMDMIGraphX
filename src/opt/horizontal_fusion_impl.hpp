#ifndef MIGRAPHX_GUARD_RTGLIB_HORIZONTAL_FUSION_IMPL_HPP
#define MIGRAPHX_GUARD_RTGLIB_HORIZONTAL_FUSION_IMPL_HPP
#include <migraphx/program.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/pass_config.hpp>
#include <migraphx/config.hpp>
#include "value_numbering.hpp"

namespace migraphx {

inline namespace MIGRAPHX_INLINE_NS {

struct horizontal_fusion_impl
{
    horizontal_fusion_impl(program* p) : p_program(p) {}
    void run();
    bool collect_inputs(std::vector<std::vector<instruction_ref>>&,
                        int&,
                        std::vector<instruction_ref>&,
                        std::unordered_map<instruction_ref, bool>&,
                        std::unordered_map<instruction_ref, instruction_ref>&,
                        std::unordered_map<instruction_ref, int>&);
    void transform(value_numbering&);
    void transform_layers(const std::vector<std::vector<instruction_ref>>&,
                          const std::unordered_map<instruction_ref, instruction_ref>&,
                          int,
                          const std::vector<instruction_ref>&);
    void transform_output(
        unsigned,
        const std::unordered_map<instruction_ref, int>&,
        const std::unordered_map<instruction_ref, std::vector<std::vector<std::size_t>>>&,
        const std::unordered_map<instruction_ref, std::vector<int>>&,
        value_numbering&);

    bool compare_inputs(const std::vector<instruction_ref>&,
                        const std::vector<instruction_ref>&,
                        instruction_ref,
                        int);
    std::vector<instruction_ref> walk(instruction_ref, std::unordered_map<instruction_ref, bool>&);
    void concat(const std::vector<instruction_ref>&,
                const std::unordered_map<instruction_ref, instruction_ref>&,
                int);
    int find_axis(instruction_ref, const std::unordered_map<instruction_ref, bool>&);
    int find_axis(instruction_ref, int dim);
    int find_axis(instruction_ref, instruction_ref, int);
    int find_unique_axis(instruction_ref, instruction_ref);
    std::vector<unsigned> find_cluster(unsigned, value_numbering&);
    bool match_dim(instruction_ref, instruction_ref, int axis);
    bool is_conv(instruction_ref);
    bool is_concat(instruction_ref);
    void remove_redundant_roots(const std::vector<instruction_ref>&);
    bool has_unique_output(const std::vector<instruction_ref>&);
    instruction_ref break_split(int, instruction_ref);
    static int get_channel_axis() { return 1; }
    static int get_conv_output_axis() { return 0; }

    private:
    program* p_program;
};
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif
