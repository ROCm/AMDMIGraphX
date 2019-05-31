#include "horizontal_fusion_impl.hpp"
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

bool horizontal_fusion_impl::is_conv(instruction_ref ins) { return (ins->name() == "convolution"); }

bool horizontal_fusion_impl::is_concat(instruction_ref ins) { return (ins->name() == "concat"); }

// Find the first axis that matches given dim.
int horizontal_fusion_impl::find_axis(instruction_ref ins, int dim)
{
    auto it = std::find(ins->get_shape().lens().begin(), ins->get_shape().lens().end(), dim);
    return (it != ins->get_shape().lens().end())
               ? static_cast<int>(std::distance(ins->get_shape().lens().begin(), it))
               : -1;
}

// Find axis for convolution filter's output and broadcast's input.
int horizontal_fusion_impl::find_axis(instruction_ref ins, instruction_ref base, int base_axis)
{
    if(is_conv(base))
    {
        return get_conv_output_axis();
    }
    else if(ins->outputs().at(0)->name() == "broadcast")
    {
        if(!match_dim(base, ins->outputs().at(0), ins->outputs().at(0)->get_shape().lens().size()))
            MIGRAPHX_THROW("Unmatched output");
        int dim = base->get_shape().lens().at(base_axis);
        return find_axis(ins, dim);
    }
    else
        return base_axis;
}

// Check whether ins1 and ins2 match in all dimensions excluding axis.
bool horizontal_fusion_impl::match_dim(instruction_ref ins1, instruction_ref ins2, int axis)
{
    auto lens1 = ins1->get_shape().lens();
    auto lens2 = ins2->get_shape().lens();
    if(lens1.size() != lens2.size())
        return false;
    int ndx = 0;
    for(auto&& size : lens1)
    {
        if((size != lens2.at(ndx)) && (ndx != axis))
            return false;
        ndx++;
    }
    return true;
}

bool horizontal_fusion_impl::compare_inputs(const std::vector<instruction_ref>& input1,
                                            const std::vector<instruction_ref>& input2,
                                            instruction_ref base_ins,
                                            int base_axis)
{
    if(input1.size() != input2.size())
        return false;
    std::size_t base_dim = base_ins->get_shape().lens().at(base_axis);

    return (std::equal(input1.begin(),
                       input1.end(),
                       input2.begin(),
                       [&](const instruction_ref& ins1, const instruction_ref& ins2) -> bool {
                           if(ins1->name() != ins2->name())
                               return false;
                           int axis = find_axis(ins2, base_ins, base_axis);
                           if(axis == -1)
                               return false;
                           if(!match_dim(ins1, ins2, axis))
                               return false;
                           if(ins2->get_shape().lens().at(axis) != base_dim)
                               return false;
                           return true;
                       }));
}

void horizontal_fusion_impl::concat(
    const std::vector<instruction_ref>& instrs,
    const std::unordered_map<instruction_ref, instruction_ref>& root,
    int root_axis)
{
    instruction_ref ins0               = instrs.at(0);
    instruction_ref base               = root.find(ins0)->second;
    std::vector<std::size_t> base_lens = base->get_shape().lens();
    int axis                           = find_axis(ins0, base, root_axis);

    int sum      = 0;
    int base_sum = 0;
    for(auto&& ins : instrs)
    {
        sum += ins->get_shape().lens().at(axis);
        base_sum += root.find(ins)->second->get_shape().lens().at(root_axis);
    }

    base_lens[root_axis] = base_sum;

    if(ins0->outputs().size() != 1)
        MIGRAPHX_THROW("unexpected output size");
    instruction_ref output            = ins0->outputs().at(0);
    shape s                           = ins0->get_shape();
    std::vector<std::size_t> new_lens = s.lens();
    new_lens[axis]                    = sum;

    if(ins0->name() == "@literal")
    {

        auto concat_ins = p_program->insert_instruction(
            output, op::concat{static_cast<std::size_t>(axis)}, instrs);
        instruction::replace_argument(output, ins0, concat_ins, false);
    }
    else
    {
        ins0->set_shape({ins0->get_shape().type(), new_lens});
        if(ins0->name() == "broadcast")
        {
            // workaround for a bad practice: broadcast has a broadcast_lens field.
            uint64_t ax  = (any_cast<op::broadcast>(ins0->get_operator())).axis;
            operation op = op::broadcast{ax, ins0->get_shape().lens()};
            std::vector<shape> input_shapes;
            input_shapes.push_back(ins0->inputs().at(0)->get_shape());
            shape new_s = op.compute_shape(input_shapes);
            ins0->replace(op);
            ins0->set_shape(new_s);
        }
    }

    if(output == base)
        output->set_shape({output->get_shape().type(), base_lens});
}

// If ins and input only diff in one axis, return that axis.
int horizontal_fusion_impl::find_unique_axis(instruction_ref ins, instruction_ref input)
{
    auto lens1 = ins->get_shape().lens();
    auto lens2 = input->get_shape().lens();
    if(lens1.size() != lens2.size())
        return -1;
    auto m = std::mismatch(lens1.begin(), lens1.end(), lens2.begin(), lens2.end());
    if(m.first == lens1.end())
        return -1;
    if(((std::distance(m.first, lens1.end()) > 1) or (std::distance(m.second, lens2.end()) > 1)) and
       not std::equal(m.first + 1, lens1.end(), m.second + 1, lens2.end()))
        return -1;
    return std::distance(lens1.begin(), m.first);
}

// find concat axis for ins.
int horizontal_fusion_impl::find_axis(instruction_ref ins,
                                      const std::unordered_map<instruction_ref, bool>& is_common)
{
    int axis = -1;
    for(auto&& input : ins->inputs())
    {
        if(is_common.find(input) != is_common.end())
        {
            int cur_axis = find_unique_axis(ins, input);
            if((cur_axis == -1) || ((axis != -1) && (cur_axis != axis)))
                return -1;
            axis = cur_axis;
        }
    }
    if(is_conv(ins) && (axis != get_channel_axis()))
        return -1;

    return axis;
}

// remove instructions in given vector except the first one.
void horizontal_fusion_impl::remove_redundant_roots(const std::vector<instruction_ref>& base_instrs)
{
    instruction_ref root_ins = base_instrs.at(0);
    for(auto base : range(base_instrs.begin() + 1, base_instrs.end()))
    {
        std::vector<instruction_ref> outputs = base->outputs();
        for(auto&& output : outputs)
            instruction::replace_argument(output, base, root_ins, false);
        p_program->remove_instruction(base);
    }
}

instruction_ref horizontal_fusion_impl::break_split(int enum_ndx, instruction_ref split_ins)
{
    const operation& op = split_ins->get_operator();
    int first           = (any_cast<op::horizontal_fusion_split>(op)).slice_selector.first;
    int second          = (any_cast<op::horizontal_fusion_split>(op)).slice_selector.second;
    if((first < 0) || (second < first))
        MIGRAPHX_THROW("unexpected selector");
    if((enum_ndx != first) && (enum_ndx != second))
        MIGRAPHX_THROW("unexpected slice enumerator");

    if(first == second)
        return split_ins;
    int axis                    = (any_cast<op::horizontal_fusion_split>(op)).axis;
    std::vector<int> slice_dims = (any_cast<op::horizontal_fusion_split>(op)).slice_dims;
    instruction_ref input       = split_ins->inputs().at(0);
    instruction_ref new_split   = p_program->insert_instruction(
        split_ins, op::horizontal_fusion_split{axis, slice_dims, {enum_ndx, enum_ndx}}, input);

    if(first == enum_ndx)
        first = enum_ndx + 1;
    else
        second = enum_ndx - 1;

    split_ins->replace(op::horizontal_fusion_split{axis, slice_dims, {first, second}});

    std::vector<shape> shapes;
    shapes.push_back(input->get_shape());
    shape new_shape =
        (any_cast<op::horizontal_fusion_split>(split_ins->get_operator())).compute_shape(shapes);
    split_ins->set_shape(new_shape);
    return new_split;
}

bool horizontal_fusion_impl::has_unique_output(const std::vector<instruction_ref>& instrs)
{
    std::unordered_map<instruction_ref, bool> seen;
    for(auto&& ins : instrs)
    {
        for(auto&& output : ins->outputs())
        {
            if(not seen.emplace(output, true).second)
                return false;
        }
    }
    return true;
}

std::vector<unsigned> horizontal_fusion_impl::find_cluster(unsigned id, value_numbering& val_num)
{
    std::vector<unsigned> cluster;
    cluster.push_back(id);
    unsigned cur = id;
    int size     = val_num.hash_instrs[id].size();
    // Find a sub-tree of the hash tree to be fused together.
    // Every node in the sub-tree contain the same amount of instructions.
    while((val_num.hash_outputs.find(cur) != val_num.hash_outputs.end()) &&
          (val_num.hash_outputs[cur].size() == 1))
    {
        unsigned output = (*(val_num.hash_outputs[cur].begin()))->id;
        // Currently instruction can not have more than one same outputs.
        // Therefore skip outputs that are not unique.
        if((val_num.hash_instrs.find(output) != val_num.hash_instrs.end()) &&
           (val_num.hash_instrs[output].size() == size) &&
           has_unique_output(val_num.get_instrs(output)))
        {
            cluster.push_back(output);
            cur = output;
        }
        else
            break;
    }
    return cluster;
}

void horizontal_fusion_impl::transform_layers(
    const std::vector<std::vector<instruction_ref>>& all_inputs,
    const std::unordered_map<instruction_ref, instruction_ref>& root,
    int axis,
    const std::vector<instruction_ref>& base_instrs)
{
    std::vector<instruction_ref> input0 = all_inputs.at(0);
    // concat inputs.
    for(unsigned long ndx = 0; ndx < input0.size(); ndx++)
    {
        std::vector<instruction_ref> instrs;
        std::transform(all_inputs.begin(),
                       all_inputs.end(),
                       std::back_inserter(instrs),
                       [&](auto&& d) -> instruction_ref { return d.at(ndx); });
        concat(instrs, root, axis);
    }
    remove_redundant_roots(base_instrs);

    // remove redundant inputs.
    for(auto&& input : range(all_inputs.begin() + 1, all_inputs.end()))
    {
        for(auto&& ins : range(input.rbegin(), input.rend()))
        {
            if(ins->name() == "@literal")
                continue;
            p_program->remove_instruction(ins);
        }
    }
}

// collect and compare inputs to be concated.
bool horizontal_fusion_impl::collect_inputs(
    std::vector<std::vector<instruction_ref>>& all_inputs,
    int& axis,
    std::vector<instruction_ref>& base_instrs,
    std::unordered_map<instruction_ref, bool>& visited,
    std::unordered_map<instruction_ref, instruction_ref>& root,
    std::unordered_map<instruction_ref, int>& split_axis)
{
    bool doit = true;
    for(auto&& ins : base_instrs)
    {
        // Find concat axis for ins.
        axis = (axis == -1) ? find_axis(ins, visited) : axis;
        if(axis == -1)
        {
            doit = false;
            break;
        }
        split_axis[ins]                     = axis;
        std::vector<instruction_ref> inputs = walk(ins, visited);
        if(inputs.empty() ||
           (!all_inputs.empty() && !compare_inputs(all_inputs.at(0), inputs, ins, axis)))
        {
            doit = false;
            break;
        }
        else
        {
            for(auto&& input : inputs)
                root[input] = ins;
            all_inputs.push_back(inputs);
        }
    }
    return doit;
}

void horizontal_fusion_impl::transform_output(
    unsigned last_hash_id,
    const std::unordered_map<instruction_ref, int>& split_axis,
    const std::unordered_map<instruction_ref, std::vector<std::vector<std::size_t>>>& orig_dims,
    const std::unordered_map<instruction_ref, std::vector<int>>& orig_clusters,
    value_numbering& val_num)
{

    std::vector<instruction_ref> base_instrs = val_num.get_instrs(last_hash_id);
    if(base_instrs.size() != 1)
        MIGRAPHX_THROW("Unexpect number of instructions");
    instruction_ref last_ins = base_instrs.at(0);
    if(split_axis.find(last_ins) == split_axis.end())
        MIGRAPHX_THROW("Split axis not found");

    int axis = split_axis.find(last_ins)->second;
    std::vector<int> slice_dims;
    std::transform(orig_dims.find(last_ins)->second.begin(),
                   orig_dims.find(last_ins)->second.end(),
                   std::back_inserter(slice_dims),
                   [&](auto&& d) -> int { return d.at(axis); });

    std::vector<instruction_ref> outputs;
    std::unordered_map<int, bool> enum2_concat;
    int enum_output           = 0;
    std::vector<int> clusters = orig_clusters.find(last_ins)->second;
    if(clusters.size() != last_ins->outputs().size())
        MIGRAPHX_THROW("Unmatched output size");

    for(auto&& output : last_ins->outputs())
    {
        outputs.push_back(output);
        if(is_concat(output))
            enum2_concat[clusters[enum_output]] = true;
        enum_output++;
    }

    instruction_ref insert_before = std::next(last_ins);
    auto split_ins                = p_program->insert_instruction(
        insert_before,
        op::horizontal_fusion_split{axis, slice_dims, {0, slice_dims.size() - 1}},
        last_ins);
    unsigned offset                            = 0;
    shape s                                    = last_ins->get_shape();
    std::vector<std::vector<std::size_t>> dims = orig_dims.find(last_ins)->second;
    std::vector<unsigned> offsets;

    for(auto&& dim : dims)
    {
        offsets.push_back(offset);
        shape orig_s = shape{s.type(), dim};
        offset += orig_s.bytes();
    }
    std::unordered_map<int, instruction_ref> enum2_instr;
    enum_output = 0;
    for(auto&& output : outputs)
    {
        int enum_ndx = clusters[enum_output++];
        shape orig_s = shape{s.type(), dims[enum_ndx]};
        instruction_ref new_ins;
        if(enum2_instr.find(enum_ndx) == enum2_instr.end())
        {
            bool add_load = true;
            if(enum2_concat.find(enum_ndx) != enum2_concat.end())
            {
                new_ins  = break_split(enum_ndx, split_ins);
                add_load = (new_ins == split_ins);
            }
            if(add_load)
            {
                const operation& op = split_ins->get_operator();
                shape input_s       = split_ins->inputs().at(0)->get_shape();
                unsigned offset_bias =
                    (any_cast<op::horizontal_fusion_split>(op)).compute_offset(input_s);
                offset_bias *= input_s.type_size();
                new_ins = p_program->insert_instruction(
                    insert_before, op::load{orig_s, offsets[enum_ndx] - offset_bias}, split_ins);
            }

            enum2_instr[enum_ndx] = new_ins;
        }
        else
        {
            new_ins = enum2_instr[enum_ndx];
        }
        instruction::replace_argument(output, last_ins, new_ins, false);
    }
}

void horizontal_fusion_impl::transform(value_numbering& val_num)
{
    for(auto&& val : val_num.values)
    {
        unsigned id = val.id;
        if((val_num.hash_instrs.find(id) == val_num.hash_instrs.end()) ||
           (val_num.hash_instrs[id].size() <= 1))
            continue;
        std::vector<unsigned> cluster = find_cluster(id, val_num);
        std::unordered_map<instruction_ref, bool> visited;
        std::unordered_map<instruction_ref, instruction_ref> root;
        std::unordered_map<instruction_ref, std::vector<std::vector<std::size_t>>> orig_dims;
        std::unordered_map<instruction_ref, int> split_axis;
        std::unordered_map<instruction_ref, std::vector<int>> orig_clusters;
        unsigned last_hash_id = 0;

        for(auto&& hash_id : cluster)
        {
            if(val_num.hash_inputs.find(hash_id) == val_num.hash_inputs.end())
                MIGRAPHX_THROW("Hash input not found");
            bool doit = true;
            // Flag common inputs which will not be concated.
            for(auto&& input : val_num.hash_inputs[hash_id])
            {
                std::vector<instruction_ref> instrs = val_num.get_instrs(input->id);
                if(instrs.size() != 1)
                {
                    doit = false;
                    break;
                }
                visited[instrs.at(0)] = true;
            }
            if(!doit)
                continue;

            std::vector<instruction_ref> base_instrs = val_num.get_instrs(hash_id);
            instruction_ref ins0                     = base_instrs.at(0);

            // save original dimensions.
            std::vector<std::vector<std::size_t>> lens;
            std::vector<int> clusters;
            int enum_ndx = 0;
            for(auto&& ins : base_instrs)
            {
                lens.push_back(ins->get_shape().lens());
                for(unsigned long i = 0; i < ins->outputs().size(); i++)
                    clusters.push_back(enum_ndx);
                enum_ndx++;
            }
            orig_dims[ins0]     = lens;
            orig_clusters[ins0] = clusters;

            if(ins0->inputs().size() == 1)
            {
                // concat single input instructions.
                instruction_ref input = ins0->inputs().at(0);
                if(split_axis.find(input) != split_axis.end())
                {
                    ins0->set_shape(input->get_shape());
                    remove_redundant_roots(base_instrs);
                    val_num.update_hash_tree(hash_id);
                    last_hash_id     = hash_id;
                    split_axis[ins0] = split_axis[input];
                }
                continue;
            }

            std::vector<std::vector<instruction_ref>> all_inputs;
            int axis = -1;

            if(!collect_inputs(all_inputs, axis, base_instrs, visited, root, split_axis))
                continue;

            transform_layers(all_inputs, root, axis, base_instrs);
            val_num.update_hash_tree(hash_id);
            last_hash_id = hash_id;
        }

        if(last_hash_id != 0)
            transform_output(last_hash_id, split_axis, orig_dims, orig_clusters, val_num);
    }
}

std::vector<instruction_ref>
horizontal_fusion_impl::walk(instruction_ref ins,
                             std::unordered_map<instruction_ref, bool>& visited)
{

    std::stack<instruction_ref> stk;
    for(auto&& input : ins->inputs())
    {
        if(visited.find(input) == visited.end())
            stk.push(input);
    }

    std::vector<instruction_ref> ret;
    while(!stk.empty())
    {
        instruction_ref top = stk.top();
        if((top->inputs().size() > 1) || (top->outputs().size() > 1) ||
           (top->inputs().empty() && (top->name() != "@literal")))
        {
            // Only seach for single-source single destination nodes and leaf nodes are literals.
            ret.clear();
            return ret;
        }
        else if(top->inputs().empty() || (visited.find(top) != visited.end()))
        {
            // Collect literal nodes and already marked nodes.
            ret.push_back(top);
            stk.pop();
        }
        else
        {
            // Mark current node and walk its input.
            instruction_ref input = top->inputs().at(0);
            stk.push(input);
            visited[top] = true;
        }
    }
    return ret;
}

void horizontal_fusion_impl::run()
{
    value_numbering val_num(p_program);
    val_num.run();
    transform(val_num);
    MIGRAPHX_DEBUG(dump_program());
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
