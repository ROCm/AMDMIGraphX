#include "horizontal_fusion_impl.hpp"
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static unsigned opcode_bits  = 16;
static unsigned hash_id_bits = 16;
static unsigned filter_bits  = 8;
static unsigned kernel_bits  = 8;

std::unordered_map<std::string, encoder> horizontal_fusion_impl::op_registry =
    horizontal_fusion_impl::create_op_registery();
std::unordered_map<std::string, int> horizontal_fusion_impl::op_flag =
    horizontal_fusion_impl::create_op_flag();

std::unordered_map<std::string, encoder> horizontal_fusion_impl::create_op_registery()
{
    std::unordered_map<std::string, encoder> m;
    m["gpu::convolution"]    = encode_conv_common;
    m["gpu::conv_bias_relu"] = encode_conv_common;
    m["hip::add_relu"]       = encode_common;
    m["convolution"]         = encode_conv_common;
    m["add"]                 = encode_common;
    m["relu"]                = encode_common;
    return m;
}

std::unordered_map<std::string, int> horizontal_fusion_impl::create_op_flag()
{
    std::unordered_map<std::string, int> m;
    m["gpu::convolution"]    = 1;
    m["gpu::conv_bias_relu"] = 1;
    m["hip::add_relu"]       = 0;
    m["convolution"]         = 1;
    m["add"]                 = 0;
    m["relu"]                = 0;
    return m;
}

static unsigned opcode_shift_count() { return ((sizeof(key_type) * 8) - opcode_bits); }

static unsigned hash_id_shift_count() { return (opcode_shift_count() - hash_id_bits); }

static unsigned filter_shift_count() { return (hash_id_shift_count() - filter_bits); }

static unsigned kernel_shift_count() { return (filter_shift_count() - kernel_bits); }

// Encode common fields.
//
// |----- opcode_bits -----|----- hash_id_bits -------|----- xxx bits -----|
// |      opcode           | 1st operand hash id      |
encode_info encode_common(instruction_ref ins, ins2_val& instr2_value, unsigned opcode)
{
    if(opcode >= (static_cast<unsigned>(1) << opcode_bits))
        return encode_info(0, false);
    key_type encode     = (static_cast<key_type>(opcode) << opcode_shift_count());
    instruction_ref op1 = ins->inputs().front();
    if(instr2_value.find(op1) == instr2_value.end())
        return encode_info(0, false);
    hash_value_ptr op1_val = instr2_value[op1];

    if((op1_val == nullptr) || (op1_val->id >= (static_cast<unsigned>(1) << hash_id_bits)))
        return encode_info(0, false);
    encode |= (static_cast<key_type>(op1_val->id) << hash_id_shift_count());
    encode_info info(encode, true);
    info.add_input(op1_val);
    return info;
}

// Encode common fields in convolution:
//
// |----- opcode_bits -----|----- hash_id_bits -------|----- filter_bits -----|-----
// kernel-bits-----|----- xxx bits -----| |     opcode            | 1st operand hash id      |
// filter size         |  kernel size         |      0x0000        |

encode_info encode_conv_common(instruction_ref ins, ins2_val& instr2_value, unsigned opcode)
{
    encode_info info = encode_common(ins, instr2_value, opcode);
    if(!info.is_valid())
        return info;
    key_type encode     = info.get_key();
    instruction_ref op2 = ins->inputs().at(1);
    auto lens           = op2->get_shape().lens();
    auto filter         = lens[lens.size() - 2];
    auto kernel         = lens[lens.size() - 1];
    if((filter < (static_cast<unsigned>(1) << filter_bits)) &&
       (kernel < (static_cast<unsigned>(1) << kernel_bits)))
    {
        encode |= (filter << filter_shift_count());
        encode |= (kernel << kernel_shift_count());
        info.set_key(encode);
        return info;
    }
    else
    {
        return encode_info(0, false);
    }
}

bool horizontal_fusion_impl::is_conv(instruction_ref ins)
{
    if(op_flag.find(ins->name()) == op_flag.end())
        return false;
    return (op_flag[ins->name()] == 1);
}

bool horizontal_fusion_impl::is_concat(instruction_ref ins) { return (ins->name() == "concat"); }

// Hash given instruction.
hash_value_ptr horizontal_fusion_impl::hash(instruction_ref ins)
{
    if(op_registry.find(ins->name()) == op_registry.end())
        return nullptr;

    encoder encode_func    = op_registry.at(ins->name());
    unsigned opcode        = hash_opcode(ins);
    encode_info encode_val = encode_func(ins, instr2_value, opcode);
    if(!encode_val.is_valid())
        return nullptr;

    key_type key            = encode_val.get_key();
    hash_value_ptr hash_val = nullptr;

    if(encode2_value.find(key) != encode2_value.end())
    {
        hash_val = encode2_value[key];
        add_instr(hash_val->id);
        instr2_value[ins] = hash_val;
    }
    else
    {
        hash_val           = &(create_value(ins));
        encode2_value[key] = hash_val;
    }
    for(auto&& input : encode_val.get_inputs())
    {
        add_input(hash_val->id, input);
        add_output(input->id, hash_val);
    }
    return hash_val;
}

hash_value& horizontal_fusion_impl::create_value(instruction_ref ins)
{
    unsigned id = static_cast<unsigned>(values.size());
    values.push_back(hash_value{id, cur_point});
    hash_value& val = get_value(id);
    add_instr(id);
    instr2_value[ins] = &val;
    return val;
}

void horizontal_fusion_impl::process(instruction_ref ins)
{
    // Do not hash literals.
    if(ins->name() == "@literal")
        return;
    if(instr2_hash.find(ins) != instr2_hash.end())
    {
        // Hash this instruction.
        if(hash(ins) != nullptr)
        {
            for(auto&& output : ins->outputs())
            {
                instr2_hash[output] = true;
            }
            return;
        }
    }

    std::unordered_map<std::string, int> op2_cnt;
    bool hash_child = false;
    // Do hash if at least two outputs have same operations.
    for(auto&& output : ins->outputs())
    {
        const std::string& str = output->name();
        if(op2_cnt.find(str) == op2_cnt.end())
            op2_cnt[str] = 1;
        else
        {
            op2_cnt[str] += 1;
            hash_child = true;
            break;
        }
    }
    if(hash_child)
    {
        // Create a value for this instruction.
        hash_value& value = create_value(ins);
        value.set_root();
        // Flag children to be hashed.
        for(auto&& output : ins->outputs())
        {
            if(op2_cnt[output->name()] > 1)
                instr2_hash[output] = true;
        }
    }
}

// Find the first axis that matches given dim.
int horizontal_fusion_impl::find_axis(instruction_ref ins, int dim)
{
    auto it = std::find(ins->get_shape().lens().begin(), ins->get_shape().lens().end(), dim);
    return (it != ins->get_shape().lens().end())
               ? (std::distance(ins->get_shape().lens().begin(), it))
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
        // concat literals.
        unsigned long long unit_slice   = 1;
        int ndx                         = 0;
        unsigned long long new_elements = 1;

        for(auto&& len : s.lens())
        {
            if(ndx > axis)
                unit_slice *= len;
            if(ndx != axis)
                new_elements *= len;
            ndx++;
        }
        new_elements *= sum;
        unsigned type_size             = s.type_size();
        unsigned long long total_bytes = new_elements * type_size;
        std::shared_ptr<char> input    = make_shared_array<char>(total_bytes);
        std::vector<unsigned long long> bytes_per_slice;
        unsigned long long unit_slice_bytes = unit_slice * type_size;

        std::transform(instrs.begin(),
                       instrs.end(),
                       std::back_inserter(bytes_per_slice),
                       [&](auto&& d) -> unsigned long long {
                           return d->get_shape().lens().at(axis) * unit_slice_bytes;
                       });

        unsigned copy_bytes = 0;
        int slice_ndx       = 0;
        char* dst           = input.get();
        while(copy_bytes < total_bytes)
        {
            unsigned ins_ndx = 0;
            for(auto&& ins : instrs)
            {
                unsigned long long bytes = bytes_per_slice[ins_ndx];
                const char* src          = &(ins->get_literal().data()[slice_ndx * bytes]);
                std::copy(src, src + bytes, dst);
                dst += bytes;
                copy_bytes += bytes;
                ins_ndx++;
            }
            slice_ndx++;
        }
        shape new_shape{s.type(), new_lens};
        auto new_literal = p_program->add_literal(literal(new_shape, input.get()));
        instruction::replace_argument(output, ins0, new_literal, false);
    }
    else
    {
        ins0->set_shape({ins0->get_shape().type(), new_lens});
        if(ins0->name() == "broadcast")
        {
            // workaround for a bad practice: broadcast has a broadcast_shape field.
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
    for(unsigned long ndx = 1; ndx < base_instrs.size(); ndx++)
    {
        instruction_ref base                 = base_instrs.at(ndx);
        std::vector<instruction_ref> outputs = base->outputs();
        for(auto&& output : outputs)
            instruction::replace_argument(output, base, root_ins, false);
        p_program->remove_instruction(base);
    }
}

// update hash tree.
void horizontal_fusion_impl::update_hash_tree(unsigned hash_id)
{
    unsigned first = *(hash_instrs[hash_id].begin());
    hash_instrs[hash_id].clear();
    //    hash_instrs[hash_id].insert(first);
    hash_instrs[hash_id] = {first};
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

std::vector<unsigned> horizontal_fusion_impl::find_cluster(unsigned id)
{
    std::vector<unsigned> cluster;
    cluster.push_back(id);
    unsigned cur = id;
    int size     = hash_instrs[id].size();
    // Find a sub-tree of the hash tree to be fused together.
    // Every node in the sub-tree contain the same amount of instructions.
    while((hash_outputs.find(cur) != hash_outputs.end()) && (hash_outputs[cur].size() == 1))
    {
        unsigned output = (*(hash_outputs[cur].begin()))->id;
        // Currently instruction can not have more than one same outputs.
        // Therefore skip outputs that are not unique.
        if((hash_instrs.find(output) != hash_instrs.end()) &&
           (hash_instrs[output].size() == size) && has_unique_output(get_instrs(output)))
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
    int ndx = 0;
    for(auto&& input : all_inputs)
    {
        for(auto&& ins : input)
        {
            bool is_literal = (ins->name() == "@literal");
            if((ndx == 0) && !is_literal)
                continue;
            p_program->remove_instruction(ins);
        }
        ndx++;
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
    const std::unordered_map<instruction_ref, std::vector<int>>& orig_clusters)
{

    std::vector<instruction_ref> base_instrs = get_instrs(last_hash_id);
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

void horizontal_fusion_impl::transform()
{
    for(auto&& val : values)
    {
        unsigned id = val.id;
        if((hash_instrs.find(id) == hash_instrs.end()) || (hash_instrs[id].size() <= 1))
            continue;
        std::vector<unsigned> cluster = find_cluster(id);
        std::unordered_map<instruction_ref, bool> visited;
        std::unordered_map<instruction_ref, instruction_ref> root;
        std::unordered_map<instruction_ref, std::vector<std::vector<std::size_t>>> orig_dims;
        std::unordered_map<instruction_ref, int> split_axis;
        std::unordered_map<instruction_ref, std::vector<int>> orig_clusters;
        unsigned last_hash_id = 0;

        for(auto&& hash_id : cluster)
        {
            if(hash_inputs.find(hash_id) == hash_inputs.end())
                MIGRAPHX_THROW("Hash input not found");
            bool doit = true;
            // Flag common inputs which will not be concated.
            for(auto&& input : hash_inputs[hash_id])
            {
                std::vector<instruction_ref> instrs = get_instrs(input->id);
                if(instrs.size() != 1)
                {
                    doit = false;
                    break;
                }
                visited[instrs.at(0)] = true;
            }
            if(!doit)
                continue;

            std::vector<instruction_ref> base_instrs = get_instrs(hash_id);
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
                    update_hash_tree(hash_id);
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
            update_hash_tree(hash_id);
            last_hash_id = hash_id;
        }

        if(last_hash_id != 0)
            transform_output(last_hash_id, split_axis, orig_dims, orig_clusters);
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
    MIGRAPHX_DEBUG(dump_program());

    for(auto ins : iterator_for(*p_program))
    {
        ins->id = cur_point;
        process(ins);
        point2_instr[cur_point] = ins;
        cur_point++;
    }
    MIGRAPHX_DEBUG(dump_hash_tree());
    transform();
    MIGRAPHX_DEBUG(dump_program());
}

#ifdef MIGRAPHX_DEBUG_OPT

void horizontal_fusion_impl::dump_program() { std::cout << *p_program << std::endl; }

void horizontal_fusion_impl::dump_hash_value(hash_value& val)
{
    unsigned id = val.id;
    std::cout << "id: " << id << " @" << val.cur_point;
    if(hash_inputs.find(id) != hash_inputs.end())
    {
        std::cout << " input: ";
        for(auto&& input : hash_inputs[id])
            std::cout << " " << input->id;
    }

    if(hash_outputs.find(id) != hash_outputs.end())
    {
        std::cout << " output: ";
        for(auto&& output : hash_outputs[id])
            std::cout << " " << output->id;
    }
    if(hash_instrs.find(id) != hash_instrs.end())
    {
        std::cout << " instrs: ";
        for(auto&& point : hash_instrs[id])
        {
            if(point2_instr.find(point) != point2_instr.end())
            {
                int ins_id = point2_instr[point]->id;
                if(ins_id > 0)
                    std::cout << " (" << ins_id << ")";
            }
            std::cout << " @" << point;
        }
    }
    std::cout << std::endl;
}

void horizontal_fusion_impl::dump_hash_tree()
{
    for(auto&& val : values)
    {
        dump_hash_value(val);
    }
}
#endif
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
