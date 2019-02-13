#include "pre_scheduling_impl.hpp"
#include <migraphx/iterator_for.hpp>
#include <migraphx/pass_config.hpp>
#include <stack>
namespace migraphx {

// Compute accumulated weights for each node in the DAG. Collect exit nodes
// and sort them according to accumulated weights.
//
void pre_scheduling_impl::compute_weights()
{
    int ndx = 0;
    std::unordered_map<dag_node*, bool> visited;
    for(auto ins : iterator_for(*p_program))
    {
        dag_node& node             = nodes[ndx];
        std::pair<int, int> weight = weight_func(ins->get_operator());
        node.weight                = weight.first;
        node.run_on_cpu            = weight.second;
        node.weight_sum += node.weight;
        visited.clear();

        for(auto&& arg : ins->inputs())
        {
            assert(instr2_node.find(arg) != instr2_node.end());
            dag_node* def_node = instr2_node[arg];
            if(visited.find(def_node) == visited.end())
            {
                node.weight_sum += def_node->weight_sum;
                visited[def_node] = true;
            }
        }
        if(ins->outputs().empty())
        {
            exit_nodes.push_back(&node);
        }
        node.ins         = ins;
        node.ins_ndx     = ndx++;
        instr2_node[ins] = &node;
    }
    int size = exit_nodes.size();
    if(size > 1)
    {
        std::sort(exit_nodes.begin(), exit_nodes.end(), compare_exit_nodes);
    }
}

// Do topology sort according to accumulated weight.  Identify critial paths.
// Schedule nodes into streams.  Reorder instructions according to topological
// order and annoate streams and events in the instructions.
//
void pre_scheduling_impl::reorder()
{
    std::list<dag_node*> sorted_nodes;
    std::stack<dag_node*> stack;
    std::priority_queue<dag_node*, std::vector<dag_node*>, weighted_topology_ordering> child_queue;
    std::unordered_map<dag_node*, bool> visited;
    std::unordered_map<dag_node*, bool> dequeued;

    for(auto&& node : exit_nodes)
    {
        stack.push(node);
        node->partition = partition_info.create_partition();
        partition_info.add_weight(node);
        while(!stack.empty())
        {
            auto cur = stack.top();
            if(dequeued.find(cur) != dequeued.end())
            {
                stack.pop();
                continue;
            }
            else if((visited.find(cur) != visited.end()) || cur->ins->inputs().empty())
            {
                stack.pop();
                sorted_nodes.push_back(cur);
                dequeued[cur] = true;
                continue;
            }
            // sort child nodes.
            for(auto&& arg : cur->ins->inputs())
            {
                dag_node* child_node = instr2_node[arg];
                if(dequeued.find(child_node) == dequeued.end())
                {
                    child_queue.push(child_node);
                }
            }

            // Last item in queue is on critical path.
            while(!child_queue.empty())
            {
                dag_node* child = child_queue.top();
                stack.push(child);
                child_queue.pop();
                if(child->weight_sum < min_partition_threshold)
                    child->partition = cur->partition;
                else if(!child_queue.empty())
                    child->partition = partition_info.create_partition();
                else
                {
                    cur->first_child = child;
                    child->partition = cur->partition;
                }
                partition_info.add_weight(child);
            }
            visited[cur] = true;
        }
    }

#ifdef MIGRAPHX_DEBUG_OPT
    MIGRAPHX_DEBUG(dump("---After weighted topology sort---"));
    MIGRAPHX_DEBUG(dump(sorted_nodes));
#endif
    schedule(sorted_nodes);
    splice(sorted_nodes);
    annotate(sorted_nodes);

    if(enable_verify)
        verify();
}

// Assign stream to nodes according to load balance.
//
int pre_scheduling_impl::get_stream(stream_info& info, dag_node* node)
{
    int max_cycle = info.max_cycle;
    if(max_cycle == 0)
        return 0;
    int partition_load   = partition_info.weight_sum[node->partition];
    int earliest_cycle   = node->earliest_cycle;
    int min_cycle        = -1;
    int min_cycle_stream = -1;
    for(auto stream = 0; stream < num_of_streams; ++stream)
    {
        int cycle = std::max(info.next_cycles[stream], earliest_cycle);
        if((cycle < max_cycle) && ((max_cycle - cycle) > partition_load))
            return stream;
        if((min_cycle_stream == -1) || (cycle < min_cycle))
        {
            min_cycle        = cycle;
            min_cycle_stream = stream;
        }
    }
    return min_cycle_stream;
}

//  Record the stream-assignment.
//
void pre_scheduling_impl::record(stream_info& info, dag_node* node)
{
    int stream               = node->stream;
    int next_cycle           = info.next_cycles[stream];
    node->sched_cycle        = std::max(node->earliest_cycle, next_cycle);
    next_cycle               = node->sched_cycle + node->weight;
    info.next_cycles[stream] = next_cycle;
    info.max_cycle           = std::max(info.max_cycle, next_cycle);
    for(auto&& arg : node->ins->outputs())
    {
        assert(instr2_node.find(arg) != instr2_node.end());
        dag_node* use_node       = instr2_node[arg];
        use_node->earliest_cycle = std::max(use_node->earliest_cycle, next_cycle);
    }
    if(node->can_use_stream())
        instr2_stream[node->ins] = stream;
}

//  Assign nodes to streams.
//
void pre_scheduling_impl::schedule(std::list<dag_node*>& sorted_nodes)
{
    if(num_of_streams == 0)
        return;
    stream_info info(num_of_streams);
    std::unordered_map<int, int> partition2_stream;
    partition2_stream.clear();

    for(auto&& node : sorted_nodes)
    {
        int cur_partition = node->partition;
        assert(cur_partition >= 0);
        if(partition2_stream.find(cur_partition) != partition2_stream.end())
        {
            node->stream = partition2_stream[cur_partition];
        }
        else
        {
            node->stream = get_stream(info, node);
        }
        assert(node->stream >= 0);
        record(info, node);
        partition2_stream[cur_partition] = node->stream;
    }

#ifdef MIGRAPHX_DEBUG_OPT
    MIGRAPHX_DEBUG(dump("---After assigning stream---"));
    MIGRAPHX_DEBUG(dump(sorted_nodes));
#endif
}

// Reorder the instructions ino topological order.
//
void pre_scheduling_impl::splice(std::list<dag_node*>& sorted_nodes)
{
    if(sorted_nodes.size() <= 1)
        return;
    auto begin                    = sorted_nodes.begin();
    auto iter                     = sorted_nodes.end();
    instruction_ref insert_before = (*(--iter))->ins;
    do
    {
        iter--;
        insert_before = p_program->move_instruction((*iter)->ins, insert_before);
    } while(iter != begin);

#ifdef MIGRAPHX_DEBUG_OPT
    MIGRAPHX_DEBUG(dump("---After splice in pre-scheduling---"));
    MIGRAPHX_DEBUG(dump_program());
#endif
}

//  Annotate streams and events in the instruction.  Insert set_stream
//  instructions.
//
void pre_scheduling_impl::annotate(std::list<dag_node*>& sorted_nodes)
{
    int event       = 0;
    int last_stream = -1;

    for(auto&& node : sorted_nodes)
    {
        instruction_ref ins = node->ins;
        if(instr2_stream.find(ins) == instr2_stream.end())
            continue;
        int stream = instr2_stream[ins];
        ins->set_stream(stream);
        if(last_stream != stream)
        {
            insert_instr.insert_stream(p_program, ins, stream);
            last_stream = stream;
        }
        std::vector<int> events;
        for(auto&& arg : ins->inputs())
        {
            if(instr2_stream.find(arg) == instr2_stream.end())
                continue;
            int arg_s = instr2_stream[arg];
            if(arg_s == stream)
                continue;
            if(!has_mask(arg, record_event))
            {
                events.push_back(event);
                insert_instr.insert_record_event(p_program, std::next(arg), event);
                event++;
            }
            add_mask(arg, record_event);
            add_mask(ins, wait_event);
        }
        for(auto&& i : events)
            insert_instr.insert_wait_event(p_program, ins, i);
    }
}

void pre_scheduling_impl::run()
{
    std::size_t num_of_instrs = p_program->size();
    if(num_of_instrs == 0)
        return;
    MIGRAPHX_DEBUG(dump("---Before pre-scheduling---"));
    MIGRAPHX_DEBUG(dump_program());
    nodes.resize(num_of_instrs);
    compute_weights();
    reorder();
}

void pre_scheduling_impl::verify()
{
    std::unordered_map<instruction_ref, bool> visited;
    for(auto ins : iterator_for(*p_program))
    {
        for(auto&& arg : ins->inputs())
        {
            if(visited.find(arg) == visited.end())
                MIGRAPHX_THROW("Input not visited");
        }
        visited[ins] = true;
    }
}

#ifdef MIGRAPHX_DEBUG_OPT
void pre_scheduling_impl::dump(const std::string& str) { std::cout << str << std::endl; }

void pre_scheduling_impl::dump_program() { std::cout << *p_program << std::endl; }

void pre_scheduling_impl::dump(std::list<dag_node*>& sorted_nodes)
{
    for(auto&& node : sorted_nodes)
    {
        node->dump();
        if(!node->ins->inputs().empty())
        {
            std::cout << " inputs: ";
            for(auto&& arg : node->ins->inputs())
            {
                dag_node* def_node = instr2_node[arg];
                std::cout << " @" << def_node->ins_ndx;
            }
            std::cout << std::endl;
        }
    }
}

void dag_node::dump()
{
    std::cout << " @" << ins_ndx;
    std::cout << " name: " << ins->name();
    std::cout << " weight: " << weight;
    std::cout << " weight_sum: " << weight_sum;
    if(can_use_stream())
        std::cout << " stream: " << stream;
    std::cout << " partition: " << partition;
    std::cout << " sched_cycle: " << sched_cycle;
    std::cout << std::endl;
}
#endif
} // namespace migraphx
