#ifndef MIGRAPHX_GUARD_RTGLIB_PRE_SCHEDULING_IMPL_HPP
#define MIGRAPHX_GUARD_RTGLIB_PRE_SCHEDULING_IMPL_HPP
#include <migraphx/common_header.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/insert_instruction.hpp>

namespace migraphx {

struct dag_node
{
    dag_node()
    {
        weight         = 0;
        run_on_cpu     = 0;
        weight_sum     = 0;
        ins_ndx        = -1;
        first_child    = nullptr;
        stream         = -1;
        partition      = -1;
        sched_cycle    = -1;
        earliest_cycle = -1;
    }
    int weight;
    int run_on_cpu;
    int weight_sum;
    int ins_ndx;
    dag_node* first_child;
    int stream;
    int partition;
    int sched_cycle;
    int earliest_cycle = -1;
    instruction_ref ins;
    bool is_literal() const { return (ins->name() == "@literal"); }
    bool can_use_stream() const { return (run_on_cpu == 0); }

#ifdef MIGRAPHX_DEBUG_OPT
    void dump();
#endif
};

struct dag_partition
{
    dag_partition()
    {
        num_of_partition = 0;
        weight_sum.clear();
    }

    int create_partition()
    {
        weight_sum.push_back(0);
        return num_of_partition++;
    }
    void add_weight(dag_node* node)
    {
        if(node->partition >= 0)
        {
            assert(node->partition < num_of_partition);
            weight_sum[node->partition] += node->weight;
        }
    }

    int num_of_partition;
    std::vector<int> weight_sum;
};

struct stream_info
{
    stream_info(int n) : num_of_streams(n)
    {
        max_cycle = 0;
        next_cycles.clear();
        for(auto stream = 0; stream < num_of_streams; ++stream)
            next_cycles.push_back(0);
    }
    std::vector<int> next_cycles;
    int num_of_streams;
    int max_cycle;
};

struct pre_scheduling_impl
{
    pre_scheduling_impl(program* p,
                        std::function<std::pair<int, int>(const operation&)> w,
                        int n,
                        insert_instruction ins,
                        bool v)
        : p_program(p),
          weight_func(std::move(w)),
          num_of_streams(n),
          insert_instr(std::move(ins)),
          enable_verify(v)
    {
        instr2_node.clear();
        instr2_mask.clear();
        instr2_stream.clear();
    }
    void schedule(std::list<dag_node*>&);
    void compute_weights();
    int get_stream(stream_info&, dag_node*);
    void record(stream_info&, dag_node*);
    void reorder();
    void run();
    void splice(std::list<dag_node*>&);
    void annotate(std::list<dag_node*>&);
    static bool compare_exit_nodes(dag_node* d1, dag_node* d2)
    {
        return (d1->weight_sum > d2->weight_sum);
    }

    struct weighted_topology_ordering
    {
        bool operator()(const dag_node* d1, const dag_node* d2) const
        {
            if(d1->weight_sum < d2->weight_sum)
            {
                // smaller weigth_sum is placed on top of the queue.
                return false;
            }
            else if(d1->weight_sum > d2->weight_sum)
            {
                return true;
            }
            else
            {
                // smaller instrution index is placed on top of the queue,
                return d1->ins_ndx > d2->ins_ndx;
            }
        }
    };

    struct post_schedule_ordering
    {
        bool operator()(const dag_node* d1, const dag_node* d2) const
        {
            if(d1->sched_cycle == d2->sched_cycle)
            {

                if(d1->stream == d2->stream)
                {
                    // smaller instruction index on top of queue.
                    return d1->ins_ndx > d2->ins_ndx;
                }
                else
                {
                    // smaller stream on top of queue.
                    return (d1->stream > d2->stream);
                }
            }
            else
            {
                // smaller sched_cycle on top of queue.
                return (d1->sched_cycle > d2->sched_cycle);
            }
        }
    };

    bool has_mask(instruction_ref ins, unsigned int m)
    {
        if(instr2_mask.find(ins) != instr2_mask.end())
        {
            unsigned int mask = instr2_mask[ins];
            return ((mask & (1u << m)) != 0);
        }
        return false;
    }

    void add_mask(instruction_ref ins, unsigned int m)
    {
        unsigned int mask = (instr2_mask.find(ins) != instr2_mask.end()) ? instr2_mask[ins] : 0;
        if((mask & (1u << m)) == 0)
            instr2_mask[ins] = (mask + (1u << m));
    }
    void verify();

#ifdef MIGRAPHX_DEBUG_OPT
    void dump(const std::string&);
    void dump_program();
    void dump(std::list<dag_node*>&);
#endif
    static const int min_partition_threshold = 2;

    private:
    program* p_program;
    std::function<std::pair<int, int>(const operation&)> weight_func;
    int num_of_streams;
    insert_instruction insert_instr;
    std::vector<dag_node> nodes;
    std::vector<dag_node*> exit_nodes;
    std::unordered_map<instruction_ref, dag_node*> instr2_node;
    std::unordered_map<instruction_ref, int> instr2_stream;
    std::unordered_map<instruction_ref, unsigned int> instr2_mask;
    dag_partition partition_info;
    bool enable_verify;
};
} // namespace migraphx
#endif
