name() != "quant_dot")
            return false;
        // dot operation where (FP8 * FP8 = FP8) is not available in MLIR. rocBLAS/hipBLASLt should
        // have the support for it.
        if(contains(fp8_types{}.get(), ins->get_shape().type()))
            return false;
        // MX types quantization has 4 inputs
        // having all MX GEMM go to MLIR
        if(ins->inputs().size() == 4)
        {
            return true;
        }
        if(mode != mlir_mode::fast)
            return true;
        auto a = ins->inputs().front()->get_shape();
        auto b = ins->inputs().back()->get_shape();
        auto g = std::accumulate(a.lens().begin(), a.lens().end() - 2, 1, std::multiplies<>{});
        auto m = a.lens()[a.lens().size() - 2];
        auto n = b.lens().back();
        auto k = a.lens().back();
        // Skipping GEMMs with a K dimension greater than 2048 is a course-grained strategy
        // to avoid poor-performing GEMM kernels from MLIR
        // TODO: Investigate a more precise strategy
        if(k > 1535)
            return false;
        if(k < 1024)
            return true;
        return (g * m * n) < (384 * 384);
    });
}

auto is_mlir_conv(mlir_mode mode)
{
    return match::make_basic_pred_matcher([=](instruction_ref ins) {
        if(mode == mlir_mode::none)
            return false;
        if(ins->name() != "convolution" and ins->name() != "quant_convolution")
            return false;
        auto input = ins->inputs().front()->get_shape();
        value v    = ins->get_operator().to_value();
        auto group = v.at("group").to<int>();
        // Avoid MLIR assertion: Index < Length && "Invalid index!"
        if(ins->get_shape().lens().size() != 4 and group > 1)
            return false;
        std::set<shape::type_t> supported_types = fp8_types{}.get();
        supported_types.insert(shape::int8_type);
        if(contains(supported_types, input.type()))
            return true;
        if(mode == mlir_mode::all)
            return true;
        // No winograd for group convolution
        if(group > 1)
            return true;
        auto w = ins->inputs().at(1)->get_shape();
        if(w.lens().size() != 4)
            return true;
        if(w.lens()[2] != w.lens()[3])
            return true;
        return (w.lens()[3] % 3) != 0;
    });
}

std::unordered_map<instruction_ref, instruction_ref>
create_param_map_with_literals(module_ref mm, const module* pm, const shape& shape)
{
    std::unordered_map<instruction_ref, instruction_ref> ins_map;
    for(auto ins : iterator_for(*pm))
    {
        if(ins->name() != "@literal")
        {
            continue;
        }
        literal r               = ins->get_literal();
        instruction_ref literal = mm->add_literal(r);
        instruction_ref mbcast =
            mm->add_instruction(make_op("multibroadcast", {{"out_lens", shape.lens()}}), literal);
        ins_map[ins] = mbcast;
    }
    return ins_map;
}

instruction_ref insert_pointwise(module& m,
                                 instruction_ref ins,
                                 const operation& op,
                                 const std::vector<instruction_ref>& inputs,
                                 const std::vector<module_ref>& mod_args)
{
    // Only used in assert
    (void)mod_args;
    assert(mod_args.empty());
    return insert_common_op(m, ins, op, inputs, {.common_type = false});
}

instruction_ref unroll_pointwise(module& main_mod,
                                 instruction_ref pos,
                                 const operation& op,
                                 const std::vector<instruction_ref>& inputs,
                                 const std::vector<module_ref>& mod_args)
{
    if(op.name() == "pointwise")
    {
        auto* sub_pm     = mod_args.front();
        auto param_map_2 = create_param_map_with_literals(
            &main_mod, sub_pm, op.compute_shape(to_shapes(inputs), mod_args));
        return main_mod.insert_inline(pos, *sub_pm, inputs, &param_map_2)
            .front(); // cppcheck-suppress returnDanglingLifetime;
    }
    return main_mod.insert_instruction(pos, op, inputs, mod_args);
}

// Whitelist supported fusion options, including imposing type constraints
// for cases where MLIR only supports an operation (usually a pointwise function)
// on particular types.
bool is_pointwise_op_supported_by_mlir(const instruction& i)
{
    using type_t                                      = shape::type_t;
    const auto& name                                  = i.name();
    const auto result_type                            = i.get_shape().type();
    const std::initializer_list<type_t> allowed_types = {type_t::float_type,
                                                         type_t::bf16_type,
                                                         type_t::half_type,
                                                         type_t::fp8e4m3fnuz_type,
                                                         type_t::fp8e5m2fnuz_type,
                                                         type_t::fp8e4m3fn_type,
                                                         type_t::fp8e5m2_type,
                                                         type_t::int8_type,
                                                         type_t::uint8_type,
                                                         type_t::int32_type,
                                                         type_t::uint32_type,
                                                         type_t::bool_type};
    // Preliminary type check.
    if(not contains(allowed_types, result_type))
    {
        return false;
    }
    const std::initializer_list<std::string> any_type_ops = {"@literal", "@param", "@return"};
    const std::initializer_list<std::string> no_bool_ops  = {
        "convolution",
        "quant_convolution",
        "dot",
        "quant_dot",
        "add",
        "clip",
        "relu",
        "sub",
        "mul",
        "div",
        "pow",
        "where",
        "quantizelinear",
        "dequantizelinear",
        "abs",
        "neg",
    };
    const std::initializer_list<std::string> fp_only_ops = {
        "ceil",
        "erf",
        "exp",
        "floor",
        "log",
        "recip",
        "sqrt",
        "rsqrt",
        "sigmoid",
        "softmax",
        "tanh",
    };
    std::set<shape::type_t> float_types = {type_t::float_type,
                                           type_t::half_type,
                                           type_t::bf16_type,
                                           type_t::fp8e4m3fnuz_type,
                                           type_t::fp8e5m2fnuz_type,
                                           type_t::fp8e4m3fn_type,
                                           type_t::fp8e5m2_type};
    bool is_float                       = contains(float_types, result_type);
    if(contains(any_type_ops, name))
        return true;
    if(result_type != type_t::bool_type and contains(no_bool_ops, name))
        return true;
    if(is_float and contains(fp_only_ops, name))
        return true;
    // Only conversions between floating types are known to be unambigiously
    // supported.
    if(is_float and name == "convert")
    {
        if(contains(fp8_types{}.get(), result_type))
        {
            return false;
        } // else
        return std::all_of(i.inputs().begin(), i.inputs().end(), [](const auto& arg) {
            return contains({type_t::float_type, type_t::half_type, type_t::bf16_type},
                            arg->get_shape().type());
        });
    }
    return false;
}

bool is_reduce_op_supported_by_mlir(const instruction& i)
{
    using type_t                                      = shape::type_t;
    const auto& name                                  = i.name();
    const auto result_type                            = i.get_shape().type();
    const std::initializer_list<type_t> allowed_types = {type_t::float_type,
                                                         type_t::half_type,
                                                         type_t::bf16_type,
                                                         type_t::fp8e4m3fnuz_type,
                                                         type_t::fp8e5m2fnuz_type,
                                                         type_t::fp8e4m3fn_type,
                                                         type_t::fp8e5m2_type};

    // Preliminary type check.
    if(not contains(allowed_types, result_type))
    {
        return false;
    }
    const std::initializer_list<std::string> reduce_ops = {"reduce_mean", "reduce_sum"};
    return contains(reduce_ops, i.name());
}

// A separate function so we can remove operators that are supported by mlir
// but not supported for an input fusion.
bool is_pointwise_op_supported_by_mlir_for_input(const instruction& i)
{
    return is_pointwise_op_supported_by_mlir(i);
}

MIGRAPHX_PRED_MATCHER(mlir_split_reduce, instruction_ref ins)
{
    if(ins->name() != "split_fused_reduce")
        return false;
    auto* mod_arg                            = ins->module_inputs().front();
    std::unordered_set<std::string> builtins = {"@param", "@literal", "@return"};
    for(const auto i : iterator_for(*mod_arg))
    {
        if(is_reduce(*i))
        {
            if(not is_reduce_op_supported_by_mlir(*i))
                return false;
        }
        else if(i->name() == "pointwise")
        {
            if(not std::all_of(i->module_inputs().front()->begin(),
                               i->module_inputs().front()->end(),
                               &is_pointwise_op_supported_by_mlir))
                return false;
        }
        else if(not contains(reshaper_names(), i->name()) and not contains(builtins, i->name()))
        {
            return false;
        }
    }
    return true;
}

MIGRAPHX_PRED_MATCHER(mlir_pointwise, instruction_ref ins)
{
    if(ins->name() != "pointwise")
        return false;
    auto* pm = ins->module_inputs().front();
    return std::all_of(pm->begin(), pm->end(), &is_pointwise_op_supported_by_mlir);
}

MIGRAPHX_PRED_MATCHER(mlir_input_pointwise, instruction_ref ins)
{
    if(ins->name() != "pointwise")
        return false;
    auto* pm = ins->module_inputs().front();
    return std::all_of(pm->begin(), pm->end(), &is_pointwise_op_supported_by_mlir_for_input);
}

std::vector<instruction_ref> mlir_contiguous(module_pass_manager& mpm,
                                             const std::vector<instruction_ref>& inputs)
{
    std::vector<instruction_ref> result;
    std::transform(
        inputs.begin(), inputs.end(), std::back_inserter(result), [&](instruction_ref input) {
            if(input->get_shape().packed() or input->get_shape().broadcasted())
                return input;
            return mpm.get_module().insert_instruction(
                std::next(input), make_op("contiguous"), input);
        });
    return result;
}

struct find_mlir_split_reduce
{
    mlir_mode conv_mode = mlir_mode::none;
    mlir_mode dot_mode  = mlir_mode::none;
    auto matcher() const
    {
        auto dot_or_conv = match::name("gpu::mlir_op");
        // TODO: Handle reshapes inbetween
        return mlir_split_reduce()(match::any_of[match::inputs()](dot_or_conv.bind("gemm")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto reduce_ins = r.result;
        auto gemm_ins   = r.instructions["gemm"];
        assert(gemm_ins->get_shape().sub_shapes().empty());
        auto* rm   = reduce_ins->module_inputs().front();
        auto names = rm->get_parameter_names();
        std::sort(names.begin(), names.end());
        module_ref gemm_old_mm = gemm_ins->module_inputs().front();
        module_ref mm = mpm.create_module(gemm_old_mm->name() + "_" + rm->name(), *gemm_old_mm);
        // remove last return instruction
        if(std::prev(mm->end())->name() == "@return")
        {
            mm->remove_instruction(std::prev(mm->end()));
        }
        mm->set_bypass();
        std::unordered_map<instruction_ref, instruction_ref> param_map;
        param_map[gemm_ins]      = std::prev(mm->end());
        bool gemm_has_multi_outs = gemm_ins->outputs().size() > 1;
        auto return_vals = mm->fuse(*rm, reduce_ins->inputs(), &param_map, &unroll_pointwise);
        if(gemm_has_multi_outs)
        {
            return_vals.insert(return_vals.end(), param_map[gemm_ins]);
        }
        mm->add_return(return_vals);
        std::vector<instruction_ref> inputs;
        std::copy_if(reduce_ins->inputs().begin(),
                     reduce_ins->inputs().end(),
                     std::back_inserter(inputs),
                     [&](auto input) { return input != gemm_ins; });
        inputs.insert(inputs.end(), gemm_ins->inputs().begin(), gemm_ins->inputs().end());
        if(gemm_has_multi_outs)
        {
            auto fused_ins = mpm.get_module().insert_instruction(
                reduce_ins, mlir_op{gemm_ins->get_operator()}, mlir_contiguous(mpm, inputs), {mm});
            auto dot_ins = mpm.get_module().insert_instruction(
                reduce_ins,
                migraphx::make_op("get_tuple_elem", {{"index", return_vals.size() - 1}}),
                fused_ins);

            mpm.get_module().replace_instruction(gemm_ins, dot_ins);
            for(const auto& outs : reduce_ins->outputs())
            {
                assert(outs->get_operator().name() == "get_tuple_elem");
                mpm.get_module().replace_instruction(outs, outs->get_operator(), fused_ins);
            }
        }
        else
        {
            mpm.get_module().replace_instruction(
                reduce_ins, mlir_op{gemm_ins->get_operator()}, mlir_contiguous(mpm, inputs), {mm});
        }
    }
};

/**
 * Fuses rocMLIR compatible dot or conv op -> reshapes -> pointwise
 * into a mlir_op with submodule.
 */
struct find_mlir_fused_ops
{
    mlir_mode conv_mode = mlir_mode::none;
    mlir_mode dot_mode  = mlir_mode::none;

    static auto make_conv_dot_reshaper_names()
    {
        auto names = reshaper_names();
        names.erase("broadcast");
        names.erase("multibroadcast");
        return names;
    }

    /**
     * Matches:
     * mlir_dot_or_conv <binds to "gemm_based_op"> ->
     * skip(conv_dot_reshaper_names) <binds to "x"> ->
     * mlir_pointwise <matcher result>
     */
    auto matcher() const
    {
        static const auto conv_dot_reshaper_names = make_conv_dot_reshaper_names();
        auto dot_or_conv = match::skip(match::name(conv_dot_reshaper_names))(
            match::any_of(is_mlir_dot(dot_mode), is_mlir_conv(conv_mode)).bind("gemm_based_op"));
        return mlir_pointwise()(match::any_of[match::inputs()](dot_or_conv.bind("x")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto pw_ins        = r.result;
        auto gemm_based_op = r.instructions["gemm_based_op"];
        auto x_ins         = r.instructions["x"]; // input to pointwise after reshaper op stream
        auto* pm           = pw_ins->module_inputs().front();
        auto pw_inputs     = pw_ins->inputs();
        // only of one of the inputs to pointwise module should be dependent on conv/gemm that is
        // being fused, otherwise it can create invalid graph transformation
        if(std::any_of(pw_inputs.begin(), pw_inputs.end(), [&](const auto& i) {
               return i != x_ins and reaches(gemm_based_op, i);
           }))
            return;

        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        module_ref mm = mpm.create_module("mlir_" + pm->name());
        mm->set_bypass();
        fuse_input_ops(mm, gemm_based_op->inputs(), &map_ins);

        bool gemm_has_multi_outs = gemm_based_op->outputs().size() > 1;
        std::vector<instruction_ref> inss_to_insert;
        auto reshape_ins = x_ins;
        for(; reshape_ins != gemm_based_op; reshape_ins = reshape_ins->inputs().front())
        {
            inss_to_insert.push_back(reshape_ins);
            gemm_has_multi_outs |= reshape_ins->outputs().size() > 1;
        }
        inss_to_insert.push_back(gemm_based_op);
        std::reverse(inss_to_insert.begin(), inss_to_insert.end());
        mm->add_instructions(inss_to_insert, &map_ins);

        fuse_input_ops(mm, pw_ins->inputs(), &map_ins);
        auto rins = mm->fuse(*pm, pw_ins->inputs(), &map_ins, &insert_pointwise);
        if(gemm_has_multi_outs)
        {
            rins.push_back(map_ins.at(gemm_based_op));
        }
        mm->add_return(rins);

        auto inputs    = find_inputs(map_ins, &mpm.get_module(), mm);
        auto fused_ins = mpm.get_module().insert_instruction(
            pw_ins, mlir_op{gemm_based_op->get_operator()}, mlir_contiguous(mpm, inputs), {mm});
        if(gemm_has_multi_outs)
        {
            auto dot_ins = mpm.get_module().insert_instruction(
                pw_ins,
                migraphx::make_op("get_tuple_elem", {{"index", rins.size() - 1}}),
                fused_ins);

            // move all the reshape instructions after the fused op to avoid
            // generating invalid migraphx program since the reshapes can be
            // used by the replaced dot_ins
            for(instruction_ref x : inss_to_insert)
            {
                if(x == gemm_based_op)
                    continue;
                mpm.get_module().move_instruction(x, pw_ins);
            }

            mpm.get_module().replace_instruction(gemm_based_op, dot_ins);
            if(rins.size() == 2)
            {
                mpm.get_module().replace_instruction(
                    pw_ins, migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused_ins);
            }
        }
        else
        {
            mpm.get_module().replace_instruction(pw_ins, fused_ins);
        }
    }
};

template <auto Matcher>
struct find_mlir_standalone_op
{
    mlir_mode mode       = mlir_mode::none;
    std::size_t* counter = nullptr;
    auto matcher() const { return Matcher(mode); }

    std::string get_count() const
    {
        if(counter == nullptr)
            MIGRAPHX_THROW("Invalid counter");
        return std::to_string((*counter)++);
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto gemm_based_op = r.result;
        // enable only for fp32/fp16/i8/fp8 types
        if(std::any_of(gemm_based_op->inputs().begin(), gemm_based_op->inputs().end(), [&](auto i) {
               return not contains({shape::type_t::float_type,
                                    shape::type_t::half_type,
                                    shape::type_t::bf16_type,
                                    shape::type_t::int8_type,
                                    shape::type_t::fp8e4m3fnuz_type,
                                    shape::type_t::fp8e5m2fnuz_type,
                                    shape::type_t::fp8e4m3fn_type,
                                    shape::type_t::fp8e5m2_type},
                                   i->get_shape().type());
           }))
            return;
        std::string module_name = "mlir_" + gemm_based_op->name() + get_count();
        if(mpm.get_module().name() != "main")
            module_name = mpm.get_module().name() + ":" + module_name;
        module_ref mm = mpm.create_module(module_name);
        mm->set_bypass();
        auto [anchor_op, top_inputs] = fuse_input_ops_and_gemm_based_op(
            mm, gemm_based_op->inputs(), gemm_based_op->get_operator());
        mm->add_return({anchor_op});
        mpm.get_module().replace_instruction(gemm_based_op,
                                             mlir_op{gemm_based_op->get_operator()},
                                             mlir_contiguous(mpm, top_inputs),
                                             {mm});
    }
};

using find_mlir_standalone_convolution_op = find_mlir_standalone_op<&is_mlir_conv>;
using find_mlir_standalone_dot_op         = find_mlir_standalone_op<&is_mlir_dot>;

struct find_mlir_attention_op
{
    auto matcher() const
    {
        return match::name("group")(match::has_op_value("tag", "attention")).bind("group");
    }

    std::unordered_map<instruction_ref, instruction_ref>
    invert_map_ins(const std::unordered_map<instruction_ref, instruction_ref>& map_ins) const
    {
        std::unordered_map<instruction_ref, instruction_ref> inverse_map;
        for(auto const& [key, value] : map_ins)
        {
            assert(not contains(inverse_map, value));
            inverse_map[value] = key;
        }
        return inverse_map;
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto group = r.instructions["group"];

        auto* group_mod = group->module_inputs().front();

        std::string module_name = "mlir_" + group_mod->name();
        module_ref mlir_attn    = mpm.create_module(module_name);
        mlir_attn->set_bypass();

        // Fuse any input reshapes
        std::unordered_map<instruction_ref, instruction_ref> map_main_to_mlir_attn;
        fuse_input_ops(mlir_attn, group->inputs(), &map_main_to_mlir_attn);

        std::unordered_map<instruction_ref, instruction_ref> map_group_mod_to_mlir_attn(
            map_main_to_mlir_attn);
        auto attn_outs = mlir_attn->fuse(*group_mod, group->inputs(), &map_group_mod_to_mlir_attn);

        std::unordered_map<instruction_ref, instruction_ref> inss_to_replace;
        // Only fuse attention outputs that are used once by pointwise ops
        if(group->outputs().size() == 1 or attn_outs.size() > 1)
        {
            std::size_t out_idx  = 0;
            auto attn_output_ins = group;
            for(auto out : group->outputs())
            {
                auto op = out->get_operator();
                if(op.name() == "get_tuple_elem")
                {
                    inss_to_replace[out]  = out;
                    out_idx               = op.to_value()["index"].to<std::size_t>();
                    auto tuple_elem_users = out->outputs();
                    if(tuple_elem_users.size() > 1)
                        continue;

                    attn_output_ins = out;
                    out             = tuple_elem_users.front();
                }

                auto match_pw = match::match_instruction(mpm.get_module(), out, mlir_pointwise());
                if(match_pw.result != out)
                    continue;

                map_main_to_mlir_attn[attn_output_ins] = attn_outs[out_idx];
                inss_to_replace[attn_output_ins]       = out;

                auto lit_map = create_param_map_with_literals(
                    mlir_attn, out->module_inputs().front(), out->get_shape());
                mlir_attn->add_params(out->inputs(), &map_main_to_mlir_attn);
                map_main_to_mlir_attn.insert(lit_map.begin(), lit_map.end());
                std::unordered_map<instruction_ref, instruction_ref> map_pm_to_mlir_attn(
                    map_main_to_mlir_attn);
                auto fused_pw_outs = mlir_attn->fuse(
                    *out->module_inputs().front(), out->inputs(), &map_pm_to_mlir_attn);
                assert(fused_pw_outs.size() == 1);

                map_main_to_mlir_attn[out] = fused_pw_outs.front();
                attn_outs[out_idx]         = fused_pw_outs.front();
            }
        }
        mlir_attn->add_return(attn_outs);

        auto map_mlir_attn_to_main = invert_map_ins(map_main_to_mlir_attn);
        auto new_inputs            = mlir_attn->get_inputs(map_mlir_attn_to_main);

        auto mlir_ins = mpm.get_module().insert_instruction(
            group, mlir_op{make_op("dot")}, mlir_contiguous(mpm, new_inputs), {mlir_attn});

        if(inss_to_replace.empty())
        {
            mpm.get_module().replace_instruction(group, mlir_ins);
        }
        else
        {
            for(auto const& [attn_out, fused_user] : inss_to_replace)
            {
                auto replace_ins = mlir_ins;
                auto output_op   = attn_out->get_operator();
                if(output_op.name() == "get_tuple_elem")
                    replace_ins = mpm.get_module().insert_instruction(group, output_op, {mlir_ins});
                mpm.get_module().replace_instruction(fused_user, replace_ins);
            }
        }
    }
};

/**
 * Input fusion of pointwise operators into a mlir_op.
 * Only fuses unary pointwise operators by default.
 * Fuses all fusable pw ops with MIGRAPHX_ENABLE_MLIR_INPUT_FUSION
 */
struct find_pointwise_mlir
{
    auto supported_pointwise() const { return mlir_input_pointwise(match::used_once()); }

    auto matcher() const
    {
        return match::name("gpu::mlir_op")(match::any_of[match::inputs()](supported_pointwise()));
    }

    static bool is_simple_op(const_module_ref pm, std::initializer_list<std::string> op_names)
    {
        auto last = std::prev(pm->end());
        assert(last->name() == "@return");
        if(last->inputs().size() != 1)
            return false;
        auto rins   = last->inputs().front();
        auto op_ins = std::find_if(pm->begin(), pm->end(), [](const instruction& x) {
            return not contains({"@param", "@literal", "broadcast", "multibroadcast"}, x.name());
        });
        if(op_ins != rins)
            return false;
        return contains(op_names, op_ins->name());
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;

        auto* mm = ins->module_inputs().front();
        std::vector<instruction_ref> pws;
        std::copy_if(
            ins->inputs().begin(),
            ins->inputs().end(),
            std::back_inserter(pws),
            [&](instruction_ref input) {
                if(not match::instruction_matches(mpm.get_module(), input, supported_pointwise()))
                    return false;
                auto* pm = input->module_inputs().front();
                if(input->inputs().size() > 1 and not is_simple_op(pm, {"dequantizelinear"}))
                {
                    if(not enabled(MIGRAPHX_ENABLE_MLIR_INPUT_FUSION{}))
                        return false;
                }
                return true;
            });
        if(pws.empty())
            return;

        std::string module_name;
        std::transform(
            pws.begin(), pws.end(), join_back_inserter(module_name), [](instruction_ref pw) {
                return pw->module_inputs().front()->name() + ":";
            });
        module_name += mm->name();
        module_ref m = mpm.create_module(module_name);
        m->set_bypass();

        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        for(auto pw : pws)
        {
            auto* pm = pw->module_inputs().front();
            fuse_input_ops(m, pw->inputs(), &map_ins);
            auto rins   = m->fuse(*pm, pw->inputs(), &map_ins, &insert_pointwise).front();
            map_ins[pw] = rins;
        }

        auto ret = m->fuse(*mm, ins->inputs(), &map_ins);
        m->add_return({ret});

        auto inputs = find_inputs(map_ins, &mpm.get_module(), m);
        mpm.get_module().replace_instruction(
            ins, ins->get_operator(), mlir_contiguous(mpm, inputs), {m});
    }
};

struct find_unpack_int4_mlir_op
{
    auto matcher() const
    {
        return match::name("gpu::mlir_op")(
            match::any_of[match::inputs()](match::name("unpack_int4").bind("unpack_int4")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins      = r.result;
        auto* mm      = ins->module_inputs().front();
        module_ref nm = mpm.create_module("int4:" + mm->name());
        nm->set_bypass();

        std::vector<instruction_ref> x_in;
        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        int ct = 0;

        for(auto input : ins->inputs())
        {
            if(input->get_operator().name() == "unpack_int4")
            {
                auto unpack_input = input->inputs()[0];
                instruction_ref t_ins =
                    nm->add_parameter(param_name(++ct), unpack_input->get_shape().as_standard());
                map_ins[input] = nm->add_instruction(input->get_operator(), t_ins);
                x_in.push_back(unpack_input);
            }
            else
            {
                map_ins[input] =
                    nm->add_parameter(param_name(++ct), input->get_shape().as_standard());
                x_in.push_back(input);
            }
        }
        auto ret = nm->fuse(*mm, ins->inputs(), &map_ins);
        nm->add_return({ret});
        mpm.get_module().replace_instruction(ins, ins->get_operator(), x_in, {nm});
    }
};

/**
 * Move unpack_fp4 instructions into the mlir_op submodule.
 * Slice and reshape instructions should already be fused into the mlir_op.
 * rocMLIR will do the unpacking and dequantization.
 */
struct find_unpack_fp4_mlir_op
{
    auto matcher() const
    {
        return match::name("gpu::mlir_op")(
            match::any_of[match::inputs()](match::name("unpack_fp4")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& mr) const
    {
        auto mlir_op  = mr.result;
        auto* mm      = mlir_op->module_inputs().front();
        module_ref nm = mpm.create_module("fp4:" + mm->name());
        nm->set_bypass();
        std::vector<instruction_ref> new_mlir_op_args;
        std::unordered_map<instruction_ref, instruction_ref> fuse_ins_map;
        int ct = 0;
        for(auto curr_ins : mlir_op->inputs())
        {
            if(curr_ins->name() == "unpack_fp4")
            {
                auto unpack_ins   = curr_ins;
                auto unpack_input = unpack_ins->inputs().at(0);
                auto param =
                    nm->add_parameter(param_name(++ct), unpack_input->get_shape().as_standard());
                auto new_unpack_ins      = nm->add_instruction(unpack_ins->get_operator(), param);
                fuse_ins_map[unpack_ins] = new_unpack_ins;
                new_mlir_op_args.push_back(unpack_input);
            }
            else
            {
                fuse_ins_map[curr_ins] =
                    nm->add_parameter(param_name(++ct), curr_ins->get_shape().as_standard());
                new_mlir_op_args.push_back(curr_ins);
            }
        }
        auto ret = nm->fuse(*mm, mlir_op->inputs(), &fuse_ins_map);
        nm->add_return({ret});
        mpm.get_module().replace_instruction(
            mlir_op, mlir_op->get_operator(), new_mlir_op_args, {nm});
    }
};

/**
            find_mlir_split_reduce{.conv_mode = get_mode("fused_convolution", mlir_mode::fast),
                                   .dot_mode  = get_mode("fused_dot", mlir_mode::fast)});
    }

    match::find_matches(mpm, find_pointwise_mlir{});
    match::find_matches(mpm, find_unpack_int4_mlir_op{});
    match::find_matches(mpm, find_unpack_fp4_mlir_op{});

    match::find_matches(mpm, find_mlir_output_reshape_ops{});

#else
    (void)mpm;
#endif
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
