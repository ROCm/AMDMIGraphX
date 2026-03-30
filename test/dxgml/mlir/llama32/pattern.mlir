// RUN: %run_test %rtml-opt %s | %FileCheck %s

module {

    // Pattern: Add + RMSNorm Fusion
    // Fuses Add as prologue to RMSNorm (common in transformer residual connections)
    // Uses explicit pattern match to avoid recursive chaining of Adds
    dxgml_pattern.pattern @add_rmsnorm : benefit(40) {
        %result_types = types
        
        // Match Add op
        %add_lhs = operand
        %add_rhs = operand
        %add_type = type
        %add_op = operation "dxgml_op.add" (%add_lhs, %add_rhs : !dxgml_pattern.value, !dxgml_pattern.value) -> (%add_type : !dxgml_pattern.type)
        %add_result = result 0 of %add_op
        
        // Match RMSNorm that consumes the Add result
        %rmsnorm_scale = operand
        %rmsnorm_op = operation "dxgml_op.rms_normalization" (%add_result, %rmsnorm_scale : !dxgml_pattern.value, !dxgml_pattern.value) -> (%result_types : !dxgml_pattern.range<type>)
        
        %config = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<
            foreign_config = {jitFunction = "FusedAddRMSNorm"}
        >
        
        rewrite {
            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%add_op, %rmsnorm_op, %config : !dxgml_pattern.operation, !dxgml_pattern.operation, !dxgml_pattern.attribute)
        }
    }

    // Pattern: GEMM + DQ Prologue + SwiGLU Epilogue Fusion
    // Fuses: DQ on weight input + GEMM + Sigmoid + Multiply (SwiGLU activation)
    dxgml_pattern.pattern @gemm_dq_swiglu : benefit(30) {
        %result_types = types
        %inputs = operands
        
        // Match the base GEMM op
        %gemm_op = operation "dxgml_op.gemm" (%inputs : !dxgml_pattern.range<value>) -> (%result_types : !dxgml_pattern.range<type>)
        
        rewrite {
            %subgraph_rewrite_description = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<
                foreign_config = {jitFunction = "FusedGemmDQSwiGLU"}
                clusterFusionDescriptions = [
                    // Prologue: Fuse DQ on weight input (GEMM input 1)
                    #dxgml_subgraph_pattern.cluster_fusion_desc<
                        operatorIndex = 0,
                        inputIndex = 1,
                        supportedTypes = ["dxgml_op.dequantize_linear"],
                        fusionBenefit = 30
                    >,
                    // Epilogue: Fuse Sigmoid and Multiply for SwiGLU activation
                    #dxgml_subgraph_pattern.cluster_fusion_desc<
                        operatorIndex = 0,
                        resultIndex = 0,
                        supportedTypes = ["dxgml_op.sigmoid", "dxgml_op.multiply"],
                        fusionBenefit = 30
                    >
                ]
            >
            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%gemm_op, %subgraph_rewrite_description : !dxgml_pattern.operation, !dxgml_pattern.attribute)
        }
    }

    dxgml_pattern.pattern @transpose_to_subgraph : benefit(10) {
        %type = type
        %input = operand

        %transpose_op = operation "dxgml_op.transpose" (%input : !dxgml_pattern.value) -> (%type : !dxgml_pattern.type)

        %config = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<
            foreign_config = {jitFunction = "Transpose"}
        >

        rewrite {
            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%transpose_op, %config : !dxgml_pattern.operation, !dxgml_pattern.attribute)
        }
    }

    dxgml_pattern.pattern @rms_normalization_to_subgraph : benefit(1) {
        %result_types = types
        %inputs = operands

        %op = operation "dxgml_op.rms_normalization" (%inputs : !dxgml_pattern.range<value>) -> (%result_types : !dxgml_pattern.range<type>)

        %config = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<
            foreign_config = {jitFunction = "RMSNormalization"}
        >

        rewrite {
            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%op, %config : !dxgml_pattern.operation, !dxgml_pattern.attribute)
        }
    }

    dxgml_pattern.pattern @binary_ops_to_subgraph : benefit(1) {
        %result_types = types
        %inputs = operands

        %op = operation (%inputs : !dxgml_pattern.range<value>) -> (%result_types : !dxgml_pattern.range<type>)

        %binary_ops = attribute = [
            "dxgml_op.add",
            "dxgml_op.subtract",
            "dxgml_op.multiply",
            "dxgml_op.divide",
            "dxgml_op.gather"
        ]

        apply_native_constraint "dxgml_is_any_of" (%op, %binary_ops : !dxgml_pattern.operation, !dxgml_pattern.attribute)

        rewrite {
            %rewrite_desc = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<foreign_config = {jitFunction = "Binary"} clusterFusionDescriptions = []>
            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%op, %rewrite_desc : !dxgml_pattern.operation, !dxgml_pattern.attribute)
        }
    }

    dxgml_pattern.pattern @unary_ops_to_subgraph : benefit(1) {
        %result_types = types
        %inputs = operands

        %op = operation (%inputs : !dxgml_pattern.range<value>) -> (%result_types : !dxgml_pattern.range<type>)

        %unary_ops = attribute = [
            "dxgml_op.reduce",
            "dxgml_op.sqrt",
            "dxgml_op.reshape",
            "dxgml_op.slice",
            "dxgml_op.gelu",
            "dxgml_op.broadcast",
            "dxgml_op.cast"
        ]

        apply_native_constraint "dxgml_is_any_of" (%op, %unary_ops : !dxgml_pattern.operation, !dxgml_pattern.attribute)

        rewrite {
            %rewrite_desc = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<foreign_config = {jitFunction = "Unary"} clusterFusionDescriptions = []>
            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%op, %rewrite_desc : !dxgml_pattern.operation, !dxgml_pattern.attribute)
        }
    }

    dxgml_pattern.pattern @concat_to_subgraph : benefit(1) {
        %result_types = types
        %inputs = operands

        %op = operation "dxgml_op.concat" (%inputs : !dxgml_pattern.range<value>) -> (%result_types : !dxgml_pattern.range<type>)

        %config = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<
            foreign_config = {jitFunction = "Concat"}
        >

        rewrite {
            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%op, %config : !dxgml_pattern.operation, !dxgml_pattern.attribute)
        }
    }

    dxgml_pattern.pattern @gemm_to_subgraph : benefit(1) {
        %result_types = types
        %inputs = operands

        %op = operation "dxgml_op.gemm" (%inputs : !dxgml_pattern.range<value>) -> (%result_types : !dxgml_pattern.range<type>)

        %config = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<
            foreign_config = {jitFunction = "Gemm"}
        >

        rewrite {
            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%op, %config : !dxgml_pattern.operation, !dxgml_pattern.attribute)
        }
    }

    dxgml_pattern.pattern @group_query_attention_to_subgraph : benefit(1) {
        %result_types = types
        %inputs = operands

        %op = operation "dxgml_op.group_query_attention" (%inputs : !dxgml_pattern.range<value>) -> (%result_types : !dxgml_pattern.range<type>)

        %config = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<
            foreign_config = {jitFunction = "GroupQueryAttention"}
        >

        rewrite {
            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%op, %config : !dxgml_pattern.operation, !dxgml_pattern.attribute)
        }
    }
}
