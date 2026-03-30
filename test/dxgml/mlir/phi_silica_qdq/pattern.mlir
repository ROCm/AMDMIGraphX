// RUN: %run_test %rtml-opt %s | %FileCheck %s

module {

    // Pattern: Add + MeanVarianceNorm + Multiply Fusion
    // Fuses Add as prologue to MeanVarianceNorm, and Multiply as epilogue
    // Common in transformer residual connections with scale
    dxgml_pattern.pattern @add_meanvariancenorm_multiply : benefit(50) {
        %result_types = types
        
        // Match Add op
        %add_lhs = operand
        %add_rhs = operand
        %add_type = type
        %add_op = operation "dxgml_op.add" (%add_lhs, %add_rhs : !dxgml_pattern.value, !dxgml_pattern.value) -> (%add_type : !dxgml_pattern.type)
        %add_result = result 0 of %add_op
        
        // Match MeanVarianceNorm that consumes the Add result (3 inputs: input, weight, bias)
        %mvn_weight = operand
        %mvn_bias = operand
        %mvn_type = type
        %mvn_op = operation "dxgml_op.mean_variance_normalization" (%add_result, %mvn_weight, %mvn_bias : !dxgml_pattern.value, !dxgml_pattern.value, !dxgml_pattern.value) -> (%mvn_type : !dxgml_pattern.type)
        %mvn_result = result 0 of %mvn_op
        
        // Match Multiply that consumes the MeanVarianceNorm result
        %mul_rhs = operand
        %mul_op = operation "dxgml_op.multiply" (%mvn_result, %mul_rhs : !dxgml_pattern.value, !dxgml_pattern.value) -> (%result_types : !dxgml_pattern.range<type>)
        
        %config = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<
            foreign_config = {jitFunction = "FusedAddMeanVarianceNormMultiply"}
        >
        
        rewrite {
            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%add_op, %mvn_op, %mul_op, %config : !dxgml_pattern.operation, !dxgml_pattern.operation, !dxgml_pattern.operation, !dxgml_pattern.attribute)
        }
    }

    // Pattern: GEMM + DQ Prologue + activation Epilogue Fusion
    // Fuses: DQ on weight input + GEMM + activation
    dxgml_pattern.pattern @gemm_dq_act : benefit(30) {
        %result_types = types
        %inputs = operands
        
        // Match the base GEMM op
        %gemm_op = operation "dxgml_op.gemm" (%inputs : !dxgml_pattern.range<value>) -> (%result_types : !dxgml_pattern.range<type>)
        
        rewrite {
            %subgraph_rewrite_description = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<
                foreign_config = {jitFunction = "FusedGemmDQact"}
                clusterFusionDescriptions = [
                    // Prologue: Fuse Multiply on activation input (GEMM input 0)
                    #dxgml_subgraph_pattern.cluster_fusion_desc<
                        operatorIndex = 0,
                        inputIndex = 0,
                        supportedTypes = ["dxgml_op.multiply"],
                        fusionBenefit = 20
                    >,
                    // Prologue: Fuse DQ on weight input (GEMM input 1)
                    #dxgml_subgraph_pattern.cluster_fusion_desc<
                        operatorIndex = 0,
                        inputIndex = 1,
                        supportedTypes = ["dxgml_op.dequantize_linear"],
                        fusionBenefit = 30
                    >,
                    // Epilogue: Fuse Sigmoid and Mul 
                    #dxgml_subgraph_pattern.cluster_fusion_desc<
                        operatorIndex = 0,
                        resultIndex = 0,
                        supportedTypes = ["dxgml_op.sigmoid", "dxgml_op.multiply", "dxgml_op.relu", "dxgml_op.pow"],
                        fusionBenefit = 30
                    >
                ]
            >
            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%gemm_op, %subgraph_rewrite_description : !dxgml_pattern.operation, !dxgml_pattern.attribute)
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
            "dxgml_op.cast"
        ]

        apply_native_constraint "dxgml_is_any_of" (%op, %unary_ops : !dxgml_pattern.operation, !dxgml_pattern.attribute)

        rewrite {
            %rewrite_desc = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<foreign_config = {jitFunction = "Unary"} clusterFusionDescriptions = []>
            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%op, %rewrite_desc : !dxgml_pattern.operation, !dxgml_pattern.attribute)
        }
    }

    dxgml_pattern.pattern @mvn_to_subgraph : benefit(1) {
        %type = type
        %input = operand
        %gamma = operand
        %beta = operand

        %op = operation "dxgml_op.mean_variance_normalization" (%input, %gamma, %beta : !dxgml_pattern.value, !dxgml_pattern.value, !dxgml_pattern.value) -> (%type : !dxgml_pattern.type)

        %config = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<
            foreign_config = {jitFunction = "MeanVarianceNormalization"}
        >

        rewrite {
            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%op, %config : !dxgml_pattern.operation, !dxgml_pattern.attribute)
        }
    }

    dxgml_pattern.pattern @rotary_embedding_to_subgraph : benefit(1) {
        %result_types = types
        %inputs = operands

        %op = operation "dxgml_op.rotary_embedding" (%inputs : !dxgml_pattern.range<value>) -> (%result_types : !dxgml_pattern.range<type>)

        %config = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<
            foreign_config = {jitFunction = "RotaryEmbedding"}
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
}
