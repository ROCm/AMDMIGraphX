// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
// This pattern matches the decomposed GELU and fuses it into dxgml_op.gelu
module {
    // Pattern matches:
    //   %div = divide(%input, %sqrt2_const)      // x / sqrt(2)
    //   %erf = erf(%div)                         // erf(x / sqrt(2))
    //   %add = add(%erf, %one_const)             // 1 + erf(...)
    //   %mul1 = multiply(%input, %add)           // x * (1 + erf(...))
    //   %mul2 = multiply(%mul1, %half_const)     // 0.5 * x * (1 + erf(...))
    // Rewrites to:
    //   %gelu = gelu(%input)
    
    dxgml_pattern.pattern @gelu_fusion : benefit(100) {
        %type = type
        %input = operand
        
        // Constants for sqrt(2), 1.0, and 0.5
        %sqrt2_const = operand
        %one_const = operand  
        %half_const = operand

        // x / sqrt(2)
        %div_op = operation "dxgml_op.divide"(%input, %sqrt2_const : !dxgml_pattern.value, !dxgml_pattern.value) -> (%type : !dxgml_pattern.type)
        %div_res = result 0 of %div_op
        
        // erf(x / sqrt(2))
        %erf_op = operation "dxgml_op.erf"(%div_res : !dxgml_pattern.value) -> (%type : !dxgml_pattern.type)
        %erf_res = result 0 of %erf_op
        
        // 1 + erf(...)
        %add_op = operation "dxgml_op.add"(%erf_res, %one_const : !dxgml_pattern.value, !dxgml_pattern.value) -> (%type : !dxgml_pattern.type)
        %add_res = result 0 of %add_op
        
        // x * (1 + erf(...))
        %mul1_op = operation "dxgml_op.multiply"(%input, %add_res : !dxgml_pattern.value, !dxgml_pattern.value) -> (%type : !dxgml_pattern.type)
        %mul1_res = result 0 of %mul1_op

        // 0.5 * x * (1 + erf(...))
        %mul2_op = operation "dxgml_op.multiply"(%mul1_res, %half_const : !dxgml_pattern.value, !dxgml_pattern.value) -> (%type : !dxgml_pattern.type)
        
        rewrite %mul2_op {
            %gelu_op = operation "dxgml_op.gelu"(%input : !dxgml_pattern.value) -> (%type : !dxgml_pattern.type)
            replace %mul2_op with %gelu_op
        }
    }


    // MVN (Mean Variance Normalization) / LayerNorm Pattern
    // Pattern matches:
    //   %mean = reduce(%input)                   // mean (reduce_function_average)
    //   %centered = subtract(%input, %mean)      // x - mean
    //   %squared = pow(%centered, %power_const)  // (x - mean)^2
    //   %variance = reduce(%squared)             // variance (reduce_function_average)
    //   %var_eps = add(%variance, %epsilon)      // variance + epsilon
    //   %stddev = sqrt(%var_eps)                 // sqrt(variance + epsilon)
    //   %normalized = divide(%centered, %stddev) // (x - mean) / stddev
    //   %scaled = multiply(%normalized, %gamma)  // scale by gamma
    //   %output = add(%scaled, %beta)            // shift by beta
    // Rewrites to:
    //   %mvn = mean_variance_normalization(%input, %gamma, %beta)

    dxgml_pattern.pattern @mvn_fusion : benefit(100) {
        %input_type = type
        %reduced_type = type
        %input = operand
        
        // Constants
        %epsilon_const = operand   // epsilon for numerical stability
        %power_const = operand     // power of 2.0
        %gamma = operand           // scale parameter
        %beta = operand            // shift parameter

        // Mean: reduce(input, average)
        %mean_op = operation "dxgml_op.reduce"(%input : !dxgml_pattern.value) -> (%reduced_type : !dxgml_pattern.type)
        %mean_result = result 0 of %mean_op

        // Centered: input - mean
        %sub_op = operation "dxgml_op.subtract"(%input, %mean_result : !dxgml_pattern.value, !dxgml_pattern.value) -> (%input_type : !dxgml_pattern.type)
        %centered = result 0 of %sub_op
        
        // Squared: centered^2
        %pow_op = operation "dxgml_op.pow"(%centered, %power_const : !dxgml_pattern.value, !dxgml_pattern.value) -> (%input_type : !dxgml_pattern.type)
        %squared = result 0 of %pow_op

        // Variance: reduce(squared, average)
        %var_op = operation "dxgml_op.reduce"(%squared : !dxgml_pattern.value) -> (%reduced_type : !dxgml_pattern.type)
        %variance = result 0 of %var_op
        
        // Variance + epsilon
        %add_eps_op = operation "dxgml_op.add"(%variance, %epsilon_const : !dxgml_pattern.value, !dxgml_pattern.value) -> (%reduced_type : !dxgml_pattern.type)
        %var_eps = result 0 of %add_eps_op

        // Stddev: sqrt(variance + epsilon)
        %sqrt_op = operation "dxgml_op.sqrt"(%var_eps : !dxgml_pattern.value) -> (%reduced_type : !dxgml_pattern.type)
        %stddev = result 0 of %sqrt_op
        
        // Normalized: centered / stddev
        %div_op = operation "dxgml_op.divide"(%centered, %stddev : !dxgml_pattern.value, !dxgml_pattern.value) -> (%input_type : !dxgml_pattern.type)
        %normalized = result 0 of %div_op

        // Scaled: normalized * gamma
        %scale_op = operation "dxgml_op.multiply"(%normalized, %gamma : !dxgml_pattern.value, !dxgml_pattern.value) -> (%input_type : !dxgml_pattern.type)
        %scaled = result 0 of %scale_op

        // Output: scaled + beta
        %shift_op = operation "dxgml_op.add"(%scaled, %beta : !dxgml_pattern.value, !dxgml_pattern.value) -> (%input_type : !dxgml_pattern.type)

        rewrite %shift_op {
            %axes_attr = attribute = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>
            %use_mean_attr = attribute = #dxgml.bool<true>
            %use_variance_attr = attribute = #dxgml.bool<true>
            %mvn_op = operation "dxgml_op.mean_variance_normalization"(%input, %gamma, %beta : !dxgml_pattern.value, !dxgml_pattern.value, !dxgml_pattern.value) {
                "axes" = %axes_attr,
                "use_mean" = %use_mean_attr,
                "use_variance" = %use_variance_attr
            } -> (%input_type : !dxgml_pattern.type)
            replace %shift_op with %mvn_op
        }
    }


    dxgml_pattern.pattern @conv_gelu_add : benefit(10) {
        %result_types = types
        %inputs = operands 
        
        // Match the base op
        %conv_op = operation "dxgml_op.convolution" (%inputs : !dxgml_pattern.range<value>) -> (%result_types : !dxgml_pattern.range<type>)
        
        rewrite {

            %subgraph_rewrite_description = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<
                foreign_config = {jitFunction = "FusedConvElementWise"} // Arbitrary IHV key-value pairs
                clusterFusionDescriptions = [
                    #dxgml_subgraph_pattern.cluster_fusion_desc<
                        operatorIndex = 0, // Index of conv_op within parameters to dxgml_subgraph_pattern_rewriter
                        resultIndex = 0,  // Index of the result from this operator where the cluster can be fused
                        supportedTypes = ["dxgml_op.gelu", "dxgml_op.add", "dxgml_op.multiply", "dxgml_op.transpose", "dxgml_op.reshape", "dxgml_op.split"], // Types allowed within the cluster
                        fusionBenefit = 10  // Same benefit as the fusion pattern in this example
                    >
                ]
            >

            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%conv_op, %subgraph_rewrite_description : !dxgml_pattern.operation, !dxgml_pattern.attribute)
        }
    }


    dxgml_pattern.pattern @matmul_gelu_add : benefit(10) {
        %result_types = types
        %inputs = operands 
        
        // Match the base op
        %gemm_op = operation "dxgml_op.gemm" (%inputs : !dxgml_pattern.range<value>) -> (%result_types : !dxgml_pattern.range<type>)
        
        rewrite {

            %subgraph_rewrite_description = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<
                foreign_config = {jitFunction = "FusedGemmElementWise"} // Arbitrary IHV key-value pairs
                clusterFusionDescriptions = [
                    #dxgml_subgraph_pattern.cluster_fusion_desc<
                        operatorIndex = 0, // Index of conv_op within parameters to dxgml_subgraph_pattern_rewriter
                        resultIndex = 0,  // Index of the result from this operator where the cluster can be fused
                        supportedTypes = ["dxgml_op.gelu", "dxgml_op.add", "dxgml_op.multiply", "dxgml_op.transpose", "dxgml_op.reshape", "dxgml_op.split"], // Types allowed within the cluster
                        fusionBenefit = 10  // Same benefit as the fusion pattern in this example
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
            "dxgml_op.gelu"
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

    dxgml_pattern.pattern @multihead_attention_to_subgraph : benefit(1) {
        %result_types = types
        %inputs = operands

        %op = operation "dxgml_op.multihead_attention" (%inputs : !dxgml_pattern.range<value>) -> (%result_types : !dxgml_pattern.range<type>)

        %config = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<
            foreign_config = {jitFunction = "MultiheadAttention"}
        >

        rewrite {
            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%op, %config : !dxgml_pattern.operation, !dxgml_pattern.attribute)
        }
    }
}