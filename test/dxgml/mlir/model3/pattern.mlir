// RUN: %run_test %rtml-opt %s | %FileCheck %s


module {
    // CHECK-LABEL: @conv_clip_add
    dxgml_pattern.pattern @conv_clip_add : benefit(10) {
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
                        supportedTypes = ["dxgml_op.clip", "dxgml_op.add"], // Types allowed within the cluster
                        fusionBenefit = 10  // Same benefit as the fusion pattern in this example
                    >
                ]
            >

            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%conv_op, %subgraph_rewrite_description : !dxgml_pattern.operation, !dxgml_pattern.attribute)
        }
    }

    dxgml_pattern.pattern @concat_basic : benefit(10) {
        %result_types = types
        %input0 = operand
        %input1 = operand
        %input2 = operand
        %input3 = operand
        %input4 = operand
        %input5 = operand
        
        // Match the base op
        %concat_op = operation "dxgml_op.concat" (%input0, %input1, %input2, %input3, %input4, %input5 : !dxgml_pattern.value, !dxgml_pattern.value, !dxgml_pattern.value, !dxgml_pattern.value, !dxgml_pattern.value, !dxgml_pattern.value) -> (%result_types : !dxgml_pattern.range<type>)
        
        rewrite {

            %subgraph_rewrite_description = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<
                foreign_config = {jitFunction = "Concat"} // Arbitrary IHV key-value pairs
                clusterFusionDescriptions = [
                    #dxgml_subgraph_pattern.cluster_fusion_desc<
                        operatorIndex = 0, // Index of op within parameters to dxgml_subgraph_pattern_rewriter
                        resultIndex = 0,  // Index of the result from this operator where the cluster can be fused
                        supportedTypes = ["dxgml_op.softmax", "dxgml_op.slice", "dxgml_op.multiply", "dxgml_op.add", "dxgml_op.exp", "dxgml_op.concat", "dxgml_op.divide", "dxgml_op.subtract"], // Types allowed within the cluster
                        fusionBenefit = 10  // Same benefit as the fusion pattern in this example
                    >,
                    #dxgml_subgraph_pattern.cluster_fusion_desc<
                        operatorIndex = 0, // Index of op within parameters to dxgml_subgraph_pattern_rewriter
                        inputIndex = 0,  // Index of the result from this operator where the cluster can be fused
                        supportedTypes = ["dxgml_op.transpose", "dxgml_op.reshape"], // Types allowed within the cluster
                        fusionBenefit = 10  // Same benefit as the fusion pattern in this example
                    >,
                    #dxgml_subgraph_pattern.cluster_fusion_desc<
                        operatorIndex = 0, // Index of op within parameters to dxgml_subgraph_pattern_rewriter
                        inputIndex = 1,  // Index of the result from this operator where the cluster can be fused
                        supportedTypes = ["dxgml_op.transpose", "dxgml_op.reshape"], // Types allowed within the cluster
                        fusionBenefit = 10  // Same benefit as the fusion pattern in this example
                    >,
                    #dxgml_subgraph_pattern.cluster_fusion_desc<
                        operatorIndex = 0, // Index of op within parameters to dxgml_subgraph_pattern_rewriter
                        inputIndex = 2,  // Index of the result from this operator where the cluster can be fused
                        supportedTypes = ["dxgml_op.transpose", "dxgml_op.reshape"], // Types allowed within the cluster
                        fusionBenefit = 10  // Same benefit as the fusion pattern in this example
                    >,
                    #dxgml_subgraph_pattern.cluster_fusion_desc<
                        operatorIndex = 0, // Index of op within parameters to dxgml_subgraph_pattern_rewriter
                        inputIndex = 3,  // Index of the result from this operator where the cluster can be fused
                        supportedTypes = ["dxgml_op.transpose", "dxgml_op.reshape"], // Types allowed within the cluster
                        fusionBenefit = 10  // Same benefit as the fusion pattern in this example
                    >,
                    #dxgml_subgraph_pattern.cluster_fusion_desc<
                        operatorIndex = 0, // Index of op within parameters to dxgml_subgraph_pattern_rewriter
                        inputIndex = 4,  // Index of the result from this operator where the cluster can be fused
                        supportedTypes = ["dxgml_op.transpose", "dxgml_op.reshape"], // Types allowed within the cluster
                        fusionBenefit = 10  // Same benefit as the fusion pattern in this example
                    >,
                    #dxgml_subgraph_pattern.cluster_fusion_desc<
                        operatorIndex = 0, // Index of op within parameters to dxgml_subgraph_pattern_rewriter
                        inputIndex = 5,  // Index of the result from this operator where the cluster can be fused
                        supportedTypes = ["dxgml_op.transpose", "dxgml_op.reshape"], // Types allowed within the cluster
                        fusionBenefit = 10  // Same benefit as the fusion pattern in this example
                    >
                ]
            >

            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%concat_op, %subgraph_rewrite_description : !dxgml_pattern.operation, !dxgml_pattern.attribute)
        }
    }

    dxgml_pattern.pattern @transpose_to_subgraph : benefit(10) {
        %type = type
        %input = operand

        %transpose_op = operation "dxgml_op.transpose" (%input : !dxgml_pattern.value) -> (%type : !dxgml_pattern.type)

        %config = attribute = #dxgml_subgraph_pattern.subgraph_rewrite_desc<
            foreign_config = {jitFunction = "Unary"}
        >

        rewrite {
            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%transpose_op, %config : !dxgml_pattern.operation, !dxgml_pattern.attribute)
        }
    }
}
