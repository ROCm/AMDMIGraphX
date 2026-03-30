// RUN: %run_test %rtml-opt %s | %FileCheck %s


module {
    // CHECK-LABEL: @conv_relu_add
    dxgml_pattern.pattern @conv_relu_add : benefit(10) {
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
                        supportedTypes = ["dxgml_op.relu", "dxgml_op.add", "dxgml_op.depth_to_space"], // Types allowed within the cluster
                        fusionBenefit = 10  // Same benefit as the fusion pattern in this example
                    >
                ]
            >

            apply_native_rewrite "dxgml_subgraph_pattern_rewriter" (%conv_op, %subgraph_rewrite_description : !dxgml_pattern.operation, !dxgml_pattern.attribute)
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
