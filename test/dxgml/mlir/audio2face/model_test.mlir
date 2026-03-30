dxgml.module @audio2face {
dxgml.entry_point @main_graph(%arg0: !dxgml.tensor<1x10000x!dxgml.float16>) -> !dxgml.tensor<1x6x!dxgml.float16> {
%0 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.0.conv.weight : !dxgml.tensor<512x1x10x!dxgml.float16>>)
%1 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.0.conv.bias : !dxgml.tensor<512x!dxgml.float16>>)
%2 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.0.layer_norm.weight : !dxgml.tensor<512x!dxgml.float16>>)
%3 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.0.layer_norm.bias : !dxgml.tensor<512x!dxgml.float16>>)
%4 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.1.conv.weight : !dxgml.tensor<512x512x3x!dxgml.float16>>)
%5 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.1.conv.bias : !dxgml.tensor<512x!dxgml.float16>>)
%6 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.1.layer_norm.weight : !dxgml.tensor<512x!dxgml.float16>>)
%7 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.1.layer_norm.bias : !dxgml.tensor<512x!dxgml.float16>>)
%8 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.2.conv.weight : !dxgml.tensor<512x512x3x!dxgml.float16>>)
%9 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.2.conv.bias : !dxgml.tensor<512x!dxgml.float16>>)
%10 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.2.layer_norm.weight : !dxgml.tensor<512x!dxgml.float16>>)
%11 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.2.layer_norm.bias : !dxgml.tensor<512x!dxgml.float16>>)
%12 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.3.conv.weight : !dxgml.tensor<512x512x3x!dxgml.float16>>)
%13 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.3.conv.bias : !dxgml.tensor<512x!dxgml.float16>>)
%14 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.3.layer_norm.weight : !dxgml.tensor<512x!dxgml.float16>>)
%15 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.3.layer_norm.bias : !dxgml.tensor<512x!dxgml.float16>>)
%16 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.4.conv.weight : !dxgml.tensor<512x512x3x!dxgml.float16>>)
%17 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.4.conv.bias : !dxgml.tensor<512x!dxgml.float16>>)
%18 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.4.layer_norm.weight : !dxgml.tensor<512x!dxgml.float16>>)
%19 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.4.layer_norm.bias : !dxgml.tensor<512x!dxgml.float16>>)
%20 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.5.conv.weight : !dxgml.tensor<512x512x2x!dxgml.float16>>)
%21 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.5.conv.bias : !dxgml.tensor<512x!dxgml.float16>>)
%22 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.5.layer_norm.weight : !dxgml.tensor<512x!dxgml.float16>>)
%23 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.5.layer_norm.bias : !dxgml.tensor<512x!dxgml.float16>>)
%24 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.6.conv.weight : !dxgml.tensor<512x512x2x!dxgml.float16>>)
%25 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.6.conv.bias : !dxgml.tensor<512x!dxgml.float16>>)
%26 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.6.layer_norm.weight : !dxgml.tensor<512x!dxgml.float16>>)
%27 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_extractor.conv_layers.6.layer_norm.bias : !dxgml.tensor<512x!dxgml.float16>>)
%28 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_projection.layer_norm.weight : !dxgml.tensor<512x!dxgml.float16>>)
%29 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_projection.layer_norm.bias : !dxgml.tensor<512x!dxgml.float16>>)
%30 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.feature_projection.projection.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%31 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.pos_conv_embed.conv.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%32 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%33 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%34 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.0.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%35 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.0.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%36 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.0.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%37 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.0.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%38 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.0.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%39 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.0.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%40 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.0.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%41 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.0.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%42 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.0.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%43 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.0.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%44 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.1.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%45 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.1.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%46 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.1.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%47 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.1.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%48 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.1.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%49 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.1.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%50 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.1.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%51 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.1.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%52 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.1.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%53 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.1.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%54 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.2.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%55 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.2.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%56 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.2.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%57 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.2.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%58 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.2.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%59 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.2.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%60 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.2.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%61 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.2.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%62 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.2.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%63 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.2.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%64 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.3.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%65 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.3.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%66 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.3.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%67 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.3.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%68 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.3.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%69 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.3.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%70 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.3.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%71 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.3.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%72 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.3.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%73 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.3.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%74 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.4.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%75 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.4.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%76 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.4.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%77 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.4.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%78 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.4.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%79 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.4.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%80 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.4.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%81 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.4.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%82 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.4.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%83 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.4.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%84 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.5.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%85 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.5.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%86 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.5.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%87 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.5.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%88 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.5.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%89 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.5.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%90 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.5.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%91 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.5.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%92 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.5.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%93 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.5.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%94 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.6.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%95 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.6.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%96 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.6.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%97 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.6.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%98 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.6.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%99 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.6.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%100 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.6.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%101 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.6.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%102 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.6.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%103 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.6.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%104 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.7.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%105 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.7.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%106 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.7.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%107 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.7.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%108 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.7.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%109 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.7.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%110 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.7.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%111 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.7.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%112 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.7.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%113 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.7.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%114 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.8.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%115 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.8.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%116 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.8.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%117 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.8.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%118 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.8.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%119 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.8.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%120 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.8.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%121 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.8.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%122 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.8.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%123 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.8.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%124 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.9.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%125 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.9.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%126 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.9.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%127 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.9.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%128 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.9.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%129 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.9.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%130 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.9.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%131 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.9.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%132 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.9.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%133 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.9.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%134 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.10.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%135 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.10.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%136 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.10.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%137 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.10.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%138 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.10.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%139 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.10.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%140 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.10.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%141 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.10.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%142 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.10.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%143 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.10.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%144 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.11.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%145 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.11.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%146 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.11.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%147 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.11.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%148 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.11.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%149 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.11.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%150 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.11.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%151 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.11.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%152 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.11.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%153 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.11.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%154 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.12.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%155 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.12.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%156 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.12.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%157 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.12.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%158 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.12.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%159 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.12.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%160 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.12.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%161 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.12.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%162 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.12.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%163 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.12.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%164 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.13.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%165 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.13.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%166 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.13.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%167 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.13.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%168 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.13.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%169 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.13.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%170 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.13.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%171 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.13.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%172 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.13.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%173 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.13.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%174 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.14.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%175 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.14.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%176 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.14.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%177 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.14.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%178 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.14.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%179 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.14.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%180 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.14.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%181 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.14.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%182 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.14.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%183 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.14.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%184 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.15.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%185 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.15.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%186 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.15.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%187 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.15.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%188 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.15.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%189 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.15.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%190 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.15.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%191 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.15.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%192 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.15.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%193 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.15.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%194 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.16.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%195 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.16.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%196 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.16.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%197 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.16.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%198 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.16.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%199 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.16.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%200 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.16.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%201 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.16.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%202 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.16.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%203 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.16.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%204 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.17.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%205 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.17.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%206 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.17.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%207 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.17.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%208 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.17.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%209 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.17.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%210 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.17.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%211 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.17.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%212 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.17.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%213 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.17.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%214 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.18.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%215 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.18.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%216 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.18.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%217 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.18.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%218 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.18.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%219 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.18.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%220 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.18.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%221 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.18.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%222 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.18.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%223 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.18.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%224 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.19.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%225 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.19.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%226 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.19.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%227 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.19.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%228 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.19.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%229 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.19.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%230 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.19.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%231 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.19.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%232 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.19.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%233 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.19.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%234 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.20.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%235 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.20.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%236 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.20.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%237 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.20.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%238 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.20.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%239 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.20.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%240 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.20.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%241 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.20.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%242 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.20.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%243 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.20.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%244 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.21.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%245 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.21.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%246 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.21.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%247 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.21.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%248 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.21.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%249 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.21.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%250 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.21.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%251 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.21.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%252 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.21.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%253 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.21.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%254 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.22.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%255 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.22.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%256 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.22.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%257 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.22.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%258 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.22.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%259 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.22.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%260 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.22.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%261 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.22.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%262 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.22.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%263 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.22.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%264 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.23.attention.k_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%265 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.23.attention.v_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%266 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.23.attention.q_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%267 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.23.attention.out_proj.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%268 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.23.layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%269 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.23.layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%270 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.23.feed_forward.intermediate_dense.bias : !dxgml.tensor<4096x!dxgml.float16>>)
%271 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.23.feed_forward.output_dense.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%272 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.23.final_layer_norm.weight : !dxgml.tensor<1024x!dxgml.float16>>)
%273 = dxgml_op.constant(#dxgml.constant_resource<_model.wav2vec2.encoder.layers.23.final_layer_norm.bias : !dxgml.tensor<1024x!dxgml.float16>>)
%274 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3400 : !dxgml.tensor<512x1024x!dxgml.float16>>)
%275 = dxgml_op.constant(#dxgml.constant_resource<_onnx__Conv_3403 : !dxgml.tensor<1024x64x128x!dxgml.float16>>)
%276 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3428 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%277 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3429 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%278 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3430 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%279 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3451 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%280 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3452 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%281 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3453 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%282 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3474 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%283 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3475 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%284 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3476 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%285 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3497 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%286 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3498 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%287 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3499 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%288 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3520 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%289 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3521 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%290 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3522 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%291 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3543 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%292 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3544 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%293 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3545 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%294 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3566 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%295 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3567 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%296 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3568 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%297 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3589 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%298 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3590 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%299 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3591 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%300 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3612 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%301 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3613 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%302 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3614 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%303 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3635 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%304 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3636 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%305 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3637 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%306 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3658 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%307 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3659 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%308 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3660 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%309 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3681 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%310 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3682 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%311 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3683 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%312 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3704 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%313 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3705 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%314 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3706 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%315 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3727 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%316 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3728 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%317 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3729 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%318 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3750 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%319 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3751 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%320 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3752 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%321 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3773 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%322 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3774 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%323 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3775 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%324 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3796 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%325 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3797 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%326 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3798 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%327 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3819 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%328 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3820 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%329 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3821 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%330 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3842 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%331 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3843 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%332 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3844 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%333 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3865 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%334 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3866 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%335 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3867 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%336 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3888 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%337 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3889 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%338 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3890 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%339 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3911 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%340 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3912 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%341 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3913 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%342 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3934 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%343 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3935 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%344 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3936 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%345 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3957 : !dxgml.tensor<1024x1024x!dxgml.float16>>)
%346 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3958 : !dxgml.tensor<1024x4096x!dxgml.float16>>)
%347 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3959 : !dxgml.tensor<4096x1024x!dxgml.float16>>)
%348 = dxgml_op.constant(#dxgml.constant_resource<_onnx__MatMul_3960 : !dxgml.tensor<1024x6x!dxgml.float16>>)
%349 = dxgml_op.constant(#dxgml.constant_resource<__Constant_1_output_0 : !dxgml.tensor<1x!dxgml.float16>>)
%350 = dxgml_op.constant(#dxgml.constant_resource<__Constant_2_output_0 : !dxgml.tensor<1x!dxgml.float16>>)
%351 = dxgml_op.constant(#dxgml.constant_resource<__model_wav2vec2_feature_extractor_conv_layers.0_layer_norm_Constant_output_0 : !dxgml.tensor<1x!dxgml.float16>>)
%352 = dxgml_op.constant(#dxgml.constant_resource<__model_wav2vec2_feature_extractor_conv_layers.0_layer_norm_Constant_1_output_0 : !dxgml.tensor<1x!dxgml.float16>>)
%353 = dxgml_op.constant(#dxgml.constant_resource<__model_wav2vec2_feature_extractor_conv_layers.0_activation_Constant_output_0 : !dxgml.tensor<1x!dxgml.float16>>)
%354 = dxgml_op.constant(#dxgml.constant_resource<__model_wav2vec2_feature_extractor_conv_layers.0_activation_Constant_2_output_0 : !dxgml.tensor<1x!dxgml.float16>>)
%355 = dxgml_op.constant(#dxgml.constant_resource<__model_wav2vec2_encoder_layers.0_attention_Constant_2_output_0 : !dxgml.tensor<1x!dxgml.float16>>)
%356 = dxgml_op.constant(#dxgml.constant_resource<__Cast_output_0 : !dxgml.tensor<1x!dxgml.float16>>)
%357 = dxgml_op.constant(#dxgml.constant_resource<__Sub_2_output_0 : !dxgml.tensor<1x!dxgml.float16>>)
%358 = dxgml_op.constant(#dxgml.constant_resource<_v_2203 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%359 = dxgml_op.constant(#dxgml.constant_resource<_v_2186 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%360 = dxgml_op.constant(#dxgml.constant_resource<_v_2173 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%361 = dxgml_op.constant(#dxgml.constant_resource<_v_2160 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%362 = dxgml_op.constant(#dxgml.constant_resource<_v_2147 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%363 = dxgml_op.constant(#dxgml.constant_resource<_v_2134 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%364 = dxgml_op.constant(#dxgml.constant_resource<_v_2121 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%365 = dxgml_op.constant(#dxgml.constant_resource<_v_2108 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%366 = dxgml_op.constant(#dxgml.constant_resource<_v_2095 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%367 = dxgml_op.constant(#dxgml.constant_resource<_v_2082 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%368 = dxgml_op.constant(#dxgml.constant_resource<_v_2069 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%369 = dxgml_op.constant(#dxgml.constant_resource<_v_2056 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%370 = dxgml_op.constant(#dxgml.constant_resource<_v_2043 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%371 = dxgml_op.constant(#dxgml.constant_resource<_v_2030 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%372 = dxgml_op.constant(#dxgml.constant_resource<_v_2017 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%373 = dxgml_op.constant(#dxgml.constant_resource<_v_2004 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%374 = dxgml_op.constant(#dxgml.constant_resource<_v_1991 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%375 = dxgml_op.constant(#dxgml.constant_resource<_v_1978 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%376 = dxgml_op.constant(#dxgml.constant_resource<_v_1965 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%377 = dxgml_op.constant(#dxgml.constant_resource<_v_1952 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%378 = dxgml_op.constant(#dxgml.constant_resource<_v_1939 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%379 = dxgml_op.constant(#dxgml.constant_resource<_v_1926 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%380 = dxgml_op.constant(#dxgml.constant_resource<_v_1913 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%381 = dxgml_op.constant(#dxgml.constant_resource<_v_1896 : !dxgml.tensor<1024x3072x!dxgml.float16>>)
%382 = dxgml_op.reduce (%arg0) {axes = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x10000x!dxgml.float16>) -> !dxgml.tensor<1x1x!dxgml.float16>
%383 = dxgml_op.subtract (%arg0, %382) : (!dxgml.tensor<1x10000x!dxgml.float16>, !dxgml.tensor<1x1x!dxgml.float16>) -> !dxgml.tensor<1x10000x!dxgml.float16>
%384 = dxgml_op.multiply (%383, %383) : (!dxgml.tensor<1x10000x!dxgml.float16>, !dxgml.tensor<1x10000x!dxgml.float16>) -> !dxgml.tensor<1x10000x!dxgml.float16>
%385 = dxgml_op.reduce (%384) {axes = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x10000x!dxgml.float16>) -> !dxgml.tensor<1x1x!dxgml.float16>
%386 = dxgml_op.multiply (%385, %356) : (!dxgml.tensor<1x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x1x!dxgml.float16>
%387 = dxgml_op.divide (%386, %357) : (!dxgml.tensor<1x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x1x!dxgml.float16>
%388 = dxgml_op.sqrt (%387) : (!dxgml.tensor<1x1x!dxgml.float16>) -> !dxgml.tensor<1x1x!dxgml.float16>
%389 = dxgml_op.add (%388, %350) : (!dxgml.tensor<1x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x1x!dxgml.float16>
%390 = dxgml_op.divide (%383, %389) : (!dxgml.tensor<1x10000x!dxgml.float16>, !dxgml.tensor<1x1x!dxgml.float16>) -> !dxgml.tensor<1x10000x!dxgml.float16>
%391 = dxgml_op.reshape (%390) : (!dxgml.tensor<1x10000x!dxgml.float16>) -> !dxgml.tensor<1x2x5000x!dxgml.float16>
%392 = dxgml_op.slice (%391) {axes = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>, ends = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, input_window_offsets = [0, 0, 0], input_window_sizes = [1, 0, 5000], input_window_strides = [1, 1, 1], starts = #dxgml.dense_integer_elements<[0]> : !dxgml.tensor<1x!dxgml.int64>, steps = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>} : (!dxgml.tensor<1x2x5000x!dxgml.float16>) -> !dxgml.tensor<1x1x5000x!dxgml.float16>
%393 = dxgml_op.slice (%391) {axes = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>, ends = #dxgml.dense_integer_elements<[9223372036854775807]> : !dxgml.tensor<1x!dxgml.int64>, input_window_offsets = [0, 1, 0], input_window_sizes = [1, 1, 5000], input_window_strides = [1, 1, 1], starts = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>, steps = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>} : (!dxgml.tensor<1x2x5000x!dxgml.float16>) -> !dxgml.tensor<1x1x5000x!dxgml.float16>
%394 = dxgml_op.concat (%392, %393) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x1x5000x!dxgml.float16>, !dxgml.tensor<1x1x5000x!dxgml.float16>) -> !dxgml.tensor<1x1x10000x!dxgml.float16>
%395 = dxgml_op.reshape (%394) : (!dxgml.tensor<1x1x10000x!dxgml.float16>) -> !dxgml.tensor<1x10000x!dxgml.float16>
%396 = dxgml_op.reshape (%395) : (!dxgml.tensor<1x10000x!dxgml.float16>) -> !dxgml.tensor<1x1x10000x!dxgml.float16>
%397 = dxgml_op.convolution (%396, %0, %1) {dilations = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>, end_padding = #dxgml.dense_integer_elements<[0]> : !dxgml.tensor<1x!dxgml.int64>, group_count = #dxgml.integer<1 : !dxgml.int64>, start_padding = #dxgml.dense_integer_elements<[0]> : !dxgml.tensor<1x!dxgml.int64>, strides = #dxgml.dense_integer_elements<[5]> : !dxgml.tensor<1x!dxgml.int64>} : (!dxgml.tensor<1x1x10000x!dxgml.float16>, !dxgml.tensor<512x1x10x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x512x1999x!dxgml.float16>
%398 = dxgml_op.transpose (%397) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<1x512x1999x!dxgml.float16>) -> !dxgml.tensor<1x1999x512x!dxgml.float16>
%399 = dxgml_op.reduce (%398) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x1999x512x!dxgml.float16>) -> !dxgml.tensor<1x1999x1x!dxgml.float16>
%400 = dxgml_op.subtract (%398, %399) : (!dxgml.tensor<1x1999x512x!dxgml.float16>, !dxgml.tensor<1x1999x1x!dxgml.float16>) -> !dxgml.tensor<1x1999x512x!dxgml.float16>
%401 = dxgml_op.pow (%400, %351) : (!dxgml.tensor<1x1999x512x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x1999x512x!dxgml.float16>
%402 = dxgml_op.reduce (%401) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x1999x512x!dxgml.float16>) -> !dxgml.tensor<1x1999x1x!dxgml.float16>
%403 = dxgml_op.add (%402, %352) : (!dxgml.tensor<1x1999x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x1999x1x!dxgml.float16>
%404 = dxgml_op.sqrt (%403) : (!dxgml.tensor<1x1999x1x!dxgml.float16>) -> !dxgml.tensor<1x1999x1x!dxgml.float16>
%405 = dxgml_op.divide (%400, %404) : (!dxgml.tensor<1x1999x512x!dxgml.float16>, !dxgml.tensor<1x1999x1x!dxgml.float16>) -> !dxgml.tensor<1x1999x512x!dxgml.float16>
%406 = dxgml_op.multiply (%405, %2) : (!dxgml.tensor<1x1999x512x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x1999x512x!dxgml.float16>
%407 = dxgml_op.add (%406, %3) : (!dxgml.tensor<1x1999x512x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x1999x512x!dxgml.float16>
%408 = dxgml_op.transpose (%407) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<1x1999x512x!dxgml.float16>) -> !dxgml.tensor<1x512x1999x!dxgml.float16>
%409 = dxgml_op.divide (%408, %353) : (!dxgml.tensor<1x512x1999x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x1999x!dxgml.float16>
%410 = dxgml_op.erf (%409) : (!dxgml.tensor<1x512x1999x!dxgml.float16>) -> !dxgml.tensor<1x512x1999x!dxgml.float16>
%411 = dxgml_op.add (%410, %349) : (!dxgml.tensor<1x512x1999x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x1999x!dxgml.float16>
%412 = dxgml_op.multiply (%408, %411) : (!dxgml.tensor<1x512x1999x!dxgml.float16>, !dxgml.tensor<1x512x1999x!dxgml.float16>) -> !dxgml.tensor<1x512x1999x!dxgml.float16>
%413 = dxgml_op.multiply (%412, %354) : (!dxgml.tensor<1x512x1999x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x1999x!dxgml.float16>
%414 = dxgml_op.convolution (%413, %4, %5) {dilations = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>, end_padding = #dxgml.dense_integer_elements<[0]> : !dxgml.tensor<1x!dxgml.int64>, group_count = #dxgml.integer<1 : !dxgml.int64>, start_padding = #dxgml.dense_integer_elements<[0]> : !dxgml.tensor<1x!dxgml.int64>, strides = #dxgml.dense_integer_elements<[2]> : !dxgml.tensor<1x!dxgml.int64>} : (!dxgml.tensor<1x512x1999x!dxgml.float16>, !dxgml.tensor<512x512x3x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x512x999x!dxgml.float16>
%415 = dxgml_op.transpose (%414) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<1x512x999x!dxgml.float16>) -> !dxgml.tensor<1x999x512x!dxgml.float16>
%416 = dxgml_op.reduce (%415) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x999x512x!dxgml.float16>) -> !dxgml.tensor<1x999x1x!dxgml.float16>
%417 = dxgml_op.subtract (%415, %416) : (!dxgml.tensor<1x999x512x!dxgml.float16>, !dxgml.tensor<1x999x1x!dxgml.float16>) -> !dxgml.tensor<1x999x512x!dxgml.float16>
%418 = dxgml_op.pow (%417, %351) : (!dxgml.tensor<1x999x512x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x999x512x!dxgml.float16>
%419 = dxgml_op.reduce (%418) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x999x512x!dxgml.float16>) -> !dxgml.tensor<1x999x1x!dxgml.float16>
%420 = dxgml_op.add (%419, %352) : (!dxgml.tensor<1x999x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x999x1x!dxgml.float16>
%421 = dxgml_op.sqrt (%420) : (!dxgml.tensor<1x999x1x!dxgml.float16>) -> !dxgml.tensor<1x999x1x!dxgml.float16>
%422 = dxgml_op.divide (%417, %421) : (!dxgml.tensor<1x999x512x!dxgml.float16>, !dxgml.tensor<1x999x1x!dxgml.float16>) -> !dxgml.tensor<1x999x512x!dxgml.float16>
%423 = dxgml_op.multiply (%422, %6) : (!dxgml.tensor<1x999x512x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x999x512x!dxgml.float16>
%424 = dxgml_op.add (%423, %7) : (!dxgml.tensor<1x999x512x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x999x512x!dxgml.float16>
%425 = dxgml_op.transpose (%424) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<1x999x512x!dxgml.float16>) -> !dxgml.tensor<1x512x999x!dxgml.float16>
%426 = dxgml_op.divide (%425, %353) : (!dxgml.tensor<1x512x999x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x999x!dxgml.float16>
%427 = dxgml_op.erf (%426) : (!dxgml.tensor<1x512x999x!dxgml.float16>) -> !dxgml.tensor<1x512x999x!dxgml.float16>
%428 = dxgml_op.add (%427, %349) : (!dxgml.tensor<1x512x999x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x999x!dxgml.float16>
%429 = dxgml_op.multiply (%425, %428) : (!dxgml.tensor<1x512x999x!dxgml.float16>, !dxgml.tensor<1x512x999x!dxgml.float16>) -> !dxgml.tensor<1x512x999x!dxgml.float16>
%430 = dxgml_op.multiply (%429, %354) : (!dxgml.tensor<1x512x999x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x999x!dxgml.float16>
%431 = dxgml_op.convolution (%430, %8, %9) {dilations = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>, end_padding = #dxgml.dense_integer_elements<[0]> : !dxgml.tensor<1x!dxgml.int64>, group_count = #dxgml.integer<1 : !dxgml.int64>, start_padding = #dxgml.dense_integer_elements<[0]> : !dxgml.tensor<1x!dxgml.int64>, strides = #dxgml.dense_integer_elements<[2]> : !dxgml.tensor<1x!dxgml.int64>} : (!dxgml.tensor<1x512x999x!dxgml.float16>, !dxgml.tensor<512x512x3x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x512x499x!dxgml.float16>
%432 = dxgml_op.transpose (%431) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<1x512x499x!dxgml.float16>) -> !dxgml.tensor<1x499x512x!dxgml.float16>
%433 = dxgml_op.reduce (%432) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x499x512x!dxgml.float16>) -> !dxgml.tensor<1x499x1x!dxgml.float16>
%434 = dxgml_op.subtract (%432, %433) : (!dxgml.tensor<1x499x512x!dxgml.float16>, !dxgml.tensor<1x499x1x!dxgml.float16>) -> !dxgml.tensor<1x499x512x!dxgml.float16>
%435 = dxgml_op.pow (%434, %351) : (!dxgml.tensor<1x499x512x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x499x512x!dxgml.float16>
%436 = dxgml_op.reduce (%435) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x499x512x!dxgml.float16>) -> !dxgml.tensor<1x499x1x!dxgml.float16>
%437 = dxgml_op.add (%436, %352) : (!dxgml.tensor<1x499x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x499x1x!dxgml.float16>
%438 = dxgml_op.sqrt (%437) : (!dxgml.tensor<1x499x1x!dxgml.float16>) -> !dxgml.tensor<1x499x1x!dxgml.float16>
%439 = dxgml_op.divide (%434, %438) : (!dxgml.tensor<1x499x512x!dxgml.float16>, !dxgml.tensor<1x499x1x!dxgml.float16>) -> !dxgml.tensor<1x499x512x!dxgml.float16>
%440 = dxgml_op.multiply (%439, %10) : (!dxgml.tensor<1x499x512x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x499x512x!dxgml.float16>
%441 = dxgml_op.add (%440, %11) : (!dxgml.tensor<1x499x512x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x499x512x!dxgml.float16>
%442 = dxgml_op.transpose (%441) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<1x499x512x!dxgml.float16>) -> !dxgml.tensor<1x512x499x!dxgml.float16>
%443 = dxgml_op.divide (%442, %353) : (!dxgml.tensor<1x512x499x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x499x!dxgml.float16>
%444 = dxgml_op.erf (%443) : (!dxgml.tensor<1x512x499x!dxgml.float16>) -> !dxgml.tensor<1x512x499x!dxgml.float16>
%445 = dxgml_op.add (%444, %349) : (!dxgml.tensor<1x512x499x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x499x!dxgml.float16>
%446 = dxgml_op.multiply (%442, %445) : (!dxgml.tensor<1x512x499x!dxgml.float16>, !dxgml.tensor<1x512x499x!dxgml.float16>) -> !dxgml.tensor<1x512x499x!dxgml.float16>
%447 = dxgml_op.multiply (%446, %354) : (!dxgml.tensor<1x512x499x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x499x!dxgml.float16>
%448 = dxgml_op.convolution (%447, %12, %13) {dilations = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>, end_padding = #dxgml.dense_integer_elements<[0]> : !dxgml.tensor<1x!dxgml.int64>, group_count = #dxgml.integer<1 : !dxgml.int64>, start_padding = #dxgml.dense_integer_elements<[0]> : !dxgml.tensor<1x!dxgml.int64>, strides = #dxgml.dense_integer_elements<[2]> : !dxgml.tensor<1x!dxgml.int64>} : (!dxgml.tensor<1x512x499x!dxgml.float16>, !dxgml.tensor<512x512x3x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x512x249x!dxgml.float16>
%449 = dxgml_op.transpose (%448) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<1x512x249x!dxgml.float16>) -> !dxgml.tensor<1x249x512x!dxgml.float16>
%450 = dxgml_op.reduce (%449) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x249x512x!dxgml.float16>) -> !dxgml.tensor<1x249x1x!dxgml.float16>
%451 = dxgml_op.subtract (%449, %450) : (!dxgml.tensor<1x249x512x!dxgml.float16>, !dxgml.tensor<1x249x1x!dxgml.float16>) -> !dxgml.tensor<1x249x512x!dxgml.float16>
%452 = dxgml_op.pow (%451, %351) : (!dxgml.tensor<1x249x512x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x249x512x!dxgml.float16>
%453 = dxgml_op.reduce (%452) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x249x512x!dxgml.float16>) -> !dxgml.tensor<1x249x1x!dxgml.float16>
%454 = dxgml_op.add (%453, %352) : (!dxgml.tensor<1x249x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x249x1x!dxgml.float16>
%455 = dxgml_op.sqrt (%454) : (!dxgml.tensor<1x249x1x!dxgml.float16>) -> !dxgml.tensor<1x249x1x!dxgml.float16>
%456 = dxgml_op.divide (%451, %455) : (!dxgml.tensor<1x249x512x!dxgml.float16>, !dxgml.tensor<1x249x1x!dxgml.float16>) -> !dxgml.tensor<1x249x512x!dxgml.float16>
%457 = dxgml_op.multiply (%456, %14) : (!dxgml.tensor<1x249x512x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x249x512x!dxgml.float16>
%458 = dxgml_op.add (%457, %15) : (!dxgml.tensor<1x249x512x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x249x512x!dxgml.float16>
%459 = dxgml_op.transpose (%458) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<1x249x512x!dxgml.float16>) -> !dxgml.tensor<1x512x249x!dxgml.float16>
%460 = dxgml_op.divide (%459, %353) : (!dxgml.tensor<1x512x249x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x249x!dxgml.float16>
%461 = dxgml_op.erf (%460) : (!dxgml.tensor<1x512x249x!dxgml.float16>) -> !dxgml.tensor<1x512x249x!dxgml.float16>
%462 = dxgml_op.add (%461, %349) : (!dxgml.tensor<1x512x249x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x249x!dxgml.float16>
%463 = dxgml_op.multiply (%459, %462) : (!dxgml.tensor<1x512x249x!dxgml.float16>, !dxgml.tensor<1x512x249x!dxgml.float16>) -> !dxgml.tensor<1x512x249x!dxgml.float16>
%464 = dxgml_op.multiply (%463, %354) : (!dxgml.tensor<1x512x249x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x249x!dxgml.float16>
%465 = dxgml_op.convolution (%464, %16, %17) {dilations = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>, end_padding = #dxgml.dense_integer_elements<[0]> : !dxgml.tensor<1x!dxgml.int64>, group_count = #dxgml.integer<1 : !dxgml.int64>, start_padding = #dxgml.dense_integer_elements<[0]> : !dxgml.tensor<1x!dxgml.int64>, strides = #dxgml.dense_integer_elements<[2]> : !dxgml.tensor<1x!dxgml.int64>} : (!dxgml.tensor<1x512x249x!dxgml.float16>, !dxgml.tensor<512x512x3x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x512x124x!dxgml.float16>
%466 = dxgml_op.transpose (%465) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<1x512x124x!dxgml.float16>) -> !dxgml.tensor<1x124x512x!dxgml.float16>
%467 = dxgml_op.reduce (%466) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x124x512x!dxgml.float16>) -> !dxgml.tensor<1x124x1x!dxgml.float16>
%468 = dxgml_op.subtract (%466, %467) : (!dxgml.tensor<1x124x512x!dxgml.float16>, !dxgml.tensor<1x124x1x!dxgml.float16>) -> !dxgml.tensor<1x124x512x!dxgml.float16>
%469 = dxgml_op.pow (%468, %351) : (!dxgml.tensor<1x124x512x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x124x512x!dxgml.float16>
%470 = dxgml_op.reduce (%469) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x124x512x!dxgml.float16>) -> !dxgml.tensor<1x124x1x!dxgml.float16>
%471 = dxgml_op.add (%470, %352) : (!dxgml.tensor<1x124x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x124x1x!dxgml.float16>
%472 = dxgml_op.sqrt (%471) : (!dxgml.tensor<1x124x1x!dxgml.float16>) -> !dxgml.tensor<1x124x1x!dxgml.float16>
%473 = dxgml_op.divide (%468, %472) : (!dxgml.tensor<1x124x512x!dxgml.float16>, !dxgml.tensor<1x124x1x!dxgml.float16>) -> !dxgml.tensor<1x124x512x!dxgml.float16>
%474 = dxgml_op.multiply (%473, %18) : (!dxgml.tensor<1x124x512x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x124x512x!dxgml.float16>
%475 = dxgml_op.add (%474, %19) : (!dxgml.tensor<1x124x512x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x124x512x!dxgml.float16>
%476 = dxgml_op.transpose (%475) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<1x124x512x!dxgml.float16>) -> !dxgml.tensor<1x512x124x!dxgml.float16>
%477 = dxgml_op.divide (%476, %353) : (!dxgml.tensor<1x512x124x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x124x!dxgml.float16>
%478 = dxgml_op.erf (%477) : (!dxgml.tensor<1x512x124x!dxgml.float16>) -> !dxgml.tensor<1x512x124x!dxgml.float16>
%479 = dxgml_op.add (%478, %349) : (!dxgml.tensor<1x512x124x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x124x!dxgml.float16>
%480 = dxgml_op.multiply (%476, %479) : (!dxgml.tensor<1x512x124x!dxgml.float16>, !dxgml.tensor<1x512x124x!dxgml.float16>) -> !dxgml.tensor<1x512x124x!dxgml.float16>
%481 = dxgml_op.multiply (%480, %354) : (!dxgml.tensor<1x512x124x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x124x!dxgml.float16>
%482 = dxgml_op.convolution (%481, %20, %21) {dilations = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>, end_padding = #dxgml.dense_integer_elements<[0]> : !dxgml.tensor<1x!dxgml.int64>, group_count = #dxgml.integer<1 : !dxgml.int64>, start_padding = #dxgml.dense_integer_elements<[0]> : !dxgml.tensor<1x!dxgml.int64>, strides = #dxgml.dense_integer_elements<[2]> : !dxgml.tensor<1x!dxgml.int64>} : (!dxgml.tensor<1x512x124x!dxgml.float16>, !dxgml.tensor<512x512x2x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x512x62x!dxgml.float16>
%483 = dxgml_op.transpose (%482) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<1x512x62x!dxgml.float16>) -> !dxgml.tensor<1x62x512x!dxgml.float16>
%484 = dxgml_op.reduce (%483) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x62x512x!dxgml.float16>) -> !dxgml.tensor<1x62x1x!dxgml.float16>
%485 = dxgml_op.subtract (%483, %484) : (!dxgml.tensor<1x62x512x!dxgml.float16>, !dxgml.tensor<1x62x1x!dxgml.float16>) -> !dxgml.tensor<1x62x512x!dxgml.float16>
%486 = dxgml_op.pow (%485, %351) : (!dxgml.tensor<1x62x512x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x62x512x!dxgml.float16>
%487 = dxgml_op.reduce (%486) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x62x512x!dxgml.float16>) -> !dxgml.tensor<1x62x1x!dxgml.float16>
%488 = dxgml_op.add (%487, %352) : (!dxgml.tensor<1x62x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x62x1x!dxgml.float16>
%489 = dxgml_op.sqrt (%488) : (!dxgml.tensor<1x62x1x!dxgml.float16>) -> !dxgml.tensor<1x62x1x!dxgml.float16>
%490 = dxgml_op.divide (%485, %489) : (!dxgml.tensor<1x62x512x!dxgml.float16>, !dxgml.tensor<1x62x1x!dxgml.float16>) -> !dxgml.tensor<1x62x512x!dxgml.float16>
%491 = dxgml_op.multiply (%490, %22) : (!dxgml.tensor<1x62x512x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x62x512x!dxgml.float16>
%492 = dxgml_op.add (%491, %23) : (!dxgml.tensor<1x62x512x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x62x512x!dxgml.float16>
%493 = dxgml_op.transpose (%492) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<1x62x512x!dxgml.float16>) -> !dxgml.tensor<1x512x62x!dxgml.float16>
%494 = dxgml_op.divide (%493, %353) : (!dxgml.tensor<1x512x62x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x62x!dxgml.float16>
%495 = dxgml_op.erf (%494) : (!dxgml.tensor<1x512x62x!dxgml.float16>) -> !dxgml.tensor<1x512x62x!dxgml.float16>
%496 = dxgml_op.add (%495, %349) : (!dxgml.tensor<1x512x62x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x62x!dxgml.float16>
%497 = dxgml_op.multiply (%493, %496) : (!dxgml.tensor<1x512x62x!dxgml.float16>, !dxgml.tensor<1x512x62x!dxgml.float16>) -> !dxgml.tensor<1x512x62x!dxgml.float16>
%498 = dxgml_op.multiply (%497, %354) : (!dxgml.tensor<1x512x62x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x62x!dxgml.float16>
%499 = dxgml_op.convolution (%498, %24, %25) {dilations = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>, end_padding = #dxgml.dense_integer_elements<[0]> : !dxgml.tensor<1x!dxgml.int64>, group_count = #dxgml.integer<1 : !dxgml.int64>, start_padding = #dxgml.dense_integer_elements<[0]> : !dxgml.tensor<1x!dxgml.int64>, strides = #dxgml.dense_integer_elements<[2]> : !dxgml.tensor<1x!dxgml.int64>} : (!dxgml.tensor<1x512x62x!dxgml.float16>, !dxgml.tensor<512x512x2x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x512x31x!dxgml.float16>
%500 = dxgml_op.transpose (%499) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<1x512x31x!dxgml.float16>) -> !dxgml.tensor<1x31x512x!dxgml.float16>
%501 = dxgml_op.reduce (%500) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x512x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%502 = dxgml_op.subtract (%500, %501) : (!dxgml.tensor<1x31x512x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x512x!dxgml.float16>
%503 = dxgml_op.pow (%502, %351) : (!dxgml.tensor<1x31x512x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x512x!dxgml.float16>
%504 = dxgml_op.reduce (%503) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x512x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%505 = dxgml_op.add (%504, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%506 = dxgml_op.sqrt (%505) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%507 = dxgml_op.divide (%502, %506) : (!dxgml.tensor<1x31x512x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x512x!dxgml.float16>
%508 = dxgml_op.multiply (%507, %26) : (!dxgml.tensor<1x31x512x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x31x512x!dxgml.float16>
%509 = dxgml_op.add (%508, %27) : (!dxgml.tensor<1x31x512x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x31x512x!dxgml.float16>
%510 = dxgml_op.transpose (%509) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<1x31x512x!dxgml.float16>) -> !dxgml.tensor<1x512x31x!dxgml.float16>
%511 = dxgml_op.divide (%510, %353) : (!dxgml.tensor<1x512x31x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x31x!dxgml.float16>
%512 = dxgml_op.erf (%511) : (!dxgml.tensor<1x512x31x!dxgml.float16>) -> !dxgml.tensor<1x512x31x!dxgml.float16>
%513 = dxgml_op.add (%512, %349) : (!dxgml.tensor<1x512x31x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x31x!dxgml.float16>
%514 = dxgml_op.multiply (%510, %513) : (!dxgml.tensor<1x512x31x!dxgml.float16>, !dxgml.tensor<1x512x31x!dxgml.float16>) -> !dxgml.tensor<1x512x31x!dxgml.float16>
%515 = dxgml_op.multiply (%514, %354) : (!dxgml.tensor<1x512x31x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x512x31x!dxgml.float16>
%516 = dxgml_op.transpose (%515) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<1x512x31x!dxgml.float16>) -> !dxgml.tensor<1x31x512x!dxgml.float16>
%517 = dxgml_op.reduce (%516) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x512x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%518 = dxgml_op.subtract (%516, %517) : (!dxgml.tensor<1x31x512x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x512x!dxgml.float16>
%519 = dxgml_op.pow (%518, %351) : (!dxgml.tensor<1x31x512x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x512x!dxgml.float16>
%520 = dxgml_op.reduce (%519) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x512x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%521 = dxgml_op.add (%520, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%522 = dxgml_op.sqrt (%521) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%523 = dxgml_op.divide (%518, %522) : (!dxgml.tensor<1x31x512x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x512x!dxgml.float16>
%524 = dxgml_op.multiply (%523, %28) : (!dxgml.tensor<1x31x512x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x31x512x!dxgml.float16>
%525 = dxgml_op.add (%524, %29) : (!dxgml.tensor<1x31x512x!dxgml.float16>, !dxgml.tensor<512x!dxgml.float16>) -> !dxgml.tensor<1x31x512x!dxgml.float16>
%526 = dxgml_op.gemm (%525, %274) : (!dxgml.tensor<1x31x512x!dxgml.float16>, !dxgml.tensor<512x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%527 = dxgml_op.add (%30, %526) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%528 = dxgml_op.transpose (%527) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x1024x31x!dxgml.float16>
%529 = dxgml_op.convolution (%528, %275, %31) {dilations = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>, end_padding = #dxgml.dense_integer_elements<[64]> : !dxgml.tensor<1x!dxgml.int64>, group_count = #dxgml.integer<16 : !dxgml.int64>, start_padding = #dxgml.dense_integer_elements<[64]> : !dxgml.tensor<1x!dxgml.int64>, strides = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>} : (!dxgml.tensor<1x1024x31x!dxgml.float16>, !dxgml.tensor<1024x64x128x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x1024x32x!dxgml.float16>
%530 = dxgml_op.slice (%529) {axes = #dxgml.dense_integer_elements<[2]> : !dxgml.tensor<1x!dxgml.int64>, ends = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, input_window_offsets = [0, 0, 0], input_window_sizes = [1, 1024, 0], input_window_strides = [1, 1, 1], starts = #dxgml.dense_integer_elements<[0]> : !dxgml.tensor<1x!dxgml.int64>, steps = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>} : (!dxgml.tensor<1x1024x32x!dxgml.float16>) -> !dxgml.tensor<1x1024x31x!dxgml.float16>
%531 = dxgml_op.divide (%530, %353) : (!dxgml.tensor<1x1024x31x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x1024x31x!dxgml.float16>
%532 = dxgml_op.erf (%531) : (!dxgml.tensor<1x1024x31x!dxgml.float16>) -> !dxgml.tensor<1x1024x31x!dxgml.float16>
%533 = dxgml_op.add (%532, %349) : (!dxgml.tensor<1x1024x31x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x1024x31x!dxgml.float16>
%534 = dxgml_op.multiply (%530, %533) : (!dxgml.tensor<1x1024x31x!dxgml.float16>, !dxgml.tensor<1x1024x31x!dxgml.float16>) -> !dxgml.tensor<1x1024x31x!dxgml.float16>
%535 = dxgml_op.multiply (%534, %354) : (!dxgml.tensor<1x1024x31x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x1024x31x!dxgml.float16>
%536 = dxgml_op.transpose (%535) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<1x1024x31x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%537 = dxgml_op.add (%527, %536) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%538 = dxgml_op.reduce (%537) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%539 = dxgml_op.subtract (%537, %538) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%540 = dxgml_op.pow (%539, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%541 = dxgml_op.reduce (%540) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%542 = dxgml_op.add (%541, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%543 = dxgml_op.sqrt (%542) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%544 = dxgml_op.divide (%539, %543) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%545 = dxgml_op.multiply (%544, %38) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%546 = dxgml_op.add (%545, %39) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%547 = dxgml_op.gemm (%546, %358) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%548:3 = dxgml_op.split(%547) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%549 = dxgml_op.add (%36, %548#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%550 = dxgml_op.multiply (%549, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%551 = dxgml_op.add (%34, %548#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%552 = dxgml_op.reshape (%551) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%553 = dxgml_op.transpose (%552) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%554 = dxgml_op.add (%35, %548#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%555 = dxgml_op.reshape (%554) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%556 = dxgml_op.transpose (%555) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%557 = dxgml_op.reshape (%550) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%558 = dxgml_op.transpose (%557) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%559 = dxgml_op.reshape (%558) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%560 = dxgml_op.reshape (%553) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%561 = dxgml_op.reshape (%556) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%562 = dxgml_op.transpose (%560) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%563 = dxgml_op.gemm (%559, %562) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%564 = dxgml_op.softmax (%563) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%565 = dxgml_op.gemm (%564, %561) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%566 = dxgml_op.reshape (%565) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%567 = dxgml_op.transpose (%566) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%568 = dxgml_op.reshape (%567) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%569 = dxgml_op.gemm (%568, %276) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%570 = dxgml_op.add (%37, %569) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%571 = dxgml_op.add (%537, %570) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%572 = dxgml_op.reduce (%571) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%573 = dxgml_op.subtract (%571, %572) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%574 = dxgml_op.pow (%573, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%575 = dxgml_op.reduce (%574) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%576 = dxgml_op.add (%575, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%577 = dxgml_op.sqrt (%576) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%578 = dxgml_op.divide (%573, %577) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%579 = dxgml_op.multiply (%578, %42) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%580 = dxgml_op.add (%579, %43) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%581 = dxgml_op.gemm (%580, %277) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%582 = dxgml_op.add (%40, %581) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%583 = dxgml_op.divide (%582, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%584 = dxgml_op.erf (%583) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%585 = dxgml_op.add (%584, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%586 = dxgml_op.multiply (%582, %585) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%587 = dxgml_op.multiply (%586, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%588 = dxgml_op.gemm (%587, %278) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%589 = dxgml_op.add (%41, %588) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%590 = dxgml_op.add (%571, %589) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%591 = dxgml_op.reduce (%590) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%592 = dxgml_op.subtract (%590, %591) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%593 = dxgml_op.pow (%592, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%594 = dxgml_op.reduce (%593) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%595 = dxgml_op.add (%594, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%596 = dxgml_op.sqrt (%595) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%597 = dxgml_op.divide (%592, %596) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%598 = dxgml_op.multiply (%597, %48) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%599 = dxgml_op.add (%598, %49) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%600 = dxgml_op.gemm (%599, %359) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%601:3 = dxgml_op.split(%600) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%602 = dxgml_op.add (%46, %601#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%603 = dxgml_op.multiply (%602, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%604 = dxgml_op.add (%44, %601#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%605 = dxgml_op.reshape (%604) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%606 = dxgml_op.transpose (%605) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%607 = dxgml_op.add (%45, %601#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%608 = dxgml_op.reshape (%607) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%609 = dxgml_op.transpose (%608) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%610 = dxgml_op.reshape (%603) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%611 = dxgml_op.transpose (%610) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%612 = dxgml_op.reshape (%611) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%613 = dxgml_op.reshape (%606) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%614 = dxgml_op.reshape (%609) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%615 = dxgml_op.transpose (%613) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%616 = dxgml_op.gemm (%612, %615) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%617 = dxgml_op.softmax (%616) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%618 = dxgml_op.gemm (%617, %614) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%619 = dxgml_op.reshape (%618) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%620 = dxgml_op.transpose (%619) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%621 = dxgml_op.reshape (%620) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%622 = dxgml_op.gemm (%621, %279) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%623 = dxgml_op.add (%47, %622) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%624 = dxgml_op.add (%590, %623) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%625 = dxgml_op.reduce (%624) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%626 = dxgml_op.subtract (%624, %625) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%627 = dxgml_op.pow (%626, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%628 = dxgml_op.reduce (%627) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%629 = dxgml_op.add (%628, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%630 = dxgml_op.sqrt (%629) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%631 = dxgml_op.divide (%626, %630) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%632 = dxgml_op.multiply (%631, %52) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%633 = dxgml_op.add (%632, %53) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%634 = dxgml_op.gemm (%633, %280) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%635 = dxgml_op.add (%50, %634) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%636 = dxgml_op.divide (%635, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%637 = dxgml_op.erf (%636) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%638 = dxgml_op.add (%637, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%639 = dxgml_op.multiply (%635, %638) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%640 = dxgml_op.multiply (%639, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%641 = dxgml_op.gemm (%640, %281) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%642 = dxgml_op.add (%51, %641) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%643 = dxgml_op.add (%624, %642) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%644 = dxgml_op.reduce (%643) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%645 = dxgml_op.subtract (%643, %644) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%646 = dxgml_op.pow (%645, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%647 = dxgml_op.reduce (%646) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%648 = dxgml_op.add (%647, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%649 = dxgml_op.sqrt (%648) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%650 = dxgml_op.divide (%645, %649) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%651 = dxgml_op.multiply (%650, %58) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%652 = dxgml_op.add (%651, %59) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%653 = dxgml_op.gemm (%652, %360) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%654:3 = dxgml_op.split(%653) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%655 = dxgml_op.add (%56, %654#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%656 = dxgml_op.multiply (%655, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%657 = dxgml_op.add (%54, %654#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%658 = dxgml_op.reshape (%657) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%659 = dxgml_op.transpose (%658) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%660 = dxgml_op.add (%55, %654#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%661 = dxgml_op.reshape (%660) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%662 = dxgml_op.transpose (%661) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%663 = dxgml_op.reshape (%656) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%664 = dxgml_op.transpose (%663) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%665 = dxgml_op.reshape (%664) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%666 = dxgml_op.reshape (%659) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%667 = dxgml_op.reshape (%662) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%668 = dxgml_op.transpose (%666) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%669 = dxgml_op.gemm (%665, %668) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%670 = dxgml_op.softmax (%669) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%671 = dxgml_op.gemm (%670, %667) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%672 = dxgml_op.reshape (%671) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%673 = dxgml_op.transpose (%672) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%674 = dxgml_op.reshape (%673) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%675 = dxgml_op.gemm (%674, %282) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%676 = dxgml_op.add (%57, %675) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%677 = dxgml_op.add (%643, %676) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%678 = dxgml_op.reduce (%677) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%679 = dxgml_op.subtract (%677, %678) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%680 = dxgml_op.pow (%679, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%681 = dxgml_op.reduce (%680) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%682 = dxgml_op.add (%681, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%683 = dxgml_op.sqrt (%682) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%684 = dxgml_op.divide (%679, %683) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%685 = dxgml_op.multiply (%684, %62) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%686 = dxgml_op.add (%685, %63) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%687 = dxgml_op.gemm (%686, %283) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%688 = dxgml_op.add (%60, %687) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%689 = dxgml_op.divide (%688, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%690 = dxgml_op.erf (%689) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%691 = dxgml_op.add (%690, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%692 = dxgml_op.multiply (%688, %691) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%693 = dxgml_op.multiply (%692, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%694 = dxgml_op.gemm (%693, %284) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%695 = dxgml_op.add (%61, %694) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%696 = dxgml_op.add (%677, %695) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%697 = dxgml_op.reduce (%696) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%698 = dxgml_op.subtract (%696, %697) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%699 = dxgml_op.pow (%698, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%700 = dxgml_op.reduce (%699) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%701 = dxgml_op.add (%700, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%702 = dxgml_op.sqrt (%701) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%703 = dxgml_op.divide (%698, %702) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%704 = dxgml_op.multiply (%703, %68) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%705 = dxgml_op.add (%704, %69) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%706 = dxgml_op.gemm (%705, %361) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%707:3 = dxgml_op.split(%706) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%708 = dxgml_op.add (%66, %707#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%709 = dxgml_op.multiply (%708, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%710 = dxgml_op.add (%64, %707#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%711 = dxgml_op.reshape (%710) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%712 = dxgml_op.transpose (%711) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%713 = dxgml_op.add (%65, %707#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%714 = dxgml_op.reshape (%713) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%715 = dxgml_op.transpose (%714) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%716 = dxgml_op.reshape (%709) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%717 = dxgml_op.transpose (%716) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%718 = dxgml_op.reshape (%717) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%719 = dxgml_op.reshape (%712) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%720 = dxgml_op.reshape (%715) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%721 = dxgml_op.transpose (%719) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%722 = dxgml_op.gemm (%718, %721) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%723 = dxgml_op.softmax (%722) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%724 = dxgml_op.gemm (%723, %720) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%725 = dxgml_op.reshape (%724) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%726 = dxgml_op.transpose (%725) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%727 = dxgml_op.reshape (%726) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%728 = dxgml_op.gemm (%727, %285) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%729 = dxgml_op.add (%67, %728) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%730 = dxgml_op.add (%696, %729) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%731 = dxgml_op.reduce (%730) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%732 = dxgml_op.subtract (%730, %731) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%733 = dxgml_op.pow (%732, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%734 = dxgml_op.reduce (%733) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%735 = dxgml_op.add (%734, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%736 = dxgml_op.sqrt (%735) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%737 = dxgml_op.divide (%732, %736) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%738 = dxgml_op.multiply (%737, %72) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%739 = dxgml_op.add (%738, %73) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%740 = dxgml_op.gemm (%739, %286) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%741 = dxgml_op.add (%70, %740) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%742 = dxgml_op.divide (%741, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%743 = dxgml_op.erf (%742) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%744 = dxgml_op.add (%743, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%745 = dxgml_op.multiply (%741, %744) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%746 = dxgml_op.multiply (%745, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%747 = dxgml_op.gemm (%746, %287) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%748 = dxgml_op.add (%71, %747) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%749 = dxgml_op.add (%730, %748) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%750 = dxgml_op.reduce (%749) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%751 = dxgml_op.subtract (%749, %750) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%752 = dxgml_op.pow (%751, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%753 = dxgml_op.reduce (%752) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%754 = dxgml_op.add (%753, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%755 = dxgml_op.sqrt (%754) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%756 = dxgml_op.divide (%751, %755) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%757 = dxgml_op.multiply (%756, %78) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%758 = dxgml_op.add (%757, %79) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%759 = dxgml_op.gemm (%758, %362) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%760:3 = dxgml_op.split(%759) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%761 = dxgml_op.add (%76, %760#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%762 = dxgml_op.multiply (%761, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%763 = dxgml_op.add (%74, %760#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%764 = dxgml_op.reshape (%763) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%765 = dxgml_op.transpose (%764) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%766 = dxgml_op.add (%75, %760#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%767 = dxgml_op.reshape (%766) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%768 = dxgml_op.transpose (%767) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%769 = dxgml_op.reshape (%762) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%770 = dxgml_op.transpose (%769) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%771 = dxgml_op.reshape (%770) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%772 = dxgml_op.reshape (%765) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%773 = dxgml_op.reshape (%768) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%774 = dxgml_op.transpose (%772) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%775 = dxgml_op.gemm (%771, %774) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%776 = dxgml_op.softmax (%775) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%777 = dxgml_op.gemm (%776, %773) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%778 = dxgml_op.reshape (%777) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%779 = dxgml_op.transpose (%778) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%780 = dxgml_op.reshape (%779) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%781 = dxgml_op.gemm (%780, %288) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%782 = dxgml_op.add (%77, %781) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%783 = dxgml_op.add (%749, %782) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%784 = dxgml_op.reduce (%783) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%785 = dxgml_op.subtract (%783, %784) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%786 = dxgml_op.pow (%785, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%787 = dxgml_op.reduce (%786) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%788 = dxgml_op.add (%787, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%789 = dxgml_op.sqrt (%788) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%790 = dxgml_op.divide (%785, %789) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%791 = dxgml_op.multiply (%790, %82) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%792 = dxgml_op.add (%791, %83) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%793 = dxgml_op.gemm (%792, %289) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%794 = dxgml_op.add (%80, %793) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%795 = dxgml_op.divide (%794, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%796 = dxgml_op.erf (%795) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%797 = dxgml_op.add (%796, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%798 = dxgml_op.multiply (%794, %797) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%799 = dxgml_op.multiply (%798, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%800 = dxgml_op.gemm (%799, %290) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%801 = dxgml_op.add (%81, %800) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%802 = dxgml_op.add (%783, %801) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%803 = dxgml_op.reduce (%802) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%804 = dxgml_op.subtract (%802, %803) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%805 = dxgml_op.pow (%804, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%806 = dxgml_op.reduce (%805) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%807 = dxgml_op.add (%806, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%808 = dxgml_op.sqrt (%807) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%809 = dxgml_op.divide (%804, %808) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%810 = dxgml_op.multiply (%809, %88) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%811 = dxgml_op.add (%810, %89) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%812 = dxgml_op.gemm (%811, %363) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%813:3 = dxgml_op.split(%812) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%814 = dxgml_op.add (%86, %813#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%815 = dxgml_op.multiply (%814, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%816 = dxgml_op.add (%84, %813#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%817 = dxgml_op.reshape (%816) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%818 = dxgml_op.transpose (%817) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%819 = dxgml_op.add (%85, %813#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%820 = dxgml_op.reshape (%819) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%821 = dxgml_op.transpose (%820) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%822 = dxgml_op.reshape (%815) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%823 = dxgml_op.transpose (%822) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%824 = dxgml_op.reshape (%823) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%825 = dxgml_op.reshape (%818) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%826 = dxgml_op.reshape (%821) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%827 = dxgml_op.transpose (%825) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%828 = dxgml_op.gemm (%824, %827) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%829 = dxgml_op.softmax (%828) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%830 = dxgml_op.gemm (%829, %826) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%831 = dxgml_op.reshape (%830) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%832 = dxgml_op.transpose (%831) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%833 = dxgml_op.reshape (%832) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%834 = dxgml_op.gemm (%833, %291) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%835 = dxgml_op.add (%87, %834) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%836 = dxgml_op.add (%802, %835) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%837 = dxgml_op.reduce (%836) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%838 = dxgml_op.subtract (%836, %837) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%839 = dxgml_op.pow (%838, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%840 = dxgml_op.reduce (%839) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%841 = dxgml_op.add (%840, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%842 = dxgml_op.sqrt (%841) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%843 = dxgml_op.divide (%838, %842) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%844 = dxgml_op.multiply (%843, %92) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%845 = dxgml_op.add (%844, %93) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%846 = dxgml_op.gemm (%845, %292) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%847 = dxgml_op.add (%90, %846) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%848 = dxgml_op.divide (%847, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%849 = dxgml_op.erf (%848) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%850 = dxgml_op.add (%849, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%851 = dxgml_op.multiply (%847, %850) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%852 = dxgml_op.multiply (%851, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%853 = dxgml_op.gemm (%852, %293) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%854 = dxgml_op.add (%91, %853) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%855 = dxgml_op.add (%836, %854) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%856 = dxgml_op.reduce (%855) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%857 = dxgml_op.subtract (%855, %856) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%858 = dxgml_op.pow (%857, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%859 = dxgml_op.reduce (%858) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%860 = dxgml_op.add (%859, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%861 = dxgml_op.sqrt (%860) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%862 = dxgml_op.divide (%857, %861) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%863 = dxgml_op.multiply (%862, %98) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%864 = dxgml_op.add (%863, %99) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%865 = dxgml_op.gemm (%864, %364) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%866:3 = dxgml_op.split(%865) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%867 = dxgml_op.add (%96, %866#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%868 = dxgml_op.multiply (%867, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%869 = dxgml_op.add (%94, %866#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%870 = dxgml_op.reshape (%869) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%871 = dxgml_op.transpose (%870) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%872 = dxgml_op.add (%95, %866#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%873 = dxgml_op.reshape (%872) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%874 = dxgml_op.transpose (%873) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%875 = dxgml_op.reshape (%868) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%876 = dxgml_op.transpose (%875) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%877 = dxgml_op.reshape (%876) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%878 = dxgml_op.reshape (%871) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%879 = dxgml_op.reshape (%874) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%880 = dxgml_op.transpose (%878) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%881 = dxgml_op.gemm (%877, %880) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%882 = dxgml_op.softmax (%881) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%883 = dxgml_op.gemm (%882, %879) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%884 = dxgml_op.reshape (%883) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%885 = dxgml_op.transpose (%884) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%886 = dxgml_op.reshape (%885) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%887 = dxgml_op.gemm (%886, %294) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%888 = dxgml_op.add (%97, %887) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%889 = dxgml_op.add (%855, %888) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%890 = dxgml_op.reduce (%889) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%891 = dxgml_op.subtract (%889, %890) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%892 = dxgml_op.pow (%891, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%893 = dxgml_op.reduce (%892) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%894 = dxgml_op.add (%893, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%895 = dxgml_op.sqrt (%894) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%896 = dxgml_op.divide (%891, %895) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%897 = dxgml_op.multiply (%896, %102) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%898 = dxgml_op.add (%897, %103) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%899 = dxgml_op.gemm (%898, %295) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%900 = dxgml_op.add (%100, %899) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%901 = dxgml_op.divide (%900, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%902 = dxgml_op.erf (%901) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%903 = dxgml_op.add (%902, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%904 = dxgml_op.multiply (%900, %903) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%905 = dxgml_op.multiply (%904, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%906 = dxgml_op.gemm (%905, %296) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%907 = dxgml_op.add (%101, %906) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%908 = dxgml_op.add (%889, %907) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%909 = dxgml_op.reduce (%908) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%910 = dxgml_op.subtract (%908, %909) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%911 = dxgml_op.pow (%910, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%912 = dxgml_op.reduce (%911) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%913 = dxgml_op.add (%912, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%914 = dxgml_op.sqrt (%913) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%915 = dxgml_op.divide (%910, %914) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%916 = dxgml_op.multiply (%915, %108) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%917 = dxgml_op.add (%916, %109) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%918 = dxgml_op.gemm (%917, %365) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%919:3 = dxgml_op.split(%918) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%920 = dxgml_op.add (%106, %919#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%921 = dxgml_op.multiply (%920, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%922 = dxgml_op.add (%104, %919#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%923 = dxgml_op.reshape (%922) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%924 = dxgml_op.transpose (%923) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%925 = dxgml_op.add (%105, %919#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%926 = dxgml_op.reshape (%925) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%927 = dxgml_op.transpose (%926) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%928 = dxgml_op.reshape (%921) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%929 = dxgml_op.transpose (%928) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%930 = dxgml_op.reshape (%929) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%931 = dxgml_op.reshape (%924) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%932 = dxgml_op.reshape (%927) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%933 = dxgml_op.transpose (%931) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%934 = dxgml_op.gemm (%930, %933) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%935 = dxgml_op.softmax (%934) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%936 = dxgml_op.gemm (%935, %932) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%937 = dxgml_op.reshape (%936) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%938 = dxgml_op.transpose (%937) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%939 = dxgml_op.reshape (%938) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%940 = dxgml_op.gemm (%939, %297) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%941 = dxgml_op.add (%107, %940) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%942 = dxgml_op.add (%908, %941) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%943 = dxgml_op.reduce (%942) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%944 = dxgml_op.subtract (%942, %943) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%945 = dxgml_op.pow (%944, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%946 = dxgml_op.reduce (%945) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%947 = dxgml_op.add (%946, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%948 = dxgml_op.sqrt (%947) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%949 = dxgml_op.divide (%944, %948) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%950 = dxgml_op.multiply (%949, %112) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%951 = dxgml_op.add (%950, %113) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%952 = dxgml_op.gemm (%951, %298) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%953 = dxgml_op.add (%110, %952) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%954 = dxgml_op.divide (%953, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%955 = dxgml_op.erf (%954) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%956 = dxgml_op.add (%955, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%957 = dxgml_op.multiply (%953, %956) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%958 = dxgml_op.multiply (%957, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%959 = dxgml_op.gemm (%958, %299) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%960 = dxgml_op.add (%111, %959) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%961 = dxgml_op.add (%942, %960) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%962 = dxgml_op.reduce (%961) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%963 = dxgml_op.subtract (%961, %962) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%964 = dxgml_op.pow (%963, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%965 = dxgml_op.reduce (%964) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%966 = dxgml_op.add (%965, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%967 = dxgml_op.sqrt (%966) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%968 = dxgml_op.divide (%963, %967) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%969 = dxgml_op.multiply (%968, %118) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%970 = dxgml_op.add (%969, %119) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%971 = dxgml_op.gemm (%970, %366) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%972:3 = dxgml_op.split(%971) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%973 = dxgml_op.add (%116, %972#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%974 = dxgml_op.multiply (%973, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%975 = dxgml_op.add (%114, %972#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%976 = dxgml_op.reshape (%975) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%977 = dxgml_op.transpose (%976) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%978 = dxgml_op.add (%115, %972#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%979 = dxgml_op.reshape (%978) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%980 = dxgml_op.transpose (%979) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%981 = dxgml_op.reshape (%974) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%982 = dxgml_op.transpose (%981) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%983 = dxgml_op.reshape (%982) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%984 = dxgml_op.reshape (%977) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%985 = dxgml_op.reshape (%980) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%986 = dxgml_op.transpose (%984) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%987 = dxgml_op.gemm (%983, %986) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%988 = dxgml_op.softmax (%987) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%989 = dxgml_op.gemm (%988, %985) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%990 = dxgml_op.reshape (%989) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%991 = dxgml_op.transpose (%990) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%992 = dxgml_op.reshape (%991) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%993 = dxgml_op.gemm (%992, %300) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%994 = dxgml_op.add (%117, %993) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%995 = dxgml_op.add (%961, %994) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%996 = dxgml_op.reduce (%995) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%997 = dxgml_op.subtract (%995, %996) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%998 = dxgml_op.pow (%997, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%999 = dxgml_op.reduce (%998) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1000 = dxgml_op.add (%999, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1001 = dxgml_op.sqrt (%1000) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1002 = dxgml_op.divide (%997, %1001) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1003 = dxgml_op.multiply (%1002, %122) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1004 = dxgml_op.add (%1003, %123) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1005 = dxgml_op.gemm (%1004, %301) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1006 = dxgml_op.add (%120, %1005) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1007 = dxgml_op.divide (%1006, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1008 = dxgml_op.erf (%1007) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1009 = dxgml_op.add (%1008, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1010 = dxgml_op.multiply (%1006, %1009) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1011 = dxgml_op.multiply (%1010, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1012 = dxgml_op.gemm (%1011, %302) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1013 = dxgml_op.add (%121, %1012) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1014 = dxgml_op.add (%995, %1013) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1015 = dxgml_op.reduce (%1014) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1016 = dxgml_op.subtract (%1014, %1015) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1017 = dxgml_op.pow (%1016, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1018 = dxgml_op.reduce (%1017) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1019 = dxgml_op.add (%1018, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1020 = dxgml_op.sqrt (%1019) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1021 = dxgml_op.divide (%1016, %1020) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1022 = dxgml_op.multiply (%1021, %128) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1023 = dxgml_op.add (%1022, %129) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1024 = dxgml_op.gemm (%1023, %367) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%1025:3 = dxgml_op.split(%1024) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%1026 = dxgml_op.add (%126, %1025#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1027 = dxgml_op.multiply (%1026, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1028 = dxgml_op.add (%124, %1025#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1029 = dxgml_op.reshape (%1028) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1030 = dxgml_op.transpose (%1029) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1031 = dxgml_op.add (%125, %1025#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1032 = dxgml_op.reshape (%1031) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1033 = dxgml_op.transpose (%1032) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1034 = dxgml_op.reshape (%1027) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1035 = dxgml_op.transpose (%1034) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1036 = dxgml_op.reshape (%1035) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1037 = dxgml_op.reshape (%1030) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1038 = dxgml_op.reshape (%1033) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1039 = dxgml_op.transpose (%1037) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%1040 = dxgml_op.gemm (%1036, %1039) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1041 = dxgml_op.softmax (%1040) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1042 = dxgml_op.gemm (%1041, %1038) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1043 = dxgml_op.reshape (%1042) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1044 = dxgml_op.transpose (%1043) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1045 = dxgml_op.reshape (%1044) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1046 = dxgml_op.gemm (%1045, %303) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1047 = dxgml_op.add (%127, %1046) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1048 = dxgml_op.add (%1014, %1047) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1049 = dxgml_op.reduce (%1048) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1050 = dxgml_op.subtract (%1048, %1049) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1051 = dxgml_op.pow (%1050, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1052 = dxgml_op.reduce (%1051) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1053 = dxgml_op.add (%1052, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1054 = dxgml_op.sqrt (%1053) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1055 = dxgml_op.divide (%1050, %1054) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1056 = dxgml_op.multiply (%1055, %132) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1057 = dxgml_op.add (%1056, %133) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1058 = dxgml_op.gemm (%1057, %304) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1059 = dxgml_op.add (%130, %1058) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1060 = dxgml_op.divide (%1059, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1061 = dxgml_op.erf (%1060) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1062 = dxgml_op.add (%1061, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1063 = dxgml_op.multiply (%1059, %1062) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1064 = dxgml_op.multiply (%1063, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1065 = dxgml_op.gemm (%1064, %305) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1066 = dxgml_op.add (%131, %1065) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1067 = dxgml_op.add (%1048, %1066) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1068 = dxgml_op.reduce (%1067) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1069 = dxgml_op.subtract (%1067, %1068) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1070 = dxgml_op.pow (%1069, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1071 = dxgml_op.reduce (%1070) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1072 = dxgml_op.add (%1071, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1073 = dxgml_op.sqrt (%1072) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1074 = dxgml_op.divide (%1069, %1073) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1075 = dxgml_op.multiply (%1074, %138) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1076 = dxgml_op.add (%1075, %139) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1077 = dxgml_op.gemm (%1076, %368) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%1078:3 = dxgml_op.split(%1077) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%1079 = dxgml_op.add (%136, %1078#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1080 = dxgml_op.multiply (%1079, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1081 = dxgml_op.add (%134, %1078#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1082 = dxgml_op.reshape (%1081) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1083 = dxgml_op.transpose (%1082) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1084 = dxgml_op.add (%135, %1078#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1085 = dxgml_op.reshape (%1084) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1086 = dxgml_op.transpose (%1085) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1087 = dxgml_op.reshape (%1080) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1088 = dxgml_op.transpose (%1087) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1089 = dxgml_op.reshape (%1088) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1090 = dxgml_op.reshape (%1083) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1091 = dxgml_op.reshape (%1086) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1092 = dxgml_op.transpose (%1090) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%1093 = dxgml_op.gemm (%1089, %1092) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1094 = dxgml_op.softmax (%1093) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1095 = dxgml_op.gemm (%1094, %1091) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1096 = dxgml_op.reshape (%1095) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1097 = dxgml_op.transpose (%1096) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1098 = dxgml_op.reshape (%1097) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1099 = dxgml_op.gemm (%1098, %306) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1100 = dxgml_op.add (%137, %1099) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1101 = dxgml_op.add (%1067, %1100) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1102 = dxgml_op.reduce (%1101) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1103 = dxgml_op.subtract (%1101, %1102) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1104 = dxgml_op.pow (%1103, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1105 = dxgml_op.reduce (%1104) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1106 = dxgml_op.add (%1105, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1107 = dxgml_op.sqrt (%1106) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1108 = dxgml_op.divide (%1103, %1107) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1109 = dxgml_op.multiply (%1108, %142) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1110 = dxgml_op.add (%1109, %143) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1111 = dxgml_op.gemm (%1110, %307) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1112 = dxgml_op.add (%140, %1111) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1113 = dxgml_op.divide (%1112, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1114 = dxgml_op.erf (%1113) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1115 = dxgml_op.add (%1114, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1116 = dxgml_op.multiply (%1112, %1115) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1117 = dxgml_op.multiply (%1116, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1118 = dxgml_op.gemm (%1117, %308) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1119 = dxgml_op.add (%141, %1118) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1120 = dxgml_op.add (%1101, %1119) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1121 = dxgml_op.reduce (%1120) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1122 = dxgml_op.subtract (%1120, %1121) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1123 = dxgml_op.pow (%1122, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1124 = dxgml_op.reduce (%1123) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1125 = dxgml_op.add (%1124, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1126 = dxgml_op.sqrt (%1125) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1127 = dxgml_op.divide (%1122, %1126) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1128 = dxgml_op.multiply (%1127, %148) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1129 = dxgml_op.add (%1128, %149) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1130 = dxgml_op.gemm (%1129, %369) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%1131:3 = dxgml_op.split(%1130) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%1132 = dxgml_op.add (%146, %1131#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1133 = dxgml_op.multiply (%1132, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1134 = dxgml_op.add (%144, %1131#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1135 = dxgml_op.reshape (%1134) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1136 = dxgml_op.transpose (%1135) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1137 = dxgml_op.add (%145, %1131#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1138 = dxgml_op.reshape (%1137) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1139 = dxgml_op.transpose (%1138) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1140 = dxgml_op.reshape (%1133) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1141 = dxgml_op.transpose (%1140) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1142 = dxgml_op.reshape (%1141) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1143 = dxgml_op.reshape (%1136) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1144 = dxgml_op.reshape (%1139) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1145 = dxgml_op.transpose (%1143) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%1146 = dxgml_op.gemm (%1142, %1145) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1147 = dxgml_op.softmax (%1146) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1148 = dxgml_op.gemm (%1147, %1144) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1149 = dxgml_op.reshape (%1148) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1150 = dxgml_op.transpose (%1149) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1151 = dxgml_op.reshape (%1150) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1152 = dxgml_op.gemm (%1151, %309) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1153 = dxgml_op.add (%147, %1152) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1154 = dxgml_op.add (%1120, %1153) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1155 = dxgml_op.reduce (%1154) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1156 = dxgml_op.subtract (%1154, %1155) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1157 = dxgml_op.pow (%1156, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1158 = dxgml_op.reduce (%1157) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1159 = dxgml_op.add (%1158, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1160 = dxgml_op.sqrt (%1159) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1161 = dxgml_op.divide (%1156, %1160) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1162 = dxgml_op.multiply (%1161, %152) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1163 = dxgml_op.add (%1162, %153) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1164 = dxgml_op.gemm (%1163, %310) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1165 = dxgml_op.add (%150, %1164) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1166 = dxgml_op.divide (%1165, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1167 = dxgml_op.erf (%1166) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1168 = dxgml_op.add (%1167, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1169 = dxgml_op.multiply (%1165, %1168) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1170 = dxgml_op.multiply (%1169, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1171 = dxgml_op.gemm (%1170, %311) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1172 = dxgml_op.add (%151, %1171) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1173 = dxgml_op.add (%1154, %1172) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1174 = dxgml_op.reduce (%1173) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1175 = dxgml_op.subtract (%1173, %1174) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1176 = dxgml_op.pow (%1175, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1177 = dxgml_op.reduce (%1176) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1178 = dxgml_op.add (%1177, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1179 = dxgml_op.sqrt (%1178) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1180 = dxgml_op.divide (%1175, %1179) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1181 = dxgml_op.multiply (%1180, %158) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1182 = dxgml_op.add (%1181, %159) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1183 = dxgml_op.gemm (%1182, %370) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%1184:3 = dxgml_op.split(%1183) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%1185 = dxgml_op.add (%156, %1184#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1186 = dxgml_op.multiply (%1185, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1187 = dxgml_op.add (%154, %1184#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1188 = dxgml_op.reshape (%1187) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1189 = dxgml_op.transpose (%1188) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1190 = dxgml_op.add (%155, %1184#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1191 = dxgml_op.reshape (%1190) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1192 = dxgml_op.transpose (%1191) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1193 = dxgml_op.reshape (%1186) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1194 = dxgml_op.transpose (%1193) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1195 = dxgml_op.reshape (%1194) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1196 = dxgml_op.reshape (%1189) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1197 = dxgml_op.reshape (%1192) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1198 = dxgml_op.transpose (%1196) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%1199 = dxgml_op.gemm (%1195, %1198) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1200 = dxgml_op.softmax (%1199) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1201 = dxgml_op.gemm (%1200, %1197) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1202 = dxgml_op.reshape (%1201) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1203 = dxgml_op.transpose (%1202) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1204 = dxgml_op.reshape (%1203) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1205 = dxgml_op.gemm (%1204, %312) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1206 = dxgml_op.add (%157, %1205) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1207 = dxgml_op.add (%1173, %1206) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1208 = dxgml_op.reduce (%1207) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1209 = dxgml_op.subtract (%1207, %1208) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1210 = dxgml_op.pow (%1209, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1211 = dxgml_op.reduce (%1210) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1212 = dxgml_op.add (%1211, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1213 = dxgml_op.sqrt (%1212) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1214 = dxgml_op.divide (%1209, %1213) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1215 = dxgml_op.multiply (%1214, %162) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1216 = dxgml_op.add (%1215, %163) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1217 = dxgml_op.gemm (%1216, %313) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1218 = dxgml_op.add (%160, %1217) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1219 = dxgml_op.divide (%1218, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1220 = dxgml_op.erf (%1219) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1221 = dxgml_op.add (%1220, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1222 = dxgml_op.multiply (%1218, %1221) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1223 = dxgml_op.multiply (%1222, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1224 = dxgml_op.gemm (%1223, %314) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1225 = dxgml_op.add (%161, %1224) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1226 = dxgml_op.add (%1207, %1225) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1227 = dxgml_op.reduce (%1226) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1228 = dxgml_op.subtract (%1226, %1227) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1229 = dxgml_op.pow (%1228, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1230 = dxgml_op.reduce (%1229) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1231 = dxgml_op.add (%1230, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1232 = dxgml_op.sqrt (%1231) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1233 = dxgml_op.divide (%1228, %1232) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1234 = dxgml_op.multiply (%1233, %168) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1235 = dxgml_op.add (%1234, %169) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1236 = dxgml_op.gemm (%1235, %371) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%1237:3 = dxgml_op.split(%1236) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%1238 = dxgml_op.add (%166, %1237#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1239 = dxgml_op.multiply (%1238, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1240 = dxgml_op.add (%164, %1237#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1241 = dxgml_op.reshape (%1240) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1242 = dxgml_op.transpose (%1241) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1243 = dxgml_op.add (%165, %1237#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1244 = dxgml_op.reshape (%1243) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1245 = dxgml_op.transpose (%1244) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1246 = dxgml_op.reshape (%1239) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1247 = dxgml_op.transpose (%1246) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1248 = dxgml_op.reshape (%1247) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1249 = dxgml_op.reshape (%1242) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1250 = dxgml_op.reshape (%1245) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1251 = dxgml_op.transpose (%1249) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%1252 = dxgml_op.gemm (%1248, %1251) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1253 = dxgml_op.softmax (%1252) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1254 = dxgml_op.gemm (%1253, %1250) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1255 = dxgml_op.reshape (%1254) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1256 = dxgml_op.transpose (%1255) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1257 = dxgml_op.reshape (%1256) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1258 = dxgml_op.gemm (%1257, %315) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1259 = dxgml_op.add (%167, %1258) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1260 = dxgml_op.add (%1226, %1259) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1261 = dxgml_op.reduce (%1260) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1262 = dxgml_op.subtract (%1260, %1261) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1263 = dxgml_op.pow (%1262, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1264 = dxgml_op.reduce (%1263) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1265 = dxgml_op.add (%1264, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1266 = dxgml_op.sqrt (%1265) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1267 = dxgml_op.divide (%1262, %1266) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1268 = dxgml_op.multiply (%1267, %172) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1269 = dxgml_op.add (%1268, %173) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1270 = dxgml_op.gemm (%1269, %316) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1271 = dxgml_op.add (%170, %1270) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1272 = dxgml_op.divide (%1271, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1273 = dxgml_op.erf (%1272) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1274 = dxgml_op.add (%1273, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1275 = dxgml_op.multiply (%1271, %1274) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1276 = dxgml_op.multiply (%1275, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1277 = dxgml_op.gemm (%1276, %317) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1278 = dxgml_op.add (%171, %1277) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1279 = dxgml_op.add (%1260, %1278) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1280 = dxgml_op.reduce (%1279) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1281 = dxgml_op.subtract (%1279, %1280) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1282 = dxgml_op.pow (%1281, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1283 = dxgml_op.reduce (%1282) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1284 = dxgml_op.add (%1283, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1285 = dxgml_op.sqrt (%1284) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1286 = dxgml_op.divide (%1281, %1285) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1287 = dxgml_op.multiply (%1286, %178) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1288 = dxgml_op.add (%1287, %179) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1289 = dxgml_op.gemm (%1288, %372) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%1290:3 = dxgml_op.split(%1289) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%1291 = dxgml_op.add (%176, %1290#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1292 = dxgml_op.multiply (%1291, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1293 = dxgml_op.add (%174, %1290#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1294 = dxgml_op.reshape (%1293) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1295 = dxgml_op.transpose (%1294) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1296 = dxgml_op.add (%175, %1290#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1297 = dxgml_op.reshape (%1296) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1298 = dxgml_op.transpose (%1297) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1299 = dxgml_op.reshape (%1292) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1300 = dxgml_op.transpose (%1299) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1301 = dxgml_op.reshape (%1300) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1302 = dxgml_op.reshape (%1295) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1303 = dxgml_op.reshape (%1298) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1304 = dxgml_op.transpose (%1302) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%1305 = dxgml_op.gemm (%1301, %1304) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1306 = dxgml_op.softmax (%1305) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1307 = dxgml_op.gemm (%1306, %1303) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1308 = dxgml_op.reshape (%1307) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1309 = dxgml_op.transpose (%1308) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1310 = dxgml_op.reshape (%1309) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1311 = dxgml_op.gemm (%1310, %318) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1312 = dxgml_op.add (%177, %1311) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1313 = dxgml_op.add (%1279, %1312) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1314 = dxgml_op.reduce (%1313) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1315 = dxgml_op.subtract (%1313, %1314) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1316 = dxgml_op.pow (%1315, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1317 = dxgml_op.reduce (%1316) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1318 = dxgml_op.add (%1317, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1319 = dxgml_op.sqrt (%1318) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1320 = dxgml_op.divide (%1315, %1319) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1321 = dxgml_op.multiply (%1320, %182) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1322 = dxgml_op.add (%1321, %183) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1323 = dxgml_op.gemm (%1322, %319) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1324 = dxgml_op.add (%180, %1323) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1325 = dxgml_op.divide (%1324, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1326 = dxgml_op.erf (%1325) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1327 = dxgml_op.add (%1326, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1328 = dxgml_op.multiply (%1324, %1327) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1329 = dxgml_op.multiply (%1328, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1330 = dxgml_op.gemm (%1329, %320) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1331 = dxgml_op.add (%181, %1330) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1332 = dxgml_op.add (%1313, %1331) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1333 = dxgml_op.reduce (%1332) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1334 = dxgml_op.subtract (%1332, %1333) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1335 = dxgml_op.pow (%1334, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1336 = dxgml_op.reduce (%1335) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1337 = dxgml_op.add (%1336, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1338 = dxgml_op.sqrt (%1337) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1339 = dxgml_op.divide (%1334, %1338) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1340 = dxgml_op.multiply (%1339, %188) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1341 = dxgml_op.add (%1340, %189) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1342 = dxgml_op.gemm (%1341, %373) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%1343:3 = dxgml_op.split(%1342) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%1344 = dxgml_op.add (%186, %1343#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1345 = dxgml_op.multiply (%1344, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1346 = dxgml_op.add (%184, %1343#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1347 = dxgml_op.reshape (%1346) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1348 = dxgml_op.transpose (%1347) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1349 = dxgml_op.add (%185, %1343#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1350 = dxgml_op.reshape (%1349) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1351 = dxgml_op.transpose (%1350) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1352 = dxgml_op.reshape (%1345) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1353 = dxgml_op.transpose (%1352) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1354 = dxgml_op.reshape (%1353) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1355 = dxgml_op.reshape (%1348) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1356 = dxgml_op.reshape (%1351) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1357 = dxgml_op.transpose (%1355) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%1358 = dxgml_op.gemm (%1354, %1357) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1359 = dxgml_op.softmax (%1358) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1360 = dxgml_op.gemm (%1359, %1356) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1361 = dxgml_op.reshape (%1360) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1362 = dxgml_op.transpose (%1361) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1363 = dxgml_op.reshape (%1362) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1364 = dxgml_op.gemm (%1363, %321) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1365 = dxgml_op.add (%187, %1364) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1366 = dxgml_op.add (%1332, %1365) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1367 = dxgml_op.reduce (%1366) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1368 = dxgml_op.subtract (%1366, %1367) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1369 = dxgml_op.pow (%1368, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1370 = dxgml_op.reduce (%1369) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1371 = dxgml_op.add (%1370, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1372 = dxgml_op.sqrt (%1371) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1373 = dxgml_op.divide (%1368, %1372) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1374 = dxgml_op.multiply (%1373, %192) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1375 = dxgml_op.add (%1374, %193) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1376 = dxgml_op.gemm (%1375, %322) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1377 = dxgml_op.add (%190, %1376) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1378 = dxgml_op.divide (%1377, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1379 = dxgml_op.erf (%1378) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1380 = dxgml_op.add (%1379, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1381 = dxgml_op.multiply (%1377, %1380) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1382 = dxgml_op.multiply (%1381, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1383 = dxgml_op.gemm (%1382, %323) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1384 = dxgml_op.add (%191, %1383) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1385 = dxgml_op.add (%1366, %1384) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1386 = dxgml_op.reduce (%1385) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1387 = dxgml_op.subtract (%1385, %1386) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1388 = dxgml_op.pow (%1387, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1389 = dxgml_op.reduce (%1388) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1390 = dxgml_op.add (%1389, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1391 = dxgml_op.sqrt (%1390) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1392 = dxgml_op.divide (%1387, %1391) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1393 = dxgml_op.multiply (%1392, %198) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1394 = dxgml_op.add (%1393, %199) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1395 = dxgml_op.gemm (%1394, %374) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%1396:3 = dxgml_op.split(%1395) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%1397 = dxgml_op.add (%196, %1396#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1398 = dxgml_op.multiply (%1397, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1399 = dxgml_op.add (%194, %1396#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1400 = dxgml_op.reshape (%1399) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1401 = dxgml_op.transpose (%1400) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1402 = dxgml_op.add (%195, %1396#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1403 = dxgml_op.reshape (%1402) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1404 = dxgml_op.transpose (%1403) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1405 = dxgml_op.reshape (%1398) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1406 = dxgml_op.transpose (%1405) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1407 = dxgml_op.reshape (%1406) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1408 = dxgml_op.reshape (%1401) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1409 = dxgml_op.reshape (%1404) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1410 = dxgml_op.transpose (%1408) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%1411 = dxgml_op.gemm (%1407, %1410) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1412 = dxgml_op.softmax (%1411) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1413 = dxgml_op.gemm (%1412, %1409) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1414 = dxgml_op.reshape (%1413) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1415 = dxgml_op.transpose (%1414) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1416 = dxgml_op.reshape (%1415) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1417 = dxgml_op.gemm (%1416, %324) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1418 = dxgml_op.add (%197, %1417) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1419 = dxgml_op.add (%1385, %1418) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1420 = dxgml_op.reduce (%1419) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1421 = dxgml_op.subtract (%1419, %1420) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1422 = dxgml_op.pow (%1421, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1423 = dxgml_op.reduce (%1422) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1424 = dxgml_op.add (%1423, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1425 = dxgml_op.sqrt (%1424) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1426 = dxgml_op.divide (%1421, %1425) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1427 = dxgml_op.multiply (%1426, %202) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1428 = dxgml_op.add (%1427, %203) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1429 = dxgml_op.gemm (%1428, %325) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1430 = dxgml_op.add (%200, %1429) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1431 = dxgml_op.divide (%1430, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1432 = dxgml_op.erf (%1431) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1433 = dxgml_op.add (%1432, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1434 = dxgml_op.multiply (%1430, %1433) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1435 = dxgml_op.multiply (%1434, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1436 = dxgml_op.gemm (%1435, %326) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1437 = dxgml_op.add (%201, %1436) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1438 = dxgml_op.add (%1419, %1437) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1439 = dxgml_op.reduce (%1438) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1440 = dxgml_op.subtract (%1438, %1439) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1441 = dxgml_op.pow (%1440, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1442 = dxgml_op.reduce (%1441) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1443 = dxgml_op.add (%1442, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1444 = dxgml_op.sqrt (%1443) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1445 = dxgml_op.divide (%1440, %1444) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1446 = dxgml_op.multiply (%1445, %208) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1447 = dxgml_op.add (%1446, %209) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1448 = dxgml_op.gemm (%1447, %375) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%1449:3 = dxgml_op.split(%1448) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%1450 = dxgml_op.add (%206, %1449#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1451 = dxgml_op.multiply (%1450, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1452 = dxgml_op.add (%204, %1449#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1453 = dxgml_op.reshape (%1452) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1454 = dxgml_op.transpose (%1453) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1455 = dxgml_op.add (%205, %1449#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1456 = dxgml_op.reshape (%1455) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1457 = dxgml_op.transpose (%1456) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1458 = dxgml_op.reshape (%1451) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1459 = dxgml_op.transpose (%1458) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1460 = dxgml_op.reshape (%1459) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1461 = dxgml_op.reshape (%1454) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1462 = dxgml_op.reshape (%1457) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1463 = dxgml_op.transpose (%1461) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%1464 = dxgml_op.gemm (%1460, %1463) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1465 = dxgml_op.softmax (%1464) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1466 = dxgml_op.gemm (%1465, %1462) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1467 = dxgml_op.reshape (%1466) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1468 = dxgml_op.transpose (%1467) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1469 = dxgml_op.reshape (%1468) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1470 = dxgml_op.gemm (%1469, %327) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1471 = dxgml_op.add (%207, %1470) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1472 = dxgml_op.add (%1438, %1471) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1473 = dxgml_op.reduce (%1472) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1474 = dxgml_op.subtract (%1472, %1473) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1475 = dxgml_op.pow (%1474, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1476 = dxgml_op.reduce (%1475) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1477 = dxgml_op.add (%1476, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1478 = dxgml_op.sqrt (%1477) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1479 = dxgml_op.divide (%1474, %1478) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1480 = dxgml_op.multiply (%1479, %212) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1481 = dxgml_op.add (%1480, %213) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1482 = dxgml_op.gemm (%1481, %328) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1483 = dxgml_op.add (%210, %1482) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1484 = dxgml_op.divide (%1483, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1485 = dxgml_op.erf (%1484) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1486 = dxgml_op.add (%1485, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1487 = dxgml_op.multiply (%1483, %1486) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1488 = dxgml_op.multiply (%1487, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1489 = dxgml_op.gemm (%1488, %329) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1490 = dxgml_op.add (%211, %1489) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1491 = dxgml_op.add (%1472, %1490) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1492 = dxgml_op.reduce (%1491) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1493 = dxgml_op.subtract (%1491, %1492) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1494 = dxgml_op.pow (%1493, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1495 = dxgml_op.reduce (%1494) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1496 = dxgml_op.add (%1495, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1497 = dxgml_op.sqrt (%1496) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1498 = dxgml_op.divide (%1493, %1497) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1499 = dxgml_op.multiply (%1498, %218) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1500 = dxgml_op.add (%1499, %219) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1501 = dxgml_op.gemm (%1500, %376) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%1502:3 = dxgml_op.split(%1501) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%1503 = dxgml_op.add (%216, %1502#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1504 = dxgml_op.multiply (%1503, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1505 = dxgml_op.add (%214, %1502#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1506 = dxgml_op.reshape (%1505) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1507 = dxgml_op.transpose (%1506) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1508 = dxgml_op.add (%215, %1502#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1509 = dxgml_op.reshape (%1508) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1510 = dxgml_op.transpose (%1509) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1511 = dxgml_op.reshape (%1504) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1512 = dxgml_op.transpose (%1511) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1513 = dxgml_op.reshape (%1512) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1514 = dxgml_op.reshape (%1507) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1515 = dxgml_op.reshape (%1510) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1516 = dxgml_op.transpose (%1514) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%1517 = dxgml_op.gemm (%1513, %1516) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1518 = dxgml_op.softmax (%1517) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1519 = dxgml_op.gemm (%1518, %1515) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1520 = dxgml_op.reshape (%1519) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1521 = dxgml_op.transpose (%1520) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1522 = dxgml_op.reshape (%1521) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1523 = dxgml_op.gemm (%1522, %330) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1524 = dxgml_op.add (%217, %1523) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1525 = dxgml_op.add (%1491, %1524) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1526 = dxgml_op.reduce (%1525) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1527 = dxgml_op.subtract (%1525, %1526) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1528 = dxgml_op.pow (%1527, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1529 = dxgml_op.reduce (%1528) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1530 = dxgml_op.add (%1529, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1531 = dxgml_op.sqrt (%1530) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1532 = dxgml_op.divide (%1527, %1531) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1533 = dxgml_op.multiply (%1532, %222) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1534 = dxgml_op.add (%1533, %223) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1535 = dxgml_op.gemm (%1534, %331) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1536 = dxgml_op.add (%220, %1535) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1537 = dxgml_op.divide (%1536, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1538 = dxgml_op.erf (%1537) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1539 = dxgml_op.add (%1538, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1540 = dxgml_op.multiply (%1536, %1539) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1541 = dxgml_op.multiply (%1540, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1542 = dxgml_op.gemm (%1541, %332) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1543 = dxgml_op.add (%221, %1542) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1544 = dxgml_op.add (%1525, %1543) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1545 = dxgml_op.reduce (%1544) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1546 = dxgml_op.subtract (%1544, %1545) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1547 = dxgml_op.pow (%1546, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1548 = dxgml_op.reduce (%1547) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1549 = dxgml_op.add (%1548, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1550 = dxgml_op.sqrt (%1549) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1551 = dxgml_op.divide (%1546, %1550) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1552 = dxgml_op.multiply (%1551, %228) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1553 = dxgml_op.add (%1552, %229) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1554 = dxgml_op.gemm (%1553, %377) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%1555:3 = dxgml_op.split(%1554) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%1556 = dxgml_op.add (%226, %1555#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1557 = dxgml_op.multiply (%1556, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1558 = dxgml_op.add (%224, %1555#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1559 = dxgml_op.reshape (%1558) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1560 = dxgml_op.transpose (%1559) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1561 = dxgml_op.add (%225, %1555#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1562 = dxgml_op.reshape (%1561) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1563 = dxgml_op.transpose (%1562) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1564 = dxgml_op.reshape (%1557) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1565 = dxgml_op.transpose (%1564) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1566 = dxgml_op.reshape (%1565) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1567 = dxgml_op.reshape (%1560) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1568 = dxgml_op.reshape (%1563) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1569 = dxgml_op.transpose (%1567) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%1570 = dxgml_op.gemm (%1566, %1569) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1571 = dxgml_op.softmax (%1570) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1572 = dxgml_op.gemm (%1571, %1568) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1573 = dxgml_op.reshape (%1572) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1574 = dxgml_op.transpose (%1573) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1575 = dxgml_op.reshape (%1574) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1576 = dxgml_op.gemm (%1575, %333) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1577 = dxgml_op.add (%227, %1576) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1578 = dxgml_op.add (%1544, %1577) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1579 = dxgml_op.reduce (%1578) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1580 = dxgml_op.subtract (%1578, %1579) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1581 = dxgml_op.pow (%1580, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1582 = dxgml_op.reduce (%1581) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1583 = dxgml_op.add (%1582, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1584 = dxgml_op.sqrt (%1583) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1585 = dxgml_op.divide (%1580, %1584) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1586 = dxgml_op.multiply (%1585, %232) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1587 = dxgml_op.add (%1586, %233) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1588 = dxgml_op.gemm (%1587, %334) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1589 = dxgml_op.add (%230, %1588) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1590 = dxgml_op.divide (%1589, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1591 = dxgml_op.erf (%1590) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1592 = dxgml_op.add (%1591, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1593 = dxgml_op.multiply (%1589, %1592) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1594 = dxgml_op.multiply (%1593, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1595 = dxgml_op.gemm (%1594, %335) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1596 = dxgml_op.add (%231, %1595) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1597 = dxgml_op.add (%1578, %1596) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1598 = dxgml_op.reduce (%1597) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1599 = dxgml_op.subtract (%1597, %1598) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1600 = dxgml_op.pow (%1599, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1601 = dxgml_op.reduce (%1600) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1602 = dxgml_op.add (%1601, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1603 = dxgml_op.sqrt (%1602) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1604 = dxgml_op.divide (%1599, %1603) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1605 = dxgml_op.multiply (%1604, %238) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1606 = dxgml_op.add (%1605, %239) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1607 = dxgml_op.gemm (%1606, %378) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%1608:3 = dxgml_op.split(%1607) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%1609 = dxgml_op.add (%236, %1608#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1610 = dxgml_op.multiply (%1609, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1611 = dxgml_op.add (%234, %1608#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1612 = dxgml_op.reshape (%1611) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1613 = dxgml_op.transpose (%1612) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1614 = dxgml_op.add (%235, %1608#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1615 = dxgml_op.reshape (%1614) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1616 = dxgml_op.transpose (%1615) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1617 = dxgml_op.reshape (%1610) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1618 = dxgml_op.transpose (%1617) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1619 = dxgml_op.reshape (%1618) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1620 = dxgml_op.reshape (%1613) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1621 = dxgml_op.reshape (%1616) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1622 = dxgml_op.transpose (%1620) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%1623 = dxgml_op.gemm (%1619, %1622) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1624 = dxgml_op.softmax (%1623) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1625 = dxgml_op.gemm (%1624, %1621) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1626 = dxgml_op.reshape (%1625) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1627 = dxgml_op.transpose (%1626) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1628 = dxgml_op.reshape (%1627) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1629 = dxgml_op.gemm (%1628, %336) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1630 = dxgml_op.add (%237, %1629) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1631 = dxgml_op.add (%1597, %1630) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1632 = dxgml_op.reduce (%1631) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1633 = dxgml_op.subtract (%1631, %1632) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1634 = dxgml_op.pow (%1633, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1635 = dxgml_op.reduce (%1634) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1636 = dxgml_op.add (%1635, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1637 = dxgml_op.sqrt (%1636) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1638 = dxgml_op.divide (%1633, %1637) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1639 = dxgml_op.multiply (%1638, %242) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1640 = dxgml_op.add (%1639, %243) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1641 = dxgml_op.gemm (%1640, %337) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1642 = dxgml_op.add (%240, %1641) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1643 = dxgml_op.divide (%1642, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1644 = dxgml_op.erf (%1643) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1645 = dxgml_op.add (%1644, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1646 = dxgml_op.multiply (%1642, %1645) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1647 = dxgml_op.multiply (%1646, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1648 = dxgml_op.gemm (%1647, %338) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1649 = dxgml_op.add (%241, %1648) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1650 = dxgml_op.add (%1631, %1649) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1651 = dxgml_op.reduce (%1650) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1652 = dxgml_op.subtract (%1650, %1651) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1653 = dxgml_op.pow (%1652, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1654 = dxgml_op.reduce (%1653) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1655 = dxgml_op.add (%1654, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1656 = dxgml_op.sqrt (%1655) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1657 = dxgml_op.divide (%1652, %1656) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1658 = dxgml_op.multiply (%1657, %248) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1659 = dxgml_op.add (%1658, %249) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1660 = dxgml_op.gemm (%1659, %379) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%1661:3 = dxgml_op.split(%1660) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%1662 = dxgml_op.add (%246, %1661#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1663 = dxgml_op.multiply (%1662, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1664 = dxgml_op.add (%244, %1661#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1665 = dxgml_op.reshape (%1664) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1666 = dxgml_op.transpose (%1665) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1667 = dxgml_op.add (%245, %1661#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1668 = dxgml_op.reshape (%1667) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1669 = dxgml_op.transpose (%1668) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1670 = dxgml_op.reshape (%1663) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1671 = dxgml_op.transpose (%1670) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1672 = dxgml_op.reshape (%1671) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1673 = dxgml_op.reshape (%1666) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1674 = dxgml_op.reshape (%1669) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1675 = dxgml_op.transpose (%1673) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%1676 = dxgml_op.gemm (%1672, %1675) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1677 = dxgml_op.softmax (%1676) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1678 = dxgml_op.gemm (%1677, %1674) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1679 = dxgml_op.reshape (%1678) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1680 = dxgml_op.transpose (%1679) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1681 = dxgml_op.reshape (%1680) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1682 = dxgml_op.gemm (%1681, %339) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1683 = dxgml_op.add (%247, %1682) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1684 = dxgml_op.add (%1650, %1683) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1685 = dxgml_op.reduce (%1684) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1686 = dxgml_op.subtract (%1684, %1685) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1687 = dxgml_op.pow (%1686, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1688 = dxgml_op.reduce (%1687) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1689 = dxgml_op.add (%1688, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1690 = dxgml_op.sqrt (%1689) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1691 = dxgml_op.divide (%1686, %1690) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1692 = dxgml_op.multiply (%1691, %252) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1693 = dxgml_op.add (%1692, %253) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1694 = dxgml_op.gemm (%1693, %340) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1695 = dxgml_op.add (%250, %1694) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1696 = dxgml_op.divide (%1695, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1697 = dxgml_op.erf (%1696) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1698 = dxgml_op.add (%1697, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1699 = dxgml_op.multiply (%1695, %1698) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1700 = dxgml_op.multiply (%1699, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1701 = dxgml_op.gemm (%1700, %341) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1702 = dxgml_op.add (%251, %1701) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1703 = dxgml_op.add (%1684, %1702) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1704 = dxgml_op.reduce (%1703) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1705 = dxgml_op.subtract (%1703, %1704) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1706 = dxgml_op.pow (%1705, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1707 = dxgml_op.reduce (%1706) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1708 = dxgml_op.add (%1707, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1709 = dxgml_op.sqrt (%1708) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1710 = dxgml_op.divide (%1705, %1709) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1711 = dxgml_op.multiply (%1710, %258) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1712 = dxgml_op.add (%1711, %259) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1713 = dxgml_op.gemm (%1712, %380) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%1714:3 = dxgml_op.split(%1713) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%1715 = dxgml_op.add (%256, %1714#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1716 = dxgml_op.multiply (%1715, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1717 = dxgml_op.add (%254, %1714#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1718 = dxgml_op.reshape (%1717) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1719 = dxgml_op.transpose (%1718) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1720 = dxgml_op.add (%255, %1714#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1721 = dxgml_op.reshape (%1720) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1722 = dxgml_op.transpose (%1721) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1723 = dxgml_op.reshape (%1716) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1724 = dxgml_op.transpose (%1723) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1725 = dxgml_op.reshape (%1724) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1726 = dxgml_op.reshape (%1719) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1727 = dxgml_op.reshape (%1722) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1728 = dxgml_op.transpose (%1726) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%1729 = dxgml_op.gemm (%1725, %1728) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1730 = dxgml_op.softmax (%1729) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1731 = dxgml_op.gemm (%1730, %1727) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1732 = dxgml_op.reshape (%1731) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1733 = dxgml_op.transpose (%1732) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1734 = dxgml_op.reshape (%1733) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1735 = dxgml_op.gemm (%1734, %342) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1736 = dxgml_op.add (%257, %1735) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1737 = dxgml_op.add (%1703, %1736) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1738 = dxgml_op.reduce (%1737) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1739 = dxgml_op.subtract (%1737, %1738) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1740 = dxgml_op.pow (%1739, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1741 = dxgml_op.reduce (%1740) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1742 = dxgml_op.add (%1741, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1743 = dxgml_op.sqrt (%1742) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1744 = dxgml_op.divide (%1739, %1743) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1745 = dxgml_op.multiply (%1744, %262) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1746 = dxgml_op.add (%1745, %263) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1747 = dxgml_op.gemm (%1746, %343) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1748 = dxgml_op.add (%260, %1747) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1749 = dxgml_op.divide (%1748, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1750 = dxgml_op.erf (%1749) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1751 = dxgml_op.add (%1750, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1752 = dxgml_op.multiply (%1748, %1751) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1753 = dxgml_op.multiply (%1752, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1754 = dxgml_op.gemm (%1753, %344) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1755 = dxgml_op.add (%261, %1754) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1756 = dxgml_op.add (%1737, %1755) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1757 = dxgml_op.reduce (%1756) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1758 = dxgml_op.subtract (%1756, %1757) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1759 = dxgml_op.pow (%1758, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1760 = dxgml_op.reduce (%1759) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1761 = dxgml_op.add (%1760, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1762 = dxgml_op.sqrt (%1761) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1763 = dxgml_op.divide (%1758, %1762) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1764 = dxgml_op.multiply (%1763, %268) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1765 = dxgml_op.add (%1764, %269) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1766 = dxgml_op.gemm (%1765, %381) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x3072x!dxgml.float16>
%1767:3 = dxgml_op.split(%1766) {axis = #dxgml.integer<2 : !dxgml.int64>} : (!dxgml.tensor<1x31x3072x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>
%1768 = dxgml_op.add (%266, %1767#0) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1769 = dxgml_op.multiply (%1768, %355) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1770 = dxgml_op.add (%264, %1767#1) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1771 = dxgml_op.reshape (%1770) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1772 = dxgml_op.transpose (%1771) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1773 = dxgml_op.add (%265, %1767#2) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1774 = dxgml_op.reshape (%1773) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1775 = dxgml_op.transpose (%1774) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1776 = dxgml_op.reshape (%1769) : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1777 = dxgml_op.transpose (%1776) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1778 = dxgml_op.reshape (%1777) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1779 = dxgml_op.reshape (%1772) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1780 = dxgml_op.reshape (%1775) : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1781 = dxgml_op.transpose (%1779) {permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>} : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x64x31x!dxgml.float16>
%1782 = dxgml_op.gemm (%1778, %1781) : (!dxgml.tensor<16x31x64x!dxgml.float16>, !dxgml.tensor<16x64x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1783 = dxgml_op.softmax (%1782) {axis = -1 : si64} : (!dxgml.tensor<16x31x31x!dxgml.float16>) -> !dxgml.tensor<16x31x31x!dxgml.float16>
%1784 = dxgml_op.gemm (%1783, %1780) : (!dxgml.tensor<16x31x31x!dxgml.float16>, !dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<16x31x64x!dxgml.float16>
%1785 = dxgml_op.reshape (%1784) : (!dxgml.tensor<16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x16x31x64x!dxgml.float16>
%1786 = dxgml_op.transpose (%1785) {permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>} : (!dxgml.tensor<1x16x31x64x!dxgml.float16>) -> !dxgml.tensor<1x31x16x64x!dxgml.float16>
%1787 = dxgml_op.reshape (%1786) : (!dxgml.tensor<1x31x16x64x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1788 = dxgml_op.gemm (%1787, %345) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1789 = dxgml_op.add (%267, %1788) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1790 = dxgml_op.add (%1756, %1789) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1791 = dxgml_op.reduce (%1790) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1792 = dxgml_op.subtract (%1790, %1791) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1793 = dxgml_op.pow (%1792, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1794 = dxgml_op.reduce (%1793) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1795 = dxgml_op.add (%1794, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1796 = dxgml_op.sqrt (%1795) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1797 = dxgml_op.divide (%1792, %1796) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1798 = dxgml_op.multiply (%1797, %272) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1799 = dxgml_op.add (%1798, %273) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1800 = dxgml_op.gemm (%1799, %346) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1801 = dxgml_op.add (%270, %1800) : (!dxgml.tensor<4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1802 = dxgml_op.divide (%1801, %353) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1803 = dxgml_op.erf (%1802) : (!dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1804 = dxgml_op.add (%1803, %349) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1805 = dxgml_op.multiply (%1801, %1804) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x31x4096x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1806 = dxgml_op.multiply (%1805, %354) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x4096x!dxgml.float16>
%1807 = dxgml_op.gemm (%1806, %347) : (!dxgml.tensor<1x31x4096x!dxgml.float16>, !dxgml.tensor<4096x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1808 = dxgml_op.add (%271, %1807) : (!dxgml.tensor<1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1809 = dxgml_op.add (%1790, %1808) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1810 = dxgml_op.reduce (%1809) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1811 = dxgml_op.subtract (%1809, %1810) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1812 = dxgml_op.pow (%1811, %351) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1813 = dxgml_op.reduce (%1812) {axes = #dxgml.dense_integer_elements<[-1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1814 = dxgml_op.add (%1813, %352) : (!dxgml.tensor<1x31x1x!dxgml.float16>, !dxgml.tensor<1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1815 = dxgml_op.sqrt (%1814) : (!dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1x!dxgml.float16>
%1816 = dxgml_op.divide (%1811, %1815) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1x31x1x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1817 = dxgml_op.multiply (%1816, %32) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1818 = dxgml_op.add (%1817, %33) : (!dxgml.tensor<1x31x1024x!dxgml.float16>, !dxgml.tensor<1024x!dxgml.float16>) -> !dxgml.tensor<1x31x1024x!dxgml.float16>
%1819 = dxgml_op.reduce (%1818) {axes = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x31x1024x!dxgml.float16>) -> !dxgml.tensor<1x1024x!dxgml.float16>
%1820 = dxgml_op.gemm (%1819, %348) : (!dxgml.tensor<1x1024x!dxgml.float16>, !dxgml.tensor<1024x6x!dxgml.float16>) -> !dxgml.tensor<1x6x!dxgml.float16>
%1821 = dxgml_op.reshape (%1820) : (!dxgml.tensor<1x6x!dxgml.float16>) -> !dxgml.tensor<1x1x6x!dxgml.float16>
%1822 = dxgml_op.reduce (%1821) {axes = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>, reduction_function = #dxgml_op.reduce_function_enum_attr<reduce_function_average>} : (!dxgml.tensor<1x1x6x!dxgml.float16>) -> !dxgml.tensor<1x6x!dxgml.float16>
dxgml.return %1822 : !dxgml.tensor<1x6x!dxgml.float16>
}
}
#-}