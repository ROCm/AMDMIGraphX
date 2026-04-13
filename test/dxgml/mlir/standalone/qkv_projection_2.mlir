// Standalone test: QKV Projection 2 (phi_silica architecture dimensions)
//
// Dual-path QKV projection:
//   Base path : NHWC → T[0,3,1,2] → 1x1 Conv (hidden→qkv_proj) → T[0,2,3,1]
//   LoRA path : Gemm(lora_A: hidden→rank) → Gemm(lora_B: rank→qkv_proj)
//   Output    : Add(base, lora) → [1,1,1,9216]
//
// This pattern appears in phi-silica and similar models where a low-rank
// LoRA adaptation is fused on top of a base linear projection implemented
// as a 1x1 convolution.
//
// Dimensions (phi_silica):
//   input NHWC : (1, 1, 1, 3072) — batch=1, H=1, W=1, C=3072 (hidden dim)
//   conv_w     : (9216, 3072, 1, 1) — 1x1 pointwise, hidden → qkv_proj
//   lora_a     : (3072, 32)  — rank-32 LoRA down-projection
//   lora_b     : (32, 9216)  — rank-32 LoRA up-projection
//   output     : (1, 1, 1, 9216)

module attributes {gpu.container_module} {
  dxgml.module @qkv_projection_2 {
    dxgml.entry_point @qkv_projection_2(
        %input_nhwc : !dxgml.tensor<1x1x1x3072x!dxgml.float16>
    ) -> !dxgml.tensor<1x1x1x9216x!dxgml.float16>
    attributes {
      torch.onnx_meta.ir_version = 8 : si64,
      torch.onnx_meta.opset_version = 17 : si64,
      torch.onnx_meta.producer_name = "pytorch",
      torch.onnx_meta.producer_version = "2.0.0"
    } {
      // Weights
      %conv_w = dxgml_op.constant(#dxgml.constant_resource<qkv.conv_weight : !dxgml.tensor<9216x3072x1x1x!dxgml.float16>>)
      %lora_a = dxgml_op.constant(#dxgml.constant_resource<qkv.lora_a : !dxgml.tensor<3072x32x!dxgml.float16>>)
      %lora_b = dxgml_op.constant(#dxgml.constant_resource<qkv.lora_b : !dxgml.tensor<32x9216x!dxgml.float16>>)

      // ---- Base path: 1x1 convolution branch ----

      // Step 1: Transpose NHWC -> NCHW  [N,H,W,C] -> [N,C,H,W]  (0,3,1,2)
      %nchw = dxgml_op.transpose(%input_nhwc) {
        permutation = #dxgml.dense_integer_elements<[0, 3, 1, 2]> : !dxgml.tensor<4x!dxgml.int64>
      } : (!dxgml.tensor<1x1x1x3072x!dxgml.float16>)
        -> !dxgml.tensor<1x3072x1x1x!dxgml.float16>

      // Step 2: 1x1 conv — channel expansion 3072 -> 9216 (QKV projection)
      //   stride=1, pad=0, spatial 1x1 -> 1x1 unchanged
      %conv_out = dxgml_op.convolution(%nchw, %conv_w) {
        group_count   = #dxgml.integer<1 : !dxgml.int64>,
        dilations     = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        start_padding = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>,
        end_padding   = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>,
        strides       = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
      } : (!dxgml.tensor<1x3072x1x1x!dxgml.float16>,
           !dxgml.tensor<9216x3072x1x1x!dxgml.float16>)
        -> !dxgml.tensor<1x9216x1x1x!dxgml.float16>

      // Step 3: Transpose NCHW -> NHWC  [N,C,H,W] -> [N,H,W,C]  (0,2,3,1)
      %base_out = dxgml_op.transpose(%conv_out) {
        permutation = #dxgml.dense_integer_elements<[0, 2, 3, 1]> : !dxgml.tensor<4x!dxgml.int64>
      } : (!dxgml.tensor<1x9216x1x1x!dxgml.float16>)
        -> !dxgml.tensor<1x1x1x9216x!dxgml.float16>

      // ---- LoRA path: low-rank adaptation ----

      // Step 4: LoRA down-projection  (1,1,1,3072) @ (3072,32) -> (1,1,1,32)
      %lora_down = dxgml_op.gemm(%input_nhwc, %lora_a)
        : (!dxgml.tensor<1x1x1x3072x!dxgml.float16>,
           !dxgml.tensor<3072x32x!dxgml.float16>)
        -> !dxgml.tensor<1x1x1x32x!dxgml.float16>

      // Step 5: LoRA up-projection  (1,1,1,32) @ (32,9216) -> (1,1,1,9216)
      %lora_up = dxgml_op.gemm(%lora_down, %lora_b)
        : (!dxgml.tensor<1x1x1x32x!dxgml.float16>,
           !dxgml.tensor<32x9216x!dxgml.float16>)
        -> !dxgml.tensor<1x1x1x9216x!dxgml.float16>

      // Step 6: Merge — add base conv output and LoRA contribution
      %result = dxgml_op.add(%base_out, %lora_up)
        : (!dxgml.tensor<1x1x1x9216x!dxgml.float16>,
           !dxgml.tensor<1x1x1x9216x!dxgml.float16>)
        -> !dxgml.tensor<1x1x1x9216x!dxgml.float16>

      dxgml.return %result : !dxgml.tensor<1x1x1x9216x!dxgml.float16>
    }
  }
  gpu.module @rock_gpu_module {
  }
}
