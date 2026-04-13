// Standalone test: QKV Projection
//   Transpose -> Conv -> Transpose -> Gemm -> Gemm -> Add
//
// Pattern: NHWC input is transposed to NCHW for a pointwise (1x1) convolution
// that expands channels, then transposed back. The expanded features are then
// projected independently for Q and K via two Gemm ops, and their outputs
// are summed (e.g. for a combined attention logit or dual-path fusion).
//
// Dimensions:
//   input NHWC : (1, 4, 4, 16) — batch=1, H=4, W=4, C=16
//   after T1   : (1, 16, 4, 4) — NCHW
//   conv weight: (32, 16, 1, 1) — 1x1 pointwise, 16->32 channels
//   after conv : (1, 32, 4, 4) — NCHW
//   after T2   : (1, 4, 4, 32) — NHWC; each spatial position is a 32-d vector
//   W_q        : (32, 16)      — Q projection
//   W_k        : (32, 16)      — K projection
//   q_proj     : (1, 4, 4, 16) — batched dot: each 32-d -> 16-d
//   k_proj     : (1, 4, 4, 16)
//   out        : (1, 4, 4, 16) — Q + K

module attributes {gpu.container_module} {
  dxgml.module @qkv_projection {
    dxgml.entry_point @qkv_projection(
        %input_nhwc : !dxgml.tensor<1x4x4x16x!dxgml.float16>
    ) -> !dxgml.tensor<1x4x4x16x!dxgml.float16>
    attributes {
      torch.onnx_meta.ir_version = 8 : si64,
      torch.onnx_meta.opset_version = 17 : si64,
      torch.onnx_meta.producer_name = "pytorch",
      torch.onnx_meta.producer_version = "2.0.0"
    } {
      // 1x1 conv weight: expands 16 -> 32 channels
      %conv_w = dxgml_op.constant(#dxgml.constant_resource<conv.weight : !dxgml.tensor<32x16x1x1x!dxgml.float16>>)

      // Q and K projection weights
      %wq = dxgml_op.constant(#dxgml.constant_resource<proj.wq : !dxgml.tensor<32x16x!dxgml.float16>>)
      %wk = dxgml_op.constant(#dxgml.constant_resource<proj.wk : !dxgml.tensor<32x16x!dxgml.float16>>)

      // Step 1: Transpose NHWC -> NCHW  [N,H,W,C] -> [N,C,H,W]  (0,3,1,2)
      %nchw = dxgml_op.transpose(%input_nhwc) {
        permutation = #dxgml.dense_integer_elements<[0, 3, 1, 2]> : !dxgml.tensor<4x!dxgml.int64>
      } : (!dxgml.tensor<1x4x4x16x!dxgml.float16>)
        -> !dxgml.tensor<1x16x4x4x!dxgml.float16>

      // Step 2: Pointwise 1x1 convolution — channel expansion 16 -> 32
      //   stride=1, pad=0, 1x1 kernel: spatial unchanged (4x4 -> 4x4)
      %conv_out = dxgml_op.convolution(%nchw, %conv_w) {
        group_count   = #dxgml.integer<1 : !dxgml.int64>,
        dilations     = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
        start_padding = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>,
        end_padding   = #dxgml.dense_integer_elements<[0, 0]> : !dxgml.tensor<2x!dxgml.int64>,
        strides       = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
      } : (!dxgml.tensor<1x16x4x4x!dxgml.float16>,
           !dxgml.tensor<32x16x1x1x!dxgml.float16>)
        -> !dxgml.tensor<1x32x4x4x!dxgml.float16>

      // Step 3: Transpose NCHW -> NHWC  [N,C,H,W] -> [N,H,W,C]  (0,2,3,1)
      //   Each spatial position (1,4,4,32) is now a 32-d feature vector
      %features = dxgml_op.transpose(%conv_out) {
        permutation = #dxgml.dense_integer_elements<[0, 2, 3, 1]> : !dxgml.tensor<4x!dxgml.int64>
      } : (!dxgml.tensor<1x32x4x4x!dxgml.float16>)
        -> !dxgml.tensor<1x4x4x32x!dxgml.float16>

      // Step 4: Q projection — batched dot over (H,W) positions
      //   (1,4,4,32) @ (32,16) -> (1,4,4,16)
      //   Weight is unsqueeze+broadcast to (1,4,32,16) before dot
      %q_proj = dxgml_op.gemm(%features, %wq)
        : (!dxgml.tensor<1x4x4x32x!dxgml.float16>,
           !dxgml.tensor<32x16x!dxgml.float16>)
        -> !dxgml.tensor<1x4x4x16x!dxgml.float16>

      // Step 5: K projection — same shape, independent weights
      //   (1,4,4,32) @ (32,16) -> (1,4,4,16)
      %k_proj = dxgml_op.gemm(%features, %wk)
        : (!dxgml.tensor<1x4x4x32x!dxgml.float16>,
           !dxgml.tensor<32x16x!dxgml.float16>)
        -> !dxgml.tensor<1x4x4x16x!dxgml.float16>

      // Step 6: Add Q + K  (e.g. combined attention logit or dual-path fusion)
      %result = dxgml_op.add(%q_proj, %k_proj)
        : (!dxgml.tensor<1x4x4x16x!dxgml.float16>,
           !dxgml.tensor<1x4x4x16x!dxgml.float16>)
        -> !dxgml.tensor<1x4x4x16x!dxgml.float16>

      dxgml.return %result : !dxgml.tensor<1x4x4x16x!dxgml.float16>
    }
  }
  gpu.module @rock_gpu_module {
  }
}
