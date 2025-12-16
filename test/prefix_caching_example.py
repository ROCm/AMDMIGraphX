#!/usr/bin/env python3
"""
Example demonstrating prefix caching usage with MIGraphX.

This script shows how to leverage the new prefix caching support in 
GroupQueryAttention for faster LLM inference with repeated prefixes.
"""

import numpy as np
import sys
import time

try:
    import migraphx as mgx
except ImportError:
    print("MIGraphX Python bindings not found. Please install migraphx.")
    sys.exit(1)


class PrefixCachingDemo:
    """
    Demonstrates three inference modes:
    1. Prefix-cached prefill
    2. No prefix-cached prefill
    """
    
    def __init__(self, model_path, no_prefix_cached_prefill_model_path, max_seq_len=1024):
        """
        Args:
            model_path: Path to compiled MIGraphX model (.mxr)
            max_seq_len: Maximum sequence length the model supports
        """
        print(f"Loading model from {model_path}...")
        self.model = mgx.load(model_path, format="msgpack")
        print(f"Loading model from {no_prefix_cached_prefill_model_path}...")
        self.no_prefix_cached_prefill_model = mgx.load(no_prefix_cached_prefill_model_path, format="msgpack")
        self.max_seq_len = max_seq_len
        self.attention_mask = None
        print(f"Model loaded. Max sequence length: {max_seq_len}")
        
    def initial_prefill(self, past_kv_shape):
        """
        Initial prefill: Process all tokens from scratch.
        
        Args:
            tokens: List of token IDs
            past_kv_shape: Shape for KV cache buffer
            
        Returns:
            Tuple of (logits, present_kv, elapsed_time)
        """
        print(f"\n=== INITIAL PREFILL ===")
        print(f"Processing 1500 tokens from scratch...")
        
        batch_size = 1
        input_ids = np.ones((batch_size, 1500), dtype=np.int64)
        self.attention_mask = np.zeros((batch_size, self.max_seq_len), dtype=np.int64)
        self.attention_mask[0, 0] = 1
        past_kv = np.zeros(past_kv_shape, dtype=np.float16)
        kv_buffer = {}
        for i in range(24):
            kv_buffer[f"past_key_values.{i}.key"] = mgx.to_gpu(mgx.argument(past_kv.copy()))
            kv_buffer[f"past_key_values.{i}.value"] = mgx.to_gpu(mgx.argument(past_kv.copy()))
        
        print("input_ids shape:", input_ids.shape)
        print("attention_mask shape:", self.attention_mask.shape)
        print("past_key_values shape:", past_kv.shape)

        gpu_input_ids = mgx.to_gpu(mgx.argument(input_ids))
        gpu_attention_mask = mgx.to_gpu(mgx.argument(self.attention_mask))
        output_shape = mgx.shape(type="half_type", lens=[batch_size, 1500, 151936])
        output_gpu = mgx.allocate_gpu(output_shape)
        print("output_shape:", output_shape)
        
        start = time.perf_counter()
        result = self.model.run({
            "input_ids": gpu_input_ids,
            "attention_mask": gpu_attention_mask,
            **kv_buffer,
            "main:#output_0": output_gpu,
        })
        # update attention mask for next step
        self.attention_mask[:, :1500] = 1
        elapsed = time.perf_counter() - start
        
        print(f"✓ Prefix-cached prefill completed in {elapsed*1000:.2f}ms")

        return output_gpu, kv_buffer, elapsed, input_ids
    
    def next_input(self, kv_buffer, past_kv_shape):
        """
        Next input: Process new tokens after the initial prefill.
        
        Args:
            kv_buffer: Cached KV buffer from the initial prefill
            past_kv_shape: Shape for KV cache buffer
            
        Returns:
            Tuple of (logits, present_kv, elapsed_time)
        """
        
        batch_size = 1
        input_ids = np.ones((batch_size, 1500), dtype=np.int64)
        self.attention_mask[0, 1500] = 1
        
        print("input_ids shape:", input_ids.shape)
        print("attention_mask shape:", self.attention_mask.shape)

        gpu_input_ids = mgx.to_gpu(mgx.argument(input_ids))
        gpu_attention_mask = mgx.to_gpu(mgx.argument(self.attention_mask))
        output_shape = mgx.shape(type="half_type", lens=[batch_size, 1500, 151936])
        output_gpu = mgx.allocate_gpu(output_shape)
        print("output_shape:", output_shape)
        
        start = time.perf_counter()
        result = self.model.run({
            "input_ids": gpu_input_ids,
            "attention_mask": gpu_attention_mask,
            **kv_buffer,
            "main:#output_0": output_gpu,
        })
        # update attention mask for next step
        self.attention_mask[:, :3000] = 1
        elapsed = time.perf_counter() - start
        
        print(f"✓ Next input completed in {elapsed*1000:.2f}ms")

        return output_gpu, kv_buffer, elapsed
    
    def no_prefix_cached_prefill(self, past_kv_shape):
        """
        No prefix-cached prefill: Process new tokens without cached prefix.
        
        Args:
            past_kv_shape: Shape for KV cache buffer
            
        Returns:
            Tuple of (logits, updated_kv, elapsed_time)
        """

        print(f"\n=== NO PREFIX-CACHED PREFILL ===")
        print(f"Processing 3000 tokens from scratch...")

        batch_size = 1
        input_ids = np.ones((batch_size, 3000), dtype=np.int64)
        self.attention_mask = np.zeros((batch_size, self.max_seq_len), dtype=np.int64)
        self.attention_mask[0, 0] = 1
        #Empty past KV cache
        past_kv = np.zeros(past_kv_shape, dtype=np.float16)
        kv_buffer = {}
        for i in range(24):
            kv_buffer[f"past_key_values.{i}.key"] = mgx.to_gpu(mgx.argument(past_kv.copy()))
            kv_buffer[f"past_key_values.{i}.value"] = mgx.to_gpu(mgx.argument(past_kv.copy()))
        
        print("input_ids shape:", input_ids.shape)
        print("attention_mask shape:", self.attention_mask.shape)
        print("past_key_values shape:", past_kv.shape)

        gpu_input_ids = mgx.to_gpu(mgx.argument(input_ids))
        gpu_attention_mask = mgx.to_gpu(mgx.argument(self.attention_mask))
        output_shape = mgx.shape(type="half_type", lens=[batch_size, 3000, 151936])
        output_gpu = mgx.allocate_gpu(output_shape)
        print("output_shape:", output_shape)
        
        start = time.perf_counter()
        result = self.no_prefix_cached_prefill_model.run({
            "input_ids": gpu_input_ids,
            "attention_mask": gpu_attention_mask,
            **kv_buffer,
            "main:#output_0": output_gpu,
        })
        # update attention mask for next step
        self.attention_mask[:, :3000] = 1
        elapsed = time.perf_counter() - start
        
        print(f"✓ No prefix-cached prefill completed in {elapsed*1000:.2f}ms")

        return output_gpu, kv_buffer, elapsed, input_ids

def main():
    """
    Demonstration script showing all three modes.
    """
    print("=" * 70)
    print("MIGraphX Prefix Caching Demo")
    print("=" * 70)
    
    # Example configuration (adjust to your model)
    import argparse
    parser = argparse.ArgumentParser(description="MIGraphX Prefix Caching Demo")
    parser.add_argument("--prefix_cached_prefill_model_path", type=str, required=False,
                        help="Path to the compiled MIGraphX prefix-cached prefill model (.mxr)")
    parser.add_argument("--no_prefix_cached_prefill_model_path", type=str, required=False,
                        help="Path to the compiled MIGraphX no-prefix-cached prefill model (.mxr)")
    args = parser.parse_args()
    prefix_cached_prefill_model_path = args.prefix_cached_prefill_model_path or "/data/alibaba_small_llm/migraphx_model/qwen_pc_1500.mxr"
    no_prefix_cached_prefill_model_path = args.no_prefix_cached_prefill_model_path or "/data/alibaba_small_llm/migraphx_model/qwen_pc_3000.mxr"
    max_seq_len = 4096
    num_kv_heads = 2
    head_dim = 64
    
    # Check if model exists
    import os
    if not os.path.exists(prefix_cached_prefill_model_path):
        print(f"\nModel not found at: {prefix_cached_prefill_model_path}")
        print("Please update model_path in the script to point to your compiled model.")
        print("\nExpected format: MIGraphX .mxr file with inputs:")
        print("  - input_ids: [batch, seq_len]")
        print("  - attention_mask: [batch, seq_len]")
        print("  - past_key_values: [batch, num_kv_heads, max_seq_len, head_dim]")
        return
    
    demo = PrefixCachingDemo(prefix_cached_prefill_model_path, no_prefix_cached_prefill_model_path, max_seq_len)
    
    # KV cache shape: [batch, num_kv_heads, max_seq_len, head_dim]
    past_kv_shape = (1, num_kv_heads, max_seq_len, head_dim)
    
    # === Scenario 1: Prefix-Cached Prefill ===
    _logits1, kv1, time1, input_ids1 = demo.initial_prefill(past_kv_shape)
    
    # === Scenario 1-1: Next Input tokens ===
    # In practice, you'd save prefix KV from the previous run
    # For demo, we'll use the kv1 as the prefix KV
    _logits2, _kv2, time2 = demo.next_input(kv1, past_kv_shape)

    logits2_arr = np.array(mgx.from_gpu(_logits2))
    logits2_last = logits2_arr[0, -1, :] # last token
    
    # === Scenario 2: No Prefix-Cached Prefill ===
    print("\n" + "=" * 70)
    print("Simulating the request without prefix caching...")
    print("=" * 70)
    
    _logits3, _kv3, time3, input_ids3 = demo.no_prefix_cached_prefill(past_kv_shape)

    logits3_arr = np.array(mgx.from_gpu(_logits3))
    logits3_last = logits3_arr[0, -1, :] # last token
    
    # === Summary ===
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Initial prefill (tokens:1500):              {time1*1000:6.2f}ms")
    print(f"Next input (another tokens:1500):                    {time2*1000:6.2f}ms")
    print(f"No prefix-cached prefill (total tokens:3000):    {time3*1000:6.2f}ms")
    
    # Estimate speedup
    speedup = time3 / time2
    print(f"\nThe maximum estimated full prefix caching speedup: {speedup:.2f}x")
    print("\n" + "=" * 70)
    print("ACCURACY:(Should be the same for both last tokens)")
    print("=" * 70)
    print("logits2 (Prefix-Cached Prefill):", logits2_last)
    print("logits3 (No Prefix-Cached Prefill):    ", logits3_last)
    diff = np.abs(logits2_last - logits3_last)
    print("abs diff logits2_last vs logits3_last:", diff)
    print("max abs diff:", np.max(diff))
    print("mean abs diff:", np.mean(diff))
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

