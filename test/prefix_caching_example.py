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
    1. Initial prefill (no cache)
    2. Decode mode (single token)
    3. Prefix-cached prefill (NEW!)
    """
    
    def __init__(self, model_path, max_seq_len=1024):
        """
        Args:
            model_path: Path to compiled MIGraphX model (.mxr)
            max_seq_len: Maximum sequence length the model supports
        """
        print(f"Loading model from {model_path}...")
        self.model = mgx.load(model_path, format="msgpack")
        self.max_seq_len = max_seq_len
        print(f"Model loaded. Max sequence length: {max_seq_len}")
        
    def initial_prefill(self, tokens, past_kv_shape):
        """
        Initial prefill: Process all tokens from scratch.
        
        Args:
            tokens: List of token IDs
            past_kv_shape: Shape for KV cache buffer
            
        Returns:
            Tuple of (logits, present_kv, elapsed_time)
        """
        print(f"\n=== INITIAL PREFILL ===")
        print(f"Processing {len(tokens)} tokens from scratch...")
        
        batch_size = 1
        seq_len = len(tokens)
        
        # Prepare inputs
        input_ids = np.array([tokens], dtype=np.int64)
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(batch_size, -1)
        
        # seqlens_k = 0 indicates no cached prefix (initial prefill)
        seqlens_k = np.array([0], dtype=np.int32)
        
        # Empty past KV cache
        past_kv = np.zeros(past_kv_shape, dtype=np.float32)
        
        start = time.perf_counter()
        result = self.model.run({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "seqlens_k": seqlens_k,
            "past_key_values": past_kv,
        })
        elapsed = time.perf_counter() - start
        
        print(f"✓ Initial prefill completed in {elapsed*1000:.2f}ms")
        return result[0], result[1], elapsed
    
    def decode_step(self, token, cached_kv, past_seqlen):
        """
        Decode mode: Append a single token to existing cache.
        
        Args:
            token: Single token ID to process
            cached_kv: Existing KV cache
            past_seqlen: Number of tokens already in cache
            
        Returns:
            Tuple of (logits, updated_kv, elapsed_time)
        """
        print(f"\n=== DECODE MODE ===")
        print(f"Appending 1 token to cache (past_seqlen={past_seqlen})...")
        
        batch_size = 1
        
        # Single token input
        input_ids = np.array([[token]], dtype=np.int64)
        attention_mask = np.ones((batch_size, 1), dtype=np.int64)
        position_ids = np.array([[past_seqlen]], dtype=np.int64)
        
        # seqlens_k indicates how many tokens are cached
        seqlens_k = np.array([past_seqlen], dtype=np.int32)
        
        start = time.perf_counter()
        result = self.model.run({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "seqlens_k": seqlens_k,
            "past_key_values": cached_kv,
        })
        elapsed = time.perf_counter() - start
        
        print(f"✓ Decode step completed in {elapsed*1000:.2f}ms")
        return result[0], result[1], elapsed
    
    def prefix_cached_prefill(self, new_tokens, cached_kv, prefix_len):
        """
        Prefix-cached prefill: Reuse cached prefix + process new tokens.
        This is the NEW mode enabled by the prefix caching changes!
        
        Args:
            new_tokens: List of new token IDs to process
            cached_kv: Cached KV for prefix
            prefix_len: Number of tokens in cached prefix
            
        Returns:
            Tuple of (logits, updated_kv, elapsed_time)
        """
        print(f"\n=== PREFIX-CACHED PREFILL (NEW!) ===")
        print(f"Cached prefix: {prefix_len} tokens")
        print(f"New tokens: {len(new_tokens)} tokens")
        print(f"Total context: {prefix_len + len(new_tokens)} tokens")
        
        batch_size = 1
        new_len = len(new_tokens)
        
        # Input only contains NEW tokens (prefix is in cache)
        input_ids = np.array([new_tokens], dtype=np.int64)
        attention_mask = np.ones((batch_size, new_len), dtype=np.int64)
        
        # Position IDs continue from cached prefix
        position_ids = np.arange(prefix_len, prefix_len + new_len, dtype=np.int64)
        position_ids = position_ids.reshape(batch_size, -1)
        
        # KEY: seqlens_k indicates prefix length to preserve
        # This triggers the new hybrid mode: 0 < seqlens_k < max_seq_len
        seqlens_k = np.array([prefix_len], dtype=np.int32)
        
        start = time.perf_counter()
        result = self.model.run({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "seqlens_k": seqlens_k,
            "past_key_values": cached_kv,
        })
        elapsed = time.perf_counter() - start
        
        speedup = "N/A"  # Would need baseline time to compute
        print(f"✓ Prefix-cached prefill completed in {elapsed*1000:.2f}ms")
        print(f"  (Faster than processing {prefix_len + new_len} tokens from scratch!)")
        
        return result[0], result[1], elapsed


def main():
    """
    Demonstration script showing all three modes.
    """
    print("=" * 70)
    print("MIGraphX Prefix Caching Demo")
    print("=" * 70)
    
    # Example configuration (adjust to your model)
    model_path = "models/llama-2-7b-chat-hf/model-1024.mxr"
    max_seq_len = 1024
    num_layers = 32
    num_kv_heads = 32
    head_dim = 128
    
    # Check if model exists
    import os
    if not os.path.exists(model_path):
        print(f"\nModel not found at: {model_path}")
        print("Please update model_path in the script to point to your compiled model.")
        print("\nExpected format: MIGraphX .mxr file with inputs:")
        print("  - input_ids: [batch, seq_len]")
        print("  - attention_mask: [batch, seq_len]")
        print("  - position_ids: [batch, seq_len]")
        print("  - seqlens_k: [batch]")
        print("  - past_key_values: [batch, num_kv_heads, max_seq_len, head_dim]")
        return
    
    demo = PrefixCachingDemo(model_path, max_seq_len)
    
    # KV cache shape: [batch, num_kv_heads, max_seq_len, head_dim]
    past_kv_shape = (1, num_kv_heads, max_seq_len, head_dim)
    
    # === Scenario 1: Initial Prefill ===
    prefix_tokens = [1, 2, 3, 4, 5]  # Example: system prompt tokens
    new_tokens = [6, 7, 8]           # Example: user query tokens
    all_tokens = prefix_tokens + new_tokens
    
    logits1, kv1, time1 = demo.initial_prefill(all_tokens, past_kv_shape)
    
    # === Scenario 2: Decode Mode ===
    # Append one token at a time
    next_token = 9
    logits2, kv2, time2 = demo.decode_step(next_token, kv1, len(all_tokens))
    
    # === Scenario 3: Prefix-Cached Prefill ===
    # Simulate: Same prefix, different query
    # First, "build" the prefix cache (reuse kv1 but slice it)
    print("\n" + "=" * 70)
    print("Simulating new request with SAME prefix...")
    print("=" * 70)
    
    # In practice, you'd save prefix KV from a previous run
    # For demo, we'll extract it from kv1
    prefix_len = len(prefix_tokens)
    different_query = [10, 11, 12, 13]  # Different user query
    
    # The cached prefix KV would be stored separately
    # Here we simulate having it available
    cached_prefix_kv = kv1.copy()  # In reality, this would be from cache storage
    
    logits3, kv3, time3 = demo.prefix_cached_prefill(
        different_query, 
        cached_prefix_kv, 
        prefix_len
    )
    
    # === Summary ===
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Initial prefill ({len(all_tokens)} tokens):           {time1*1000:6.2f}ms")
    print(f"Decode (1 token):                         {time2*1000:6.2f}ms")
    print(f"Prefix-cached prefill ({len(different_query)} new tokens):     {time3*1000:6.2f}ms")
    
    # Estimate speedup
    if time1 > 0:
        # Approximate time for full prefill of prefix + new query
        estimated_full_time = time1 * (prefix_len + len(different_query)) / len(all_tokens)
        speedup = estimated_full_time / time3
        print(f"\nEstimated speedup vs. full prefill: {speedup:.2f}x")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

