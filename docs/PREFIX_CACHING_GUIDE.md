# Prefix Caching Support for LLM Inference

## Overview

Prefix caching is an optimization technique that allows reusing KV cache states for common prompt prefixes across multiple inference requests. This significantly reduces computation time when:

- Using the same system prompt across multiple user queries
- Processing few-shot examples that remain constant
- Maintaining chat history that's shared across conversation turns

## What Changed

MIGraphX now supports **hybrid mode** for `GroupQueryAttention` and `concat_past_present` operations, enabling three modes of operation:

| Mode | `sequence_length` | `seqlens_k` | Behavior |
|------|-------------------|-------------|----------|
| **Initial Prefill** | > 1 | 0 or max_seq_len | Process all tokens from scratch, no cached KV |
| **Decode** | 1 | N (cached length) | Append 1 new token to existing KV cache |
| **Prefix-Cached Prefill** | > 1 | N (0 < N < max) | Reuse N cached prefix tokens + process new tokens |

### Modified Components

1. **`concat_past_present` operator** (`src/include/migraphx/op/concat_past_present.hpp`)
   - Now uses `seqlens_k[batch_index]` to determine actual cached prefix length
   - Preserves cached KV when `0 < seqlens_k < max_seq_len` during prefill mode

2. **GPU kernel** (`src/targets/gpu/kernels/include/migraphx/kernels/concat_past_present.hpp`)
   - Matches CPU implementation for consistency

3. **GroupQueryAttention masking** (`src/onnx/parse_group_query_attention.cpp`)
   - Causal mask now offset by `past_sl` to account for cached prefix positions
   - Window attention correctly handles absolute positions with prefix

## How It Works

### Traditional Flow (Before Prefix Caching)

**Request 1:**
```
Prompt: "You are a helpful AI. What is 2+2?"
- sequence_length = 10 tokens
- seqlens_k = 0 (no cache)
- Result: Process all 10 tokens → Generate KV cache for positions [0-9]
```

**Request 2:**
```
Prompt: "You are a helpful AI. What is 3+3?"
- sequence_length = 10 tokens
- seqlens_k = 0 (cache not reused)
- Result: Process all 10 tokens again from scratch
```

### With Prefix Caching (After)

**Request 1:**
```
Prefix: "You are a helpful AI. " (5 tokens)
New: "What is 2+2?" (5 tokens)
- sequence_length = 10 tokens
- seqlens_k = 0 (initial prefill)
- Result: Process all 10 tokens → Save KV cache
- Store prefix KV for positions [0-4] separately
```

**Request 2:**
```
Prefix: "You are a helpful AI. " (5 tokens) ← CACHED!
New: "What is 3+3?" (5 tokens)
- sequence_length = 5 tokens (only new tokens)
- seqlens_k = 5 (cached prefix length)
- Result: 
  1. Load cached KV for positions [0-4]
  2. Process only 5 new tokens
  3. Attention mask allows new tokens to attend to [0-4] + themselves with causal masking
  4. KV cache becomes positions [0-9] with prefix reused
```

### Attention Mask Behavior

With prefix caching, attention patterns for new tokens are:

```
Cached Prefix: [0, 1, 2, 3, 4]
New Tokens:    [5, 6, 7]

Attention matrix (✓ = can attend, ✗ = masked):
       KV Positions
       0 1 2 3 4 | 5 6 7
Q   5  ✓ ✓ ✓ ✓ ✓ | ✓ ✗ ✗  ← Token 5 sees all prefix + itself
u   6  ✓ ✓ ✓ ✓ ✓ | ✓ ✓ ✗  ← Token 6 sees all prefix + [5,6]
e   7  ✓ ✓ ✓ ✓ ✓ | ✓ ✓ ✓  ← Token 7 sees all prefix + [5,6,7]
r
y
```

The key change: Causal mask is now offset by `past_sl` (prefix length), so new tokens treat cached positions as fully visible while maintaining causality among themselves.

## Usage Pattern

### ONNX Model Requirements

Your ONNX model should use `GroupQueryAttention` with:
- **Input `seqlens_k`**: Scalar or batch-size tensor indicating cached prefix length
- **Past KV inputs**: Buffers containing cached states
- **Output KV states**: Updated cache to reuse in subsequent calls

### Setting `seqlens_k` Values

The `seqlens_k` input controls prefix caching behavior:

```python
# Initial prefill (no cache)
seqlens_k = np.array([0], dtype=np.int32)
# OR
seqlens_k = np.array([max_seq_len], dtype=np.int32)

# Decode mode (append 1 token)
seqlens_k = np.array([current_cached_length], dtype=np.int32)
# Example: If cache has 50 tokens
seqlens_k = np.array([50], dtype=np.int32)

# Prefix-cached prefill (NEW!)
# Cache has 20 tokens, processing 10 new tokens
seqlens_k = np.array([20], dtype=np.int32)
```

### Python Example

```python
import numpy as np
import migraphx as mgx

class LLMWithPrefixCache:
    def __init__(self, model_path, max_seq_len=2048):
        self.model = mgx.load(model_path)
        self.max_seq_len = max_seq_len
        
        # Prefix cache storage: maps prefix hash -> (KV cache, length)
        self.prefix_cache = {}
        
    def _hash_prefix(self, tokens):
        """Create hash for prefix tokens"""
        return hash(tuple(tokens))
    
    def generate_with_prefix(self, prefix_tokens, new_tokens):
        """
        Generate with cached prefix.
        
        Args:
            prefix_tokens: List[int] - Common prefix to cache
            new_tokens: List[int] - New tokens to process
        """
        prefix_hash = self._hash_prefix(prefix_tokens)
        prefix_len = len(prefix_tokens)
        new_len = len(new_tokens)
        
        # Check if prefix is cached
        if prefix_hash in self.prefix_cache:
            cached_kv, cached_len = self.prefix_cache[prefix_hash]
            print(f"Using cached prefix: {cached_len} tokens")
            
            # Prepare inputs for prefix-cached prefill
            input_ids = np.array([new_tokens], dtype=np.int64)
            seqlens_k = np.array([cached_len], dtype=np.int32)
            
            # Attention mask: [1] * new_len for new tokens
            # (prefix is implicitly visible via cached KV)
            attention_mask = np.ones((1, new_len), dtype=np.int64)
            
            # Position IDs: Continue from cached prefix
            position_ids = np.arange(cached_len, cached_len + new_len, dtype=np.int64)
            position_ids = position_ids.reshape(1, -1)
            
            # Run inference with cached KV
            outputs = self.model.run({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "seqlens_k": seqlens_k,
                "past_key_values": cached_kv,  # Reuse cached KV!
            })
            
        else:
            # No cache available - do full prefill
            print(f"No cache found, full prefill: {prefix_len + new_len} tokens")
            
            all_tokens = prefix_tokens + new_tokens
            input_ids = np.array([all_tokens], dtype=np.int64)
            seqlens_k = np.array([0], dtype=np.int32)  # No cache
            attention_mask = np.ones((1, len(all_tokens)), dtype=np.int64)
            position_ids = np.arange(len(all_tokens), dtype=np.int64).reshape(1, -1)
            
            outputs = self.model.run({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "seqlens_k": seqlens_k,
                "past_key_values": np.zeros((1, num_heads, self.max_seq_len, head_dim)),
            })
            
            # Extract and cache prefix KV
            present_kv = outputs["present_key_values"]
            self.prefix_cache[prefix_hash] = (present_kv, prefix_len)
        
        return outputs

# Usage
llm = LLMWithPrefixCache("llama2-7b.mxr", max_seq_len=2048)

# System prompt (cached)
system_prompt = tokenizer.encode("You are a helpful AI assistant.")

# Multiple queries reusing the same prefix
query1 = tokenizer.encode("What is the capital of France?")
query2 = tokenizer.encode("What is 2+2?")
query3 = tokenizer.encode("Tell me a joke.")

# First call: full prefill (slow)
result1 = llm.generate_with_prefix(system_prompt, query1)

# Subsequent calls: reuse cached prefix (FAST!)
result2 = llm.generate_with_prefix(system_prompt, query2)  # ~2x faster
result3 = llm.generate_with_prefix(system_prompt, query3)  # ~2x faster
```

## Performance Benefits

Expected speedup depends on prefix/new token ratio:

| Scenario | Prefix Tokens | New Tokens | Expected Speedup |
|----------|---------------|------------|------------------|
| Short system prompt | 50 | 200 | ~1.2x |
| Long system prompt | 500 | 200 | ~2.5x |
| Few-shot examples | 1000 | 100 | ~5x |
| Chat with history | 1500 | 50 | ~10x |

## Limitations & Considerations

1. **Cache invalidation**: You must manage cache eviction when memory is limited
2. **Exact prefix matching**: Even 1 token difference invalidates the cache
3. **Batch processing**: Current implementation supports per-batch caching
4. **Memory overhead**: Each cached prefix consumes `(num_layers × num_kv_heads × prefix_len × head_dim × sizeof(dtype))` bytes

## Testing

To verify prefix caching works correctly:

```python
import numpy as np

def test_prefix_caching():
    """Verify that prefix-cached results match full prefill results"""
    
    # Full prefill
    full_input = [1, 2, 3, 4, 5, 6, 7]
    output_full = model.run({
        "input_ids": np.array([full_input]),
        "seqlens_k": np.array([0]),
        ...
    })
    
    # Prefix-cached: Cache [1,2,3,4], process [5,6,7]
    prefix = [1, 2, 3, 4]
    new_tokens = [5, 6, 7]
    
    # Step 1: Build prefix cache
    output_prefix = model.run({
        "input_ids": np.array([prefix]),
        "seqlens_k": np.array([0]),
        ...
    })
    prefix_kv = output_prefix["present_key_values"]
    
    # Step 2: Use prefix cache
    output_cached = model.run({
        "input_ids": np.array([new_tokens]),
        "seqlens_k": np.array([len(prefix)]),  # Indicate cached length!
        "past_key_values": prefix_kv,
        ...
    })
    
    # Results should be identical
    assert np.allclose(output_full["logits"][:, -3:, :], 
                      output_cached["logits"], 
                      rtol=1e-5)
    print("✓ Prefix caching produces correct results!")
```

## Debugging

If prefix caching isn't working:

1. **Check `seqlens_k` values**:
   ```python
   print(f"seqlens_k={seqlens_k}, sequence_length={len(input_ids[0])}")
   # Should be: 0 < seqlens_k < max_seq_len for prefix caching
   ```

2. **Verify attention mask shape**:
   ```python
   # Should match new tokens length, not total length
   assert attention_mask.shape[1] == len(new_tokens)
   ```

3. **Confirm position IDs are offset**:
   ```python
   # Position IDs should continue from cached prefix
   expected_pos = np.arange(prefix_len, prefix_len + new_len)
   assert np.array_equal(position_ids[0], expected_pos)
   ```

4. **Enable verbose logging** (if available):
   ```bash
   MIGRAPHX_TRACE_CONCAT_PAST_PRESENT=1 python your_script.py
   ```

## Migration from Non-Caching Code

If you have existing inference code:

**Before (no caching):**
```python
result = model.run({
    "input_ids": full_prompt_tokens,
    "seqlens_k": np.array([0]),
})
```

**After (with prefix caching):**
```python
# Split prompt into prefix + new
prefix, new = split_prompt(full_prompt_tokens)

# Use cached prefix
result = model.run({
    "input_ids": new,
    "seqlens_k": np.array([len(prefix)]),
    "past_key_values": cached_prefix_kv,
})
```

## Future Enhancements

Potential improvements:

1. **Automatic prefix detection**: Detect common prefixes across requests
2. **Partial prefix matching**: Use longest matching prefix
3. **Multi-level caching**: Cache at multiple granularities
4. **Distributed cache**: Share prefix cache across multiple GPUs/nodes

## References

- Original ONNX GroupQueryAttention spec
- MIGraphX attention fusion documentation
- KV cache optimization best practices

