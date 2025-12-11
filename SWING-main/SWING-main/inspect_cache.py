import pickle
import numpy as np
import os

cache_path = 'Data/protbert_cache.pkl'

if not os.path.exists(cache_path):
    print("Cache file not found.")
    exit()

with open(cache_path, 'rb') as f:
    cache = pickle.load(f)

print(f"Number of entries in cache: {len(cache)}")

keys = list(cache.keys())
if not keys:
    print("Cache is empty.")
    exit()

print(f"First 5 keys: {keys[:5]}")

# Check shapes and values
first_val = cache[keys[0]]
print(f"Shape of first embedding: {first_val.shape}")
print(f"First 10 values of first embedding: {first_val[:10]}")

# Check for zeros
is_zero = np.all(first_val == 0)
print(f"Is first embedding all zeros? {is_zero}")

# Check for duplicates
if len(keys) > 1:
    second_val = cache[keys[1]]
    is_same = np.allclose(first_val, second_val)
    print(f"Are first two embeddings identical? {is_same}")
    
    # Check if all are same
    all_same = True
    for k in keys[1:100]:
        if not np.allclose(first_val, cache[k]):
            all_same = False
            break
    print(f"Are first 100 embeddings identical? {all_same}")

# Check variance
vals = np.array([cache[k] for k in keys[:100]])
print(f"Variance of first 100 embeddings (mean): {np.var(vals)}")
