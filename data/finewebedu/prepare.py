# saves the fineweb-edu dataset to a binary file for training
# adapted from openwebtext preprocessing

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 64

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    dataset_dir = "/jfs-dialogue-mmos02-rs02/users/Zp/data/fineweb-edu"
    cache_dir = "/jfs-dialogue-mmos02-rs02/users/Zp/.cache/huggingface/datasets"
    
    dataset = load_dataset(dataset_dir, num_proc=num_proc_load_dataset, cache_dir=cache_dir)
    
    print(f"Dataset loaded: {dataset}")
    print(f"Number of examples: {len(dataset['train'])}")  # type: ignore

    # Create train/val split since FineWeb-Edu only has 'train' split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)  # type: ignore
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    print("Dataset split:")
    print(split_dataset)

    # tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        # FineWeb-Edu has 'text' field like OpenWebText
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    print("Tokenizing dataset...")
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],  # Remove text column and keep any other metadata if needed
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        print(f"\nProcessing {split} split...")
        
        # Calculate total length first
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)

        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        # Use much larger batch size for better performance
        batch_size = max(1000, len(dset) // 100)  # Adaptive batch size
        idx = 0
        
        for i in tqdm(range(0, len(dset), batch_size), desc=f'writing {filename}'):
            # Process batch
            batch_end = min(i + batch_size, len(dset))
            batch_ids = []
            for j in range(i, batch_end):
                batch_ids.extend(dset[j]['ids'])
            
            # Convert to numpy array and write
            batch_array = np.array(batch_ids, dtype=dtype)
            arr[idx:idx + len(batch_array)] = batch_array
            idx += len(batch_array)
        
        arr.flush()

    # Print statistics
    train_tokens = np.sum(tokenized['train']['len'], dtype=np.uint64)
    val_tokens = np.sum(tokenized['val']['len'], dtype=np.uint64)
    
    print(f"\nDataset statistics:")
    print(f"Train tokens: {train_tokens:,}")
    print(f"Val tokens: {val_tokens:,}")
    print(f"Train file size: ~{train_tokens * 2 / 1024**3:.1f}GB")
    print(f"Val file size: ~{val_tokens * 2 / 1024**3:.1f}GB")

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
    print("\nFiles saved as train.bin and val.bin")
    print("To read later: m = np.memmap('train.bin', dtype=np.uint16, mode='r')")