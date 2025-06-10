import os
from tqdm import tqdm
from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import cmasher as cmr
import sys
from typing import List, Iterable, Optional
from functools import partial
import time
import random
import math
import json
import torch
from torch import Tensor
from tokenizers import Tokenizer
from datasets import load_dataset, Dataset
from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from transformers import (AutoTokenizer, 
                          AutoModelForSeq2SeqLM, 
                          AutoModelForCausalLM,
                          LogitsProcessorList,
                          BertTokenizer, 
                          BertForMaskedLM)

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

dataset_name = "c4"
dataset_config_name = "realnewslike"

dataset = load_dataset(dataset_name, dataset_config_name, split="train", streaming=True, trust_remote_code=True)

# log an example
ds_iterator = iter(dataset)
idx = 75 # if this is c4, it's the schumacher example lol
i = 0
while i < idx: 
    next(ds_iterator)
    i += 1

example = next(ds_iterator)
print(example)

hf_model_name = "facebook/opt-1.3b"

model = AutoModelForCausalLM.from_pretrained(hf_model_name)

tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

# defaults to device 0
# will need to use 'parallelize' for multi-gpu sharding
device = torch.device("cuda")
model = model.to(device)
model.eval()

model_name = "bert-base-uncased"
paraphrase_tokenizer = BertTokenizer.from_pretrained(model_name)
paraphrase_model = BertForMaskedLM.from_pretrained(model_name)
paraphrase_model = paraphrase_model.to(device)

watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                               gamma=0.25,
                                               delta=2.0,
                                               use_temp=False,
                                               temp_h=10,
                                               temp_t0=1.0,
                                               temp_m=0.7,
                                               temp_M=1.3,
                                               seeding_scheme="selfhash") 

watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=0.25, # should match original setting
                                        seeding_scheme="selfhash", # should match original setting
                                        device=model.device, # must match the original rng device type
                                        model=model,
                                        delta=2.0,
                                        tokenizer=tokenizer,
                                        z_threshold=4.0,
                                        use_temp=False,
                                        normalizers=[],
                                        ignore_repeated_ngrams=True)

def mask_tokens(tokens, mask_fraction):
    """Randomly mask some tokens"""
    n_to_mask = max(1, int(len(tokens) * mask_fraction))
    mask_indices = random.sample(range(1, len(tokens)-1), n_to_mask)  # skip [CLS] and [SEP]
    masked_tokens = tokens.copy()
    for idx in mask_indices:
        masked_tokens[idx] = paraphrase_tokenizer.mask_token
    return masked_tokens, mask_indices

def iterative_paraphrase(text, mask_fraction):
    # Tokenize
    tokens = paraphrase_tokenizer.tokenize(text)
    tokens = [paraphrase_tokenizer.cls_token] + tokens + [paraphrase_tokenizer.sep_token]
    masked_tokens, mask_indices = mask_tokens(tokens, mask_fraction)
    
    for idx in mask_indices:
        input_ids = paraphrase_tokenizer.convert_tokens_to_ids(masked_tokens)
        input_tensor = torch.tensor([input_ids]).to(device)
        
        with torch.no_grad():
            outputs = paraphrase_model(input_tensor)
            predictions = outputs.logits
        
        topk = torch.topk(predictions[0, idx], k=50)
        for token_id in topk.indices:
            token = paraphrase_tokenizer.convert_ids_to_tokens([token_id.item()])[0]
            if not token.startswith("[unused") and token != paraphrase_tokenizer.unk_token:
                predicted_token = token
                break
        else:
            predicted_token = paraphrase_tokenizer.unk_token
        
        masked_tokens[idx] = predicted_token
    
    # Detokenize
    return paraphrase_tokenizer.convert_tokens_to_string(masked_tokens[1:-1])  # remove [CLS], [SEP]








results = []

num_samples = 50

for i in range(num_samples):
    print(f"\nRunning sample {i+1}/{num_samples}")
    example = next(ds_iterator)
    try:
        tokenized_prompt = tokenizer(example["text"], return_tensors='pt', truncation=True, max_length=50)
        input = tokenizer.decode(tokenized_prompt["input_ids"][0], skip_special_tokens=True)

        tokenized_prompt = {k: v.to(device) for k, v in tokenized_prompt.items()}

        output_tokens = model.generate(**tokenized_prompt,
                                        do_sample=True, 
                                        top_k=0,
                                        temperature=0.7,
                                        max_new_tokens=100,
                                        repetition_penalty=1.1,
                                        logits_processor=LogitsProcessorList([watermark_processor])
                                        )

        output_tokens = output_tokens[:,tokenized_prompt["input_ids"].shape[-1]:]
        output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

        score_dict = watermark_detector.detect(output_text)

        paraphrased = iterative_paraphrase(output_text, mask_fraction=0.3)

        paraphrased_score_dict = watermark_detector.detect(paraphrased)
        
        results.append({
            "input": input,
            "output": output_text,
            "score": score_dict,
            "paraphrased_output": paraphrased,
            "paraphrased_score": paraphrased_score_dict,
        })
    except Exception as e:
        print(f"Error on sample {i+1}: {e}")
        continue

# Convert results to DataFrame and save to CSV
df = pd.DataFrame([
    {
        "input": r["input"],
        "output": r["output"],
        "z_score": r["score"].get("z_score"),
        "p_value": r["score"].get("p_value"),
        "prediction": r["score"].get("prediction"),
        "sequence_prob_score": r["score"].get("sequence_prob_score"),
        "paraphrased_output": r["paraphrased_output"],
        "paraphrased_z_score": r["paraphrased_score"].get("z_score"),
        "paraphrased_p_value": r["paraphrased_score"].get("p_value"),
        "paraphrased_prediction": r["paraphrased_score"].get("prediction"),
        "paraphrased_sequence_prob_score": r["paraphrased_score"].get("sequence_prob_score"),
    }
    for r in results
])

df.to_json("watermark_paraphrase_results_with_temp_1_0.3.json", orient="records", indent=2)