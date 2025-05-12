# coding=utf-8
from argparse import Namespace
import torch
import torch_directml

# Initialize DirectML first
dml = torch_directml.device()

args = Namespace()

arg_dict = {
    'run_gradio': True, 
    'demo_public': False, 
    # 'model_name_or_path': 'facebook/opt-125m', 
    # 'model_name_or_path': 'facebook/opt-1.3b', 
    # 'model_name_or_path': 'facebook/opt-2.7b', 
    'model_name_or_path': 'facebook/opt-6.7b',
    # 'model_name_or_path': 'facebook/opt-13b',
    'load_fp16' : True,
    # 'load_fp16' : False,
    'prompt_max_length': None, 
    'max_new_tokens': 200, 
    'generation_seed': 123, 
    'use_sampling': True, 
    'n_beams': 1, 
    'sampling_temp': 0.7, 
    'use_gpu': True, 
    'seeding_scheme': 'simple_1', 
    'gamma': 0.25, 
    'delta': 2.0, 
    'normalizers': '', 
    'ignore_repeated_bigrams': False, 
    'detection_z_threshold': 4.0, 
    'select_green_tokens': True,
    'skip_model_load': False,
    'seed_separately': True,
}

args.__dict__.update(arg_dict)

args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5","T0"]])
args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt","opt","bloom"]])
   

# Add DirectML-specific modifications
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_directml(args):
    print("Loading model with DirectML optimization...")
    
    
    # Load with FP16 and low-memory settings
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if args.load_fp16 else torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Move to DirectML device
    model = model.to(dml)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    return model, tokenizer, dml

# Monkey-patch the original load_model function
from demo_watermark import load_model as original_load_model
import demo_watermark
demo_watermark.load_model = load_model_directml

# Run main
from demo_watermark import main
main(args)