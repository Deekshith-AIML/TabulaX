#import torch
import os
import openai
#from huggingface_hub import login
from langchain_openai import ChatOpenAI
#from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging, BitsAndBytesConfig
# Authenticate with Hugging Face (update with your token securely)
#login(token=os.environ["HF_TOKEN"])

os.environ["OPENAI_API_KEY"] = ""

langchain_model = ChatOpenAI(model="gpt-4o-mini")

# Attempt to load Mistral model from Hugging Face
"""def load_llama_model():
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=False,
        )
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        #model_name="mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config,
        )
        llama_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        print("Model loaded successfully\n")
        return llama_pipeline
    except Exception as e:
        print("Error loading llama", e)
        return None
"""

model = None

def get_cached_model():
    global model
    if model is None:
        model = None
    return model
