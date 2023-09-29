# from transformers import AutoTokenizer, TextGenerationPipeline
# from auto_gptq import AutoGPTQForCausalLM

# MODEL = "TheBloke/WizardLM-7B-uncensored-GPTQ"
# model_basename ="WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order"

# import logging

# logging.basicConfig(
#     format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
# )

# device = "cuda:0"

# tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
# # download quantized model from Hugging Face Hub and load to the first GPU
# model = AutoGPTQForCausalLM.from_quantized(MODEL,
#         model_basename=model_basename,
#         device=device,
#         use_safetensors=True,
#         use_triton=False)

# # inference with model.generate
# prompt = "Tell me about AI"
# prompt_template=f'''### Human: {prompt}
# ### Assistant:'''

# input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
# output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=256, min_new_tokens=100)
# print(tokenizer.decode(output[0]))

from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch

quantized_model_dir = "TheBloke/stable-vicuna-13B-GPTQ"
model_basename = "wizard-vicuna-13B-GPTQ-4bit.compat.no-act-order"

tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)

AutoGPTQForCausalLM.from_quantized(quantized_model_dir, use_safetensors=True, model_basename=model_basename)