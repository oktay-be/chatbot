from flask import Flask, request
import json
#python -m pip install -r requirements.txt

import logging
import time
import os
import torch
from auto_gptq import AutoGPTQForCausalLM
import streamlit as st
from htmlTemplates import css, bot_template, user_template

from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
)



# LOAD MODELS FROM LOCAL
def load_embedding_model():

    # Check if embeddings are already in the session state
    # if "embeddings" in st.session_state:
    #     return st.session_state.embeddings

    model_name = "hkunlp/instructor-xl"
    local_model_dir = "./models/sentence_transformers/" + model_name
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_name, cache_folder=local_model_dir, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    # Store embeddings in the session state for reuse
    # st.session_state.embeddings = embeddings

    return embeddings

def load_chat_model(device_type, model_id, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """

    if model_basename is not None:
        if ".ggml" in model_basename:
            logging.info("Using Llamacpp for GGML quantized models")
            #model_path = hf_hub_download(repo_id=model_id, filename=model_basename, resume_download=True, local_dir="models/llama2_with_transformers", local_dir_use_symlinks="false")
            model_path = hf_hub_download(repo_id=model_id, filename=model_basename, resume_download=True, cache_dir="models/llama2_ggml_with_transformers")
            max_ctx_size = 4096
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 20
                kwargs["n_batch"] = max_ctx_size
            return LlamaCpp(**kwargs)

        else:
            # The code supports all huggingface models that ends with GPTQ and have some variation
            # of .no-act.order or .safetensors in their HF repo.
            logging.info("Using AutoGPTQForCausalLM for quantized models")

            if ".safetensors" in model_basename:
                # Remove the ".safetensors" ending if present
                model_basename = model_basename.replace(".safetensors", "")

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            logging.info("Tokenizer loaded")

            #model_path = hf_hub_download(repo_id=model_id, filename=model_basename, resume_download=True, cache_dir="models/llama2_gptq_with_transformers")
            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,          
            )
    elif (
        device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin
        # file in their HF repo.
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=4096,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


#embeddings = load_embedding_model()

model_name = "hkunlp/instructor-xl"
local_model_dir = "./models/sentence_transformers/" + model_name
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceInstructEmbeddings(model_name=model_name, cache_folder=local_model_dir, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

# load the vectorstore
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,

)

# initiate retriever
# retriever = db.as_retriever()

# template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
# just say that you don't know, don't try to make up an answer.

# {context}

# {history}
# Question: {question}
# Helpful Answer:"""

# prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
# memory = ConversationBufferMemory(input_key="question", memory_key="history")

#llm = load_chat_model("cuda", model_id=MODEL_ID, model_basename=MODEL_BASENAME)

retriever = db.as_retriever() 

memory = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=load_chat_model("cuda", model_id=MODEL_ID, model_basename=MODEL_BASENAME),
    retriever=retriever,
    memory=memory
)


app = Flask(__name__)

@app.route("/llama2/prompt", methods=['POST'])
def respond():
    user_prompt = request.form.get("user_prompt")
    res = conversation_chain({"question": user_prompt, "chat_history": memory})
    ch = res["chat_history"]
    reshaped_chat_history = []
    for i in ch:
        reshaped_chat_history.append(i.content)
    
    result = {"question": res["question"], "answer": res["answer"], "chat_history": reshaped_chat_history }
    json_result = json.dumps(result)
    return json_result

if __name__ == '__main__':
    app.run(debug=True)




    #response = requests.post(main_prompt_url, data={"user_prompt": user_prompt})