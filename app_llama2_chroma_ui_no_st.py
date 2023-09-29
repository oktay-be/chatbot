#python -m pip install -r requirements.txt

import logging
import time
import os
import torch
from auto_gptq import AutoGPTQForCausalLM
import streamlit as st

from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

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

def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    return loader.load()[0]


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)
    
def load_documents(source_dir: str) -> list[Document]:
    # Can be changed to a specific number
    INGEST_THREADS = os.cpu_count() or 8
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            contents, _ = future.result()
            docs.extend(contents)

    return docs

def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        file_extension = os.path.splitext(doc.metadata["source"])[1]
        if file_extension == ".py":
            python_docs.append(doc)
        else:
            text_docs.append(doc)

    return text_docs, python_docs

def ingest_docs():
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=440, chunk_overlap=100
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    # embeddings = load_embedding_model()

    # db = Chroma.from_documents(
    #     texts,
    #     embeddings,
    #     persist_directory=PERSIST_DIRECTORY,
    #     client_settings=CHROMA_SETTINGS,

    # )

def handle_user_input(user_question):
    print("handle user input")
    # response = st.session_state.conversation({'question': user_question})
    # st.write(response)

# LOAD MODELS FROM LOCAL
def load_embedding_model():

    # Check if embeddings are already in the session state
    if "embeddings" in st.session_state:
        return st.session_state.embeddings

    model_name = "hkunlp/instructor-xl"
    local_model_dir = "./models/sentence_transformers/" + model_name
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_name, cache_folder=local_model_dir, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    # Store embeddings in the session state for reuse
    st.session_state.embeddings = embeddings

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


def main():

    # load the embedding model
    embeddings = load_embedding_model()

    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,

    )

    # initiate retriever
    retriever = db.as_retriever()

    # uncomment for llmchain usage
    #docs = db.similarity_search("How did Lannisters acquire power?")


    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    llm = load_chat_model("cuda", model_id=MODEL_ID, model_basename=MODEL_BASENAME)


    # st.set_page_config(page_title="OgeBot", page_icon=":books:")

    # if "conversation" not in st.session_state:
    #     st.session_state.conversation = None

    # st.header("Chatting with :books:")
    # user_question = st.text_input("Ask a question")
    # if  user_question:
    #     handle_user_input(user_question)

    # with st.sidebar:
    #     st.subheader("Documents you uploaded")
    #     docs = st.file_uploader("Upload your docs here:", accept_multiple_files=True)

    #     if st.button("Learn"):
    #         with st.spinner("Learnging stuff"):

    #             if docs:
    #                 for doc in docs:
    #                     # The name of the file
    #                     filename = doc.name
                        
    #                     # Full path to save the file
    #                     file_path = os.path.join(SOURCE_DIRECTORY, filename)
                        
    #                     # 2. Loop through each uploaded document
    #                     # 3. Write the contents of each document into a new file inside the directory
    #                     with open(file_path, 'wb') as f:
    #                         f.write(doc.getvalue())

    ingest_docs()

    vectorstore = get_vectorstore(text_chunks)

    print ("vectors written")

    #conversation history with reinitialization
    #conversation = get_conversation_chain(vectorstore)

    #conversation history with session
    #st.session_state.conversation = get_conversation_chain(vectorstore)


    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        # Get the answer from the chain
        start_time = time.time()
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)
        print("ANSWER PROCURED %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()

