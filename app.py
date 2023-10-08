#python -m pip install -r requirements.txt

import os
import streamlit as st
from htmlTemplates import css, bot_template, user_template
import requests

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
)

import subprocess

def handle_user_input(user_prompt):
    main_prompt_url = 'http://127.0.0.1:5000/llama2/prompt'
    response = requests.post(main_prompt_url, data={"user_prompt": user_prompt})
    response_content = response.json()

    st.session_state.chat_history = response_content['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)


def main():

    st.set_page_config(page_title="OgeBot", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Chatting with :books:")
    user_question = st.text_input("Ask a question")

    # create conversation chain
    # st.session_state.conversation = get_conversation_chain()

    if  user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Documents you uploaded!!!")
        docs = st.file_uploader("Upload your docs here:", accept_multiple_files=True)

        if st.button("Learn"):
            with st.spinner("Learnging stuff"):

                if docs:
                    for doc in docs:
                        # The name of the file
                        filename = doc.name
                        
                        # Full path to save the file
                        file_path = os.path.join(SOURCE_DIRECTORY, filename)
                        
                        # 2. Loop through each uploaded document
                        # 3. Write the contents of each document into a new file inside the directory
                        with open(file_path, 'wb') as f:
                            f.write(doc.getvalue())
                subprocess.run(["python", "ingest.py"])
                # print ("files processed")

                
if __name__ == '__main__':
    main()

