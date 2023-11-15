import streamlit as st
import requests
import io
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplaytes import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

# Image captioning code
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
# Access the API token
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Use the token in your headers
headers = {"Authorization": f"Bearer {api_token}"}

def query_image_caption(image):
    # Read the uploaded image as bytes
    image_data = image.read()

    response = requests.post(API_URL, headers=headers, data=image_data)
    image_caption = response.json()[0]['generated_text']

    return image_caption


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question, image_caption):
    # Concatenate the user's question and image caption
    universalprompt = "Assume me that im a high level electronic engineer so give me the answer more and more technical that only  very high experience people can understand and also dont mention anywhere about me just give answer.question is "
    user_question = universalprompt + user_question
    user_input = f"{user_question}. Additional information is {image_caption}"
    
    response = st.session_state.conversation({'question': user_input})
    st.session_state.chat_history = response['chat_history']

    bot_responses = [message.content for i, message in enumerate(st.session_state.chat_history) if i % 2 == 1]

    for bot_response in bot_responses:
        st.write(bot_template.replace("{{MSG}}", bot_response), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Arc FaultBOt", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Arc FaultBot:books:")
    user_question = st.text_input("Ask Anything about your products:")
    image_caption = None

    with st.sidebar:
        st.subheader("Image Captioning")
        image = st.file_uploader("Upload an image for captioning")
        if image:
            image_caption = query_image_caption(image)
            st.image(image, caption=image_caption)

        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

    if user_question:
        handle_userinput(user_question, image_caption)

if __name__ == '__main__':
    main()
