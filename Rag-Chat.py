import streamlit as st
from llama_index.llms.ollama import Ollama
from pathlib import Path
import qdrant_client
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from qdrant_client import QdrantClient



class Chatbot:
    def __init__(self, llm="qwen2.5:3b", embedding_model="intfloat/multilingual-e5-large", vector_store=None):
        self.Settings = self.set_setting(llm, embedding_model)

        # Indexing
        self.index = self.load_data()

        # Memory
        self.memory = self.create_memory()

        # Chat Engine
        self.chat_engine = self.create_chat_engine(self.index)

    def set_setting(_arg, llm, embedding_model):
        Settings.llm = Ollama(model=llm, base_url="http://127.0.0.1:11434")
        Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")
        Settings.system_prompt = """
                                You are an expert system specialized in providing laptop purchase consultations and education about laptop technologies. 
                                You have access to detailed information about laptops, including specifications, prices, and comparisons. 
                                Always provide clear and helpful guidance. If you don't know the answer, say you DON'T KNOW.
                                """

        return Settings

    @st.cache_resource(show_spinner=False)
    def load_data(_arg, vector_store=None):
        with st.spinner(text="Loading and indexing – hang tight! This should take a few minutes."):
            # Read & load document from folder
            reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
            documents = reader.load_data()

        if vector_store is None:
            client = QdrantClient(
                url=st.secrets["qdrant"]["connection_url"], 
                api_key=st.secrets["qdrant"]["api_key"],
            )
            vector_store = QdrantVectorStore(client=client, collection_name="Documents")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        return index

    def set_chat_history(self, messages):
        self.chat_history = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
        self.chat_store.store = {"chat_history": self.chat_history}

    def create_memory(self):
        self.chat_store = SimpleChatStore()
        return ChatMemoryBuffer.from_defaults(chat_store=self.chat_store, chat_store_key="chat_history", token_limit=16000)

    def create_chat_engine(self, index):
        return CondensePlusContextChatEngine(
            verbose=True,
            memory=self.memory,
            retriever=index.as_retriever(),
            llm=Settings.llm
        )
        # return index.as_chat_engine(chat_mode="condense_plus_context", chat_store_key="chat_history", memory=self.memory, verbose=True)





# Sidebar untuk sesi chat
def create_new_session():
    session_id = f"session_{len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[session_id] = {"name": "New Chat", "messages": []}
    st.session_state.selected_session = session_id

# Sidebar: Mengelola sesi chat
with st.sidebar:
    st.header("Chat Sessions")

    # Inisialisasi sesi chat jika tidak ada
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}

    if not st.session_state.chat_sessions:
        create_new_session()  # Buat sesi baru jika tidak ada

    # Menampilkan daftar sesi yang ada
    session_names = {session_id: session_data["name"] for session_id, session_data in st.session_state.chat_sessions.items()}
    selected_name = st.selectbox("Pilih sesi", list(session_names.values()))

    # Mendapatkan sesi yang dipilih berdasarkan nama
    if selected_name:
        selected_session_id = list(st.session_state.chat_sessions.keys())[list(session_names.values()).index(selected_name)]
        st.session_state.selected_session = selected_session_id

    # Opsi untuk memulai sesi baru
    if st.button("Mulai Sesi Baru"):
        create_new_session()

# Main Program
st.title("Laptop Purchase Consultation Chatbot")
chatbot = Chatbot()

# Suggested questions
st.markdown("**Pertanyaan Umum:**")
faq_questions = [
    "rekomendasi laptop dengan ram 16gb?",
    "Berapa budget ideal untuk laptop kerja?",
    "Rekomendasi laptop merek asus?"
]
# Menampilkan chat history dari sesi yang dipilih
if st.session_state.selected_session:
    session_data = st.session_state.chat_sessions[st.session_state.selected_session]
    session_messages = session_data["messages"]

    for message in session_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    chatbot.set_chat_history(session_messages)

    for question in faq_questions:
        if st.button(question):
            session_messages.append({"role": "user", "content": question})

            with st.chat_message("assistant"):
                response = chatbot.chat_engine.chat(question)
                st.markdown(response.response)
            session_messages.append({"role": "assistant", "content": response.response})
            
    # Reaksi terhadap input pengguna
    if prompt := st.chat_input("Ada yang bisa saya bantu?"):
        # Menampilkan pesan pengguna
        with st.chat_message("user"):
            st.markdown(prompt)

        # Menambahkan pesan pengguna ke history sesi
        session_messages.append({"role": "user", "content": prompt})

        # Update nama sesi dengan pesan terakhir dari pengguna
        session_data["name"] = prompt[:50]

        # Mendapatkan respons dari chatbot
        with st.chat_message("assistant"):
            response = chatbot.chat_engine.chat(prompt)
            st.markdown(response.response)

        # Menambahkan respons chatbot ke history sesi
        session_messages.append({"role": "assistant", "content": response.response})
else:
    st.write("Silakan pilih atau mulai sesi baru.")