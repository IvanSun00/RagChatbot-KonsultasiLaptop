import streamlit as st
from AgentBot import agent

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
st.title("Laptop Consultation Chatbot")
# chatbot = Chatbot()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello there 👋!\n\n Good to see you, how may I help you today? Feel free to ask me 😁"}
    ]

# print(chatbot.chat_store.store)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# chatbot.set_chat_history(st.session_state.messages)

if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = agent.chat(prompt)
        st.markdown(response.response)

    # Add user message to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response.response})
