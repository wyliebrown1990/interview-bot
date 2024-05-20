import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
import faiss
import numpy as np

#STEP 1: INITIALIZE THE CHAT MODEL YOU WILL USE AND THEN EMBED YOUR TRAINING DATA INTO CHUNKS AND EMBEDDINGS FOR THE VECTOR STORE

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables from .env file
load_dotenv('.env')
api_key = os.getenv('OPENAI_API_KEY')

# Initialize the OpenAI chat model
model = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)

# Initialize the OpenAI embeddings model
embedder = OpenAIEmbeddings(openai_api_key=api_key)

# Prompt the user for the data directory path
data_directory_path = input("Please enter the directory path containing the .txt files: ").strip()

# Load data from all .txt files in the directory
data = ""
file_count = 0
for filename in os.listdir(data_directory_path):
    if filename.endswith(".txt"):
        file_count += 1
        with open(os.path.join(data_directory_path, filename), "r") as f:
            data += f.read() + "\n"

# Split the data into chunks (you can adjust the chunk size as needed)
chunk_size = 1000  # Example chunk size, you can change this as needed
chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

# Save chunks to chunks.txt
with open("chunks.txt", "w") as f:
    for chunk in chunks:
        f.write(chunk + "\n")

# Generate embeddings for the chunks
embeddings = embedder.embed_documents(chunks)

# Convert embeddings to a numpy array and save to embeddings.npy
embedding_array = np.array(embeddings).astype('float32')
np.save("embeddings.npy", embedding_array)

# Initialize FAISS index
dimension = embedding_array.shape[1]  # Example dimension, ensure it matches your embeddings
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the FAISS index
index.add(embedding_array)

# Print the number of files found, chunks created, and embeddings generated
print(f"Number of files found: {file_count}")
print(f"Number of chunks created: {len(chunks)}")
print(f"Number of embeddings generated: {embedding_array.shape[0]}")

print("Chunks and embeddings have been successfully created and saved.")

# Function to get relevant context from FAISS index
def get_relevant_context(question, index, chunks, embedder, k=5):
    question_embedding = embedder.embed_documents([question])[0]
    D, I = index.search(np.array([question_embedding]), k)
    relevant_chunks = [chunks[i] for i in I[0]]
    return relevant_chunks

# STEP 2: BUILD OUT YOUR INTERVIEW BOT ONCE YOUR VECTOR DATABASE INDEX IS CREATED AND READY FOR RETRIEVAL

# Define in-memory store for chat histories
store = {}

# Function to get chat history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Define prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are are here to act as the user's interview coach. The user wants to change careers and industries. The user is coming from a Product Management background and transitioning into Sales Engineering, also called a Solutions Consultant at some companies. You should be asking challenging interview questions about how to be a Sales Engineer and how to be a Sales Engineer at the specific company they are interviewing for. Answer all questions to the best of your ability."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Filter messages to manage conversation history length
def filter_messages(messages, k=10):
    return messages[-k:]

chain = (
    RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"]))
    | prompt
    | model
)

# Wrap chain with message history
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

# Interview loop
def interview_loop(index, chunks, embedder):
    session_id = "interview_session"
    config = {"configurable": {"session_id": session_id}}
    
    while True:
        initial_prompt = (
            "Ask this candidate a question they would very likely encounter in an interview about the company they are interviewing at called Observe Inc."
        )
        relevant_context = get_relevant_context(initial_prompt, index, chunks, embedder)
        context_str = " ".join(relevant_context)
        
        # Ask interview question
        messages = [
            {"role": "system", "content": f"Context: {context_str}"},
            {"role": "user", "content": initial_prompt}
        ]
        response = with_message_history.invoke(
            {"messages": [HumanMessage(content=initial_prompt)]},
            config=config,
        )
        question = response.content
        print(f"Interview Question: {question}")
        
        # Get user's answer
        user_answer = input("Your Answer: ")
        
        # Analyze user's answer
        messages = [
            {"role": "system", "content": f"Context: {context_str}"},
            {"role": "user", "content": f"Question: {question}"},
            {"role": "user", "content": f"Answer: {user_answer}"},
            {"role": "user", "content": "Analysis: Provide a thorough analysis of how the user answered the interview question you gave them. Check that the user covered every topic in your question. Check that they weren't too verbose and that they stayed on topic. Make sure they are giving ansers that a Sales Engineer would give. Make sure their information about Observe Inc is accurate. Make sure they are following interview best practices. Be as critical as possible to help them succeed in the future. If the user doesnt' provide a real example of how they accomplished something related to the interview question then remind them they should always use real world examples from their work history when answering interview questions. Give them examples of how they could do better if they don't have a perfect answer."}
        ]
        response = with_message_history.invoke(
            {"messages": [HumanMessage(content=user_answer)]},
            config=config,
        )
        analysis = response.content
        print(f"Analysis: {analysis}")
        
        # Check if user wants another question
        another_question = input("Do you want another question? (yes/no): ").strip().lower()
        if another_question != 'yes':
            break
        else:
            # Ask a new question only if the user wants another question
            continue

# Start the interview loop
interview_loop(index, chunks, embedder)
