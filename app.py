import os
import logging
from dotenv import load_dotenv

from sqlalchemy import create_engine, Column, Integer, String, Text, LargeBinary, TIMESTAMP, select
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
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
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
database_url = os.getenv('DATABASE_URL')  # e.g., 'postgresql://user:password@localhost/dbname'

if not database_url:
    raise ValueError("DATABASE_URL is not set in the environment variables")

# Initialize the database engine and session
engine = create_engine(database_url)
Session = sessionmaker(bind=engine)
session = Session()

# Define the database model
Base = declarative_base()

class TrainingData(Base):
    __tablename__ = 'training_data'
    id = Column(Integer, primary_key=True)
    job_title = Column(String, nullable=False)
    company_name = Column(String, nullable=False)
    data = Column(Text, nullable=False)
    embeddings = Column(LargeBinary)
    processed_files = Column(Text)  # Store the list of processed files
    created_at = Column(TIMESTAMP, server_default=func.now())

# Initialize the OpenAI chat model
model = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)

# Initialize the OpenAI embeddings model
embedder = OpenAIEmbeddings(openai_api_key=api_key)

# Function to check if training data exists and load it
def load_training_data(job_title, company_name):
    stmt = select(TrainingData).where(
        TrainingData.job_title == job_title,
        TrainingData.company_name == company_name
    )
    result = session.execute(stmt).first()
    return result[0] if result else None

# Function to create embeddings and chunks from new data files
def create_chunks_and_embeddings_from_new_files(data_directory_path, processed_files):
    data = ""
    new_files = []
    for filename in os.listdir(data_directory_path):
        if filename.endswith(".txt") and filename not in processed_files:
            new_files.append(filename)
            with open(os.path.join(data_directory_path, filename), "r") as f:
                data += f.read() + "\n"

    # If no new files, return empty results
    if not new_files:
        return None, None, None

    # Split the data into chunks (you can adjust the chunk size as needed)
    chunk_size = 1000  # Example chunk size, you can change this as needed
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Generate embeddings for the chunks
    embeddings = embedder.embed_documents(chunks)

    # Convert embeddings to a numpy array
    embedding_array = np.array(embeddings).astype('float32')

    return chunks, embedding_array, new_files

# Prompt the user for job title and company name
job_title = input("Enter the job title (e.g., Sales Engineer): ").strip().lower()
company_name = input("Enter the company name (e.g., Observe Inc.): ").strip().lower()

# Load or create training data
training_data = load_training_data(job_title, company_name)
if training_data:
    processed_files = training_data.processed_files.split(',') if training_data.processed_files else []
    if training_data.embeddings:
        # Load chunks and embeddings from the database
        chunks = training_data.data.split('\n')
        embedding_array = np.frombuffer(training_data.embeddings, dtype='float32').reshape(-1, 1536)
        print(f"Loaded {len(chunks)} chunks and {embedding_array.shape[0]} embeddings from the database.")
    else:
        chunks, embedding_array = [], []

    # Ask if there are new files to embed
    add_new_data = input("Do you have new training data to embed? (yes/no): ").strip().lower()
    if add_new_data == 'yes':
        data_directory_path = input("Provide the full file path to your new training data (i.e. /Users/wyliebrown/docs/youtube): ").strip()
        new_chunks, new_embedding_array, new_files = create_chunks_and_embeddings_from_new_files(data_directory_path, processed_files)
        if new_chunks and new_embedding_array:
            # Update existing data and embeddings
            chunks.extend(new_chunks)
            embedding_array = np.concatenate((embedding_array, new_embedding_array), axis=0)
            
            # Save updated embeddings and processed files back to the database
            training_data.data += '\n' + '\n'.join(new_chunks)
            training_data.embeddings = embedding_array.tobytes()
            training_data.processed_files = ','.join(processed_files + new_files)
            session.commit()
            print(f"Updated training data with new files: {new_files}")
        else:
            print("No new .txt files found to embed.")
else:
    add_data = input(f"No training data found for {job_title} at {company_name}. Would you like to add training data? (yes/no): ").strip().lower()
    if add_data == 'yes':
        data_directory_path = input("Provide the full file path to your training data (i.e. /Users/wyliebrown/docs/youtube): ").strip()
        chunks, embedding_array, new_files = create_chunks_and_embeddings_from_new_files(data_directory_path, [])
        new_training_data = TrainingData(
            job_title=job_title,
            company_name=company_name,
            data='\n'.join(chunks),
            embeddings=embedding_array.tobytes(),
            processed_files=','.join(new_files)
        )
        session.add(new_training_data)
        session.commit()
        print(f"Added training data for {job_title} at {company_name}.")
    else:
        print("Exiting application.")
        exit()

# Initialize FAISS index
dimension = embedding_array.shape[1]  # Example dimension, ensure it matches your embeddings
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the FAISS index
index.add(embedding_array)

print("Chunks and embeddings have been successfully created and saved.")

# Function to get relevant context from FAISS index
def get_relevant_context(question, index, chunks, embedder, k=5):
    question_embedding = embedder.embed_documents([question])[0]
    D, I = index.search(np.array([question_embedding]), k)
    relevant_chunks = [chunks[i] for i in I[0]]
    return relevant_chunks

# Prompt the user for interview customization inputs
print("To help train the chatbot to deliver great interview preparation, please provide the following information:")

industry = input("Enter the industry the company operates in (e.g., micro-service applications, Kubernetes, etc.): ").strip().lower()

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
            f"You are are here to act as the user's interview coach. The user is inteviewing for a position as a {job_title}. You should be asking challenging interview questions about how to be a {job_title} specifically at {company_name} which is in the {industry} industry. Answer all questions to the best of your ability."
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
    
    initial_prompt = (
        f"Ask this candidate a question they would very likely encounter in an interview about the company they are interviewing at called {company_name}."
    )
    relevant_context = get_relevant_context(initial_prompt, index, chunks, embedder)
    context_str = " ".join(relevant_context)

    # Ask the initial interview question
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

    while True:
        # Get user's answer
        user_answer = input("Your Answer: ")

        # Analyze user's answer and ask the next question in the same response
        analysis_prompt = f"Analysis: Provide feedback on the accuracy and completeness of the answer for the role of {job_title} at {company_name}. Highlight how well the answer addresses the specific needs and challenges within the {industry} industry. Explain how the answer could be improved. After that, ask another question related to the company {company_name} and the role of {job_title}."
        messages = [
            {"role": "system", "content": f"Context: {context_str}"},
            {"role": "user", "content": f"Question: {question}"},
            {"role": "user", "content": f"Answer: {user_answer}"},
            {"role": "user", "content": analysis_prompt}
        ]
        response = with_message_history.invoke(
            {"messages": [HumanMessage(content=user_answer)]},
            config=config,
        )
        analysis_and_next_question = response.content
        print(f"Analysis and Next Question: {analysis_and_next_question}")

        # Extract the new question from the analysis and next question response
        # Assuming the new question starts after the analysis part, you might need to adjust this part based on how the response is formatted
        if "Next Question:" in analysis_and_next_question:
            analysis, next_question = analysis_and_next_question.split("Next Question:", 1)
            print(f"Analysis: {analysis.strip()}")
            question = next_question.strip()
        else:
            # If we cannot split properly, assume the entire response is the new question for simplicity
            question = analysis_and_next_question.strip()

# Start the interview loop
interview_loop(index, chunks, embedder)

