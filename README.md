# Fine Tune an AI Bot to do Interview Prep

## This project uses the following technologies:
1. Langchain for retrieval and chaining
2. OpenAI Chatmodel
3. FAISS vectorstore 

## The documentation used:
* https://python.langchain.com/v0.2/docs/tutorials/chatbot/

### How to set up this project locally:
1. Clone github repo locally
2. Create a .env environment for you OPENAI_API_KEY
3. Install dependencies: pip install -qU langchain-openai, pip install dotenv, pip install faiss-cpu
4. Prepare a file in your directory with .txt documents holding the information you want your model trained on. I will be launching an update later on this project with youtube transcription bot, web scraper and a user friendly way to build out this folder. 
5. run the python file and when prompted provided the file path to your training data
6. Start chatting, feel free to ask bot to reference previous converation, continue getting interview questions until you are satisfied. # interview-bot
