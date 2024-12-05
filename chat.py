
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

# @title Enter the secret passphrase! Psst... it's not open sesame.
from dotenv import load_dotenv
import os
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
# def test_on_submit(query):
#     random_number = random.randint(1, 100)  # Generate a random number between 1 and 100
#     return "query " + str(random_number) + ": >> " + query

# @title Click play to chat!
import warnings
warnings.filterwarnings("ignore")
#get contents in webpage from url with library
import requests
from bs4 import BeautifulSoup

#get text data from url
url="https://textdoc.co/4YxEOjwkb6rMHyop"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
text = soup.get_text(strip=True)  # Get all text and strip whitespace
text = text[169:]
text= text[:18860]

#create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

#split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])

# get embedding model and store in Chroma db
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter
embeddings = OpenAIEmbeddings()

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(chunks, embedding_function)

# create QA chain to integrate similarity search with user queries (answer query from knowledge base)

chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

query = "What is sin?"
docs = db.similarity_search(query)

print(chain.run(input_documents=docs, question=query))
#
# #ATTRIBUTION: the following code block is taken straight from https://colab.research.google.com/drive/1OZpmLgd5D_qmjTnL5AsD1_ZJDdb7LQZI
# # make interactable UI
# from message_handler import receive_message
# # create conversation chain that uses our vectordb as retriver, this also allows for chat history management
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

chat_history = []

def test_on_submit(query):
    print(f"Message from front end flask: {query}")

    if query.lower() == 'exit':
        return "Thank you for the fun conversation! Play with me again!"

    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    return result["answer"]

