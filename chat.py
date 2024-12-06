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
url = "https://textdoc.co/fCAmzT1RyWtlN9qj"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
text = soup.get_text(strip=True)  # Get all text and strip whitespace
text = text.replace(
    "Online Text Editor - Create, Edit, Share and Save Text FilesTextdocZipdocWriteurlTxtshareOnline CalcLoading…Open FileSave to Drive",
    "")
text = text.replace("/ Drive Add-on", "")

#create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


#split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=24,
    length_function=count_tokens,
)

chunks = text_splitter.create_documents([text])

# get embedding model and store in Chroma db
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings, )
from langchain_text_splitters import CharacterTextSplitter

embeddings = OpenAIEmbeddings()

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(chunks, embedding_function)

# create QA chain to integrate similarity search with user queries (answer query from knowledge base)

chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

# query = "What is sin?"
# docs = db.similarity_search(query)


def initialize_qa():
    global qa
    qa = None
    if qa is None:
        # Initialize your models here
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1),
                                                   db.as_retriever())
    return qa


chat_history = []


def test_on_submit(query):
    qa = initialize_qa()
    print(f"Message from front end flask: {query}")
    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    return result["answer"]
