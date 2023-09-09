from config import openai_key, openai_api_base, CONNECTION_STRING
import openai
import os

from langchain.vectorstores.pgvector import PGVector
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
# from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from modules.tag_meta import tag_meta
from modules.retrieve_deep_unconsciousness import retrieve_deep_unconsciousness
from modules.retrieve_working_memory import retrieve_working_memory
from modules.retrieve_memory_assembly import retrieve_memory_assembly
from modules.tweet_per_content_prompt import tweet_per_content_prompt


os.environ['OPENAI_API_KEY'] = openai_key
openai.api_key = openai_key
openai.api_base = openai_api_base

llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k-0613',
                 openai_api_base=openai_api_base)
# llm = Ollama(base_url="http://localhost:11434", 
#              model="llama2", 
#              callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

store_basic_raw = PGVector(
    collection_name='basic_raw',
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

store_latest_raw = PGVector(
    collection_name='latest_raw',
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

tag_metadata = tag_meta()
tag_list = [entry['tag'] for entry in tag_metadata['cmetadata']]


app = FastAPI()

origins = [
    "http://localhost"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TweetMessage(BaseModel):
    link: str
    weight: float


class Testing(BaseModel):
    content: str
    tag: str


class TestingAssembly(BaseModel):
    content: str
    tag: str
    weight: float


@app.post("/tweet_per_content/")
async def tweet_per_content(payload: TweetMessage):
    link = payload.link
    weight_latest = payload.weight
    response = tweet_per_content_prompt(
        link, llm, store_basic_raw, store_latest_raw, weight_latest)
    return response


@app.post("/deep_unconsciousness_testing/")
async def deep_unconsciousness_testing(payload: Testing):
    content = payload.content
    tag = payload.tag
    if tag in tag_list:
        response = retrieve_deep_unconsciousness(
            llm, store_basic_raw, content, tag)
    else:
        response = retrieve_deep_unconsciousness(llm, store_basic_raw, content)
    return response


@app.post("/working_memory_testing/")
async def working_memory_testing(payload: Testing):
    content = payload.content
    tag = payload.tag
    if tag in tag_list:
        response = retrieve_working_memory(llm, store_latest_raw, content, tag)
    else:
        response = retrieve_working_memory(llm, store_latest_raw, content)
    return response


@app.post("/memory_assembly_testing/")
async def memory_assembly_testing(payload: TestingAssembly):
    content = payload.content
    tag = payload.tag
    weight_latest = payload.weight
    if tag in tag_list:
        response = retrieve_memory_assembly(
            llm, store_basic_raw, store_latest_raw, content, weight_latest, tag)
    else:
        response = retrieve_memory_assembly(
            llm, store_basic_raw, store_latest_raw, content, weight_latest)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='localhost', port=8080)
