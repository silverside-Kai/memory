from config import openai_key, openai_api_base
import openai
import os

from langchain.chat_models import ChatOpenAI
# from langchain.llms import Ollama

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from modules.load_data.tag_meta import tag_meta
from modules.retrieve_memory.retrieve_memory_assembly import retrieve_memory_assembly
from modules.prompts.tweet_per_content_prompt import tweet_per_content_prompt
from modules.prompts.retweet_prompt import retweet_prompt
from modules.prompts.latest_summary_opening_prompt import latest_summary_opening_prompt
from modules.prompts.latest_summary_retweet_prompt import latest_summary_retweet_prompt
from modules.prompts.insight_qa_prompt import insight_qa_prompt
from modules.load_data.load_vector_stores import store_basic_raw, store_latest_raw

os.environ['OPENAI_API_KEY'] = openai_key
openai.api_key = openai_key
openai.api_base = openai_api_base

llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k-0613',
                 openai_api_base=openai_api_base)
# llm = Ollama(base_url="http://localhost:11434",
#              model="llama2",
#              callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))

tag_metadata = tag_meta()
tag_list = [entry['tag'] for entry in tag_metadata['cmetadata']]

origins = ["*"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TweetMessage(BaseModel):
    link: str
    weight: float


class RetweetMessage(BaseModel):
    link: str


class TestingAssembly(BaseModel):
    content: str
    tag: str
    weight: float


class SummaryBulletPoint(BaseModel):
    num_contents: int
    media: str


@app.post("/tweet_per_content/")
async def tweet_per_content(payload: TweetMessage):
    link = payload.link
    weight_latest = payload.weight
    response = tweet_per_content_prompt(
        link, llm, store_basic_raw, store_latest_raw, weight_latest)
    return response


@app.post("/retweet/")
async def retweet(payload: RetweetMessage):
    link = payload.link
    response = retweet_prompt(
        link, llm, store_basic_raw)
    return response


@app.post("/summary_tweet_opening/")
async def summary_opening(payload: SummaryBulletPoint):
    num_contents = payload.num_contents
    media = payload.media
    response = latest_summary_opening_prompt(num_contents, media)
    return response


@app.post("/summary_retweet/")
async def summary_retweet(payload: RetweetMessage):
    link = payload.link
    response = latest_summary_retweet_prompt(link)
    return response


@app.post("/insight_qa/")
async def insight_qa(payload: RetweetMessage):
    link = payload.link
    response = insight_qa_prompt(link)
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
