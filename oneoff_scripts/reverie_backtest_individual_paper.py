from langchain.document_loaders import ArxivLoader
from semanticscholar import SemanticScholar
from modules.prompts.simple_prompt import simple_prompt


sch = SemanticScholar()
target_paper_id = '2312.10003'
docs = ArxivLoader(query=target_paper_id, load_max_docs=1).load()
ref_papers = sch.get_paper_references('ARXIV:'+target_paper_id)

arxiv_fields = []
for entry in ref_papers._data:
    cited_paper = entry.get('citedPaper')
    if cited_paper and isinstance(cited_paper, dict):
        external_ids = cited_paper.get('externalIds')
        if external_ids and isinstance(external_ids, dict):
            arxiv_value = external_ids.get('ArXiv')
            if arxiv_value:
                arxiv_fields.append(arxiv_value)

# Case 1: Ask GPT to summarise knowledge directly
target_paper_content = docs[0].metadata['Summary']

# Case 2: RAG + references (with noises) to summarise knowledge
ref_papers_df = []
for ref_paper_id in arxiv_fields:
    print(ref_paper_id)
    ref_paper_docs = ArxivLoader(query=ref_paper_id, load_max_docs=1).load()
    ref_papers_df.append(ref_paper_docs[0].metadata)


from litellm import completion

response = completion(
    model="ollama/mistral",
    messages=[{"content": "who are you? respond in 20 words",
    "role": "user"}],
    api_base="http://localhost:11434"
)

from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatLiteLLM
from langchain.chat_models import ChatOpenAI
from config import openai_key, openai_api_base, pgvector_string
import os
import openai


os.environ['OPENAI_API_KEY'] = openai_key
openai.api_key = openai_key
openai.api_base = openai_api_base


chat = ChatLiteLLM(model="ollama/mistral")

llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k-0613',
                 openai_api_base=openai_api_base,
                 request_timeout=120)

messages = [
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    )
]
llm(messages)