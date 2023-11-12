from langchain.document_loaders.rtf import UnstructuredRTFLoader
import openai
from langchain.chains import RetrievalQA
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.pgvector import PGVector
from config import openai_key, openai_api_base, pgvector_string


os.environ['OPENAI_API_KEY'] = openai_key
openai.api_key = openai_key
openai.api_base = openai_api_base

# load the data
loader = UnstructuredRTFLoader("./docs/kai_ai_agent.rtf")
documents = loader.load()
docs = documents[0].page_content

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
texts = text_splitter.split_documents(documents)

# step 1: ask 5 questions
prompt = 'Given only the above information, what are 5 most salient highlevel questions we can answer about the subjects in the statements? '
prompt = docs[:1000] + prompt

prompts = [{"role": "user",
            "content": prompt}]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    request_timeout=30,
    messages=prompts
)

response_text = response['choices'][0]['message']['content']

questions = response_text.splitlines()
questions = [item for item in questions if item != '']

# Step 1: Generate tag metadata and store in the vectordb
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

store = PGVector(
    collection_name='kai',
    connection_string=pgvector_string,
    embedding_function=embeddings,
)

store.add_documents(documents)

retriever = store.as_retriever()
llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k-0613',
                 openai_api_base=openai_api_base,
                 request_timeout=120)
qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever)

llm_responses = []
for question in questions:
    print(question)
    llm_response = qa(question)
    print(llm_response)
    llm_responses.append(llm_response)

result_list = [entry['result'] for entry in llm_responses]
