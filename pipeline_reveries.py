# from modules.load_data.load_vector_stores import store_latest_raw, store_basic_raw, embeddings
from config import openai_key, openai_api_base, pgvector_string
import openai
import os
# from langchain.chat_models import ChatOpenAI
from modules.load_data.read_sql import read_sql
# from langchain.chains import RetrievalQA
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine


os.environ['OPENAI_API_KEY'] = openai_key
openai.api_key = openai_key
openai.api_base = openai_api_base

# llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k-0613',
#                  openai_api_base=openai_api_base)

# retriever_latest_raw = store_latest_raw.as_retriever(search_kwargs={'k': 100})

# qa = RetrievalQA.from_chain_type(
#         llm=llm, chain_type="stuff", retriever=retriever_latest_raw)

data = read_sql('read_latest_docs.sql', 500)
data = data.sample(50)

docs = ''
for index, row in data.iterrows():
    line = row['document']
    new_str = ', '.join(line.split(', ')[1:])
    docs = docs + new_str + ' '

prompt = 'Given only the above information, what are 5 most salient highlevel questions we can answer about the subjects in the statements? '
prompt = docs + prompt

prompts = [{"role": "user",
            "content": prompt}]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    request_timeout=300,
    messages=prompts
)

response_text = response['choices'][0]['message']['content']
questions = response_text.splitlines()

current_date = datetime.now().strftime('%Y-%m-%d')
df = pd.DataFrame({"day": current_date,
                   "questions": questions})

postgresql_engine = create_engine(
    pgvector_string
)
df.to_sql(name='salient_questions', schema='memory', if_exists='append', index=False,
          con=postgresql_engine, method='multi')

# llm_response = qa(questions[0])