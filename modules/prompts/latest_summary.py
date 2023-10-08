def latest_summary(num_docs):
    import openai
    from config import openai_key, openai_api_base
    from modules.load_data.read_sql import read_sql
    import os


    os.environ['OPENAI_API_KEY'] = openai_key
    openai.api_key = openai_key
    openai.api_base = openai_api_base
    data = read_sql('read_latest_docs.sql', num_docs)

    docs = ''
    for index, row in data.iterrows():
        line = row['document']
        new_str = ', '.join(line.split(', ')[1:])
        docs = docs + new_str + ' '

    prompt = """Given only the information following,
    select the most important content
    and compile a summary of AI news with three bullet points.
    Before the bullet points, start the summary with this first line:
    BAM! Latest AI news 📢"""
    prompt = prompt + docs

    prompts = [{"role": "user",
                "content": prompt}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        request_timeout=30,
        messages=prompts
    )

    response_text = response['choices'][0]['message']['content']

    return response_text