def latest_summary_retweet_prompt(last_n_days):
    import openai
    from config import openai_key, openai_api_base
    from modules.load_data.load_docs_excl_daily_tweeted import load_docs_excl_daily_tweeted
    from modules.misc.max_length import truncate_string
    import os


    os.environ['OPENAI_API_KEY'] = openai_key
    openai.api_key = openai_key
    openai.api_base = openai_api_base
    data = load_docs_excl_daily_tweeted(last_n_days)

    contents = []
    links = []
    for document in data:
        content = document['content']
        link = document['link']
        if len(content) == 0:
            content = document['content_long']
        contents.append(content)
        links.append(link)
    top_content = contents[0]
    top_link = links[0]

    prompt = """Generate one sentence to explain what the AI content is about given the following abstract, in an appealing way.
    Directly generate. No quote needed."""
    prompt = prompt + 'abstract: ' + top_content

    prompts = [{"role": "user",
                "content": prompt}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        request_timeout=30,
        messages=prompts
    )

    response_text = response['choices'][0]['message']['content']

    result = truncate_string(response_text, max_length=255)
    result = result + ' ' + top_link

    return result