def latest_summary_opening_prompt(last_n_days):
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
    for document in data:
        content = document['content']
        if len(content) == 0:
            content = document['content_long']
        contents.append(content)
    combined_contents = "".join(contents)

    prompt = """I am writing a content template for my twitter account. This type of tweet is called "AI Daily Pulse"

Write an opening sentence to give an introduction of last-24-hour AI important news and trends, no need to show the detailed information, making readers curious about what are the news. You need to consider the text layout and appeal.
Directly write. No need for quotation marks."""
    prompt = prompt + combined_contents

    prompts = [{"role": "user",
                "content": prompt}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        request_timeout=30,
        messages=prompts
    )

    response_text = response['choices'][0]['message']['content']

    result = truncate_string(response_text, max_length=270)

    return result