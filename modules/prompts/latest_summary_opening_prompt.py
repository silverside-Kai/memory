def latest_summary_opening_prompt():
    import openai
    from modules.misc.max_length import truncate_string

    prompt = """I am writing a content template for my twitter account. This type of tweet is called "AI Daily Pulse"
Write an opening sentence to give an introduction of last-24-hour AI important news and trends.
You need to consider the text layout and appeal.
Directly write. No need to add quotation marks!"""

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