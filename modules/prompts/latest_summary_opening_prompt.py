def latest_summary_opening_prompt():
    import openai
    from modules.misc.max_length import truncate_string

    prompt = """I am writing a content template for my twitter account. This type of tweet is called AI Daily Pulse
Help me write an introduction of last-24-hour AI important news and trends.
You need to consider the text layout and appeal.
Don't enclose output in single (') or double quotes("), provide it as it is.
Use fewer than 270 characters.
"""

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