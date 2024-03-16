def simple_prompt(prompt):
    import openai
    from config import openai_key, openai_api_base
    import os

    
    os.environ['OPENAI_API_KEY'] = openai_key
    openai.api_key = openai_key
    openai.api_base = openai_api_base
    prompts = [{"role": "user",
            "content": prompt}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        request_timeout=300,
        messages=prompts
    )

    response_text = response['choices'][0]['message']['content']
    return response_text