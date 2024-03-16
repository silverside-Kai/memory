def simple_prompt(prompt):
    from litellm import completion

    response = completion(
        model="ollama/mistral",
        messages=[{"content": prompt,
                   "role": "user"}],
        api_base="http://localhost:11434"
    )

    response_text = response['choices'][0]['message']['content']
    return response_text
