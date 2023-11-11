def latest_summary_retweet_prompt(link):
    import openai
    from config import openai_key, openai_api_base
    from modules.misc.max_length import truncate_string
    import os
    import pymongo
    from config import mongo_string


    os.environ['OPENAI_API_KEY'] = openai_key
    openai.api_key = openai_key
    openai.api_base = openai_api_base

    mongo_collection = pymongo.MongoClient(mongo_string)['mvp']['source']

    query = {"link": link}
    data = mongo_collection.find(query)

    for document in data:
        top_content = document['content']
        top_link = document['link']

    prompt = """Generate one sentence to explain what the AI content is about given the above abstract, in an appealing way.
    Don't enclose output in single (') or double quotes("), provide it as it is.
    Use fewer than 255 characters."""
    prompt = 'Abstract: ' + top_content + ' ' + prompt

    prompts = [{"role": "user",
                "content": prompt}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        request_timeout=90,
        messages=prompts
    )

    response_text = response['choices'][0]['message']['content']

    result = truncate_string(response_text, max_length=255)
    result = result + ' ' + top_link

    return result