def latest_summary_opening_prompt(num_contents):
    import openai
    from modules.misc.max_length import truncate_string
    import pymongo
    from config import mongo_string


    mongo_collection = pymongo.MongoClient(mongo_string)['mvp']['source']

    query = {"daily_tweeted": {"$exists": False}}
    sort_order = [("created_time", pymongo.DESCENDING)]

    data = mongo_collection.find(query).sort(sort_order).limit(num_contents)

    top_contents = ''
    for document in data:
        top_content = document['content']
        top_contents = top_contents + top_content + '\n'

    prompt = f"""Write a very short sentence to attract readers that we're doing AI Daily Pulse.
And then summarise the above AI content in one sentence.
Use fewer than 270 characters in total.
"""
    prompt = top_contents + prompt

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