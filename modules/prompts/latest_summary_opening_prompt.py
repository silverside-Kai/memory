def latest_summary_opening_prompt(num_contents, media):
    import openai
    from modules.misc.max_length import truncate_string
    import pymongo
    from config import mongo_string
    from datetime import datetime


    mongo_collection = pymongo.MongoClient(mongo_string)['mvp']['source']

    query = {
        "$and": [
            {"daily_tweeted": {"$exists": False}},
            {"score": {"$exists": True}},
            {"media": media}
        ]
    }
    sort_order = [("score", pymongo.DESCENDING)]

    data = mongo_collection.find(query).sort(sort_order).limit(num_contents)

    top_contents = ''
    for index, document in enumerate(data):
        top_content = f'Bite {index+1}: ' + document['content'] + '\n' + document['content_long']
        top_content = top_content[:500]
        top_contents = top_contents + top_content + '\n'

    prompt = f"""Summarise the above content in {num_contents} very short bullet points (each <10 words) covering the {num_contents} bites respectively.
Use fewer than 270 characters in total.
"""
    prompt = top_contents + prompt

    prompts = [{"role": "user",
                "content": prompt}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        request_timeout=90,
        messages=prompts
    )

    response_text = response['choices'][0]['message']['content']
    current_date = datetime.now().strftime('%Y-%m-%d')
    result = f'🍱 AI Daily Bento {current_date}\n' + truncate_string(response_text, max_length=270)

    return result