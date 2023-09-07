def tldr(media, title, content_long):
    response = [{"role": "user",
                "content": f"Can you provide me a TLDR of this {media} with at most 3 bullet points?\
                 Every bullet point should be independent so that it can be understood without refering to the other points.\
                 You can directly start without responding me.\
                 Title: {title}. Content: {content_long}"}]
    
    return response
