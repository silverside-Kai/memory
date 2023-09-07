def tagify(bullet_point, tag_metadata):
    tag_list = [entry['tag'] for entry in tag_metadata['cmetadata']]
    tag_explanation = [entry for entry in tag_metadata['document']]
    # potentially replace tag_list with tag_metadata, which not only has a list of tags but also the explantions of each tag.
    response = [{"role": "system",
                 "content": f"You can only respond one of the tags among {tag_list}, without any other single character."},
                {"role": "user",
                "content": f"Can you tag the following content '{bullet_point}' using one tag among {tag_list}?\
                The explanations of {tag_list} are respectively as follows {tag_explanation}."}]

    return response
