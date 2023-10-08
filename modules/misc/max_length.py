def truncate_string(input_string, max_length=250):
    import re
    
    if len(input_string) <= max_length:
        return input_string

    # Define a regular expression pattern to split the string into sentences.
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'

    # Split the string into sentences.
    sentences = re.split(sentence_pattern, input_string)

    # Initialize a variable to keep track of the total length.
    total_length = sum(len(sentence) for sentence in sentences)

    # Use a while loop to remove sentences until the total length is less than max_length.
    while total_length > max_length:
        # Remove the last sentence and update the total length.
        removed_sentence = sentences.pop()
        total_length -= len(removed_sentence)

    # Join the remaining sentences and return the truncated string.
    return ' '.join(sentences)