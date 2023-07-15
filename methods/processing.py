def processing_one_sentence(sentence):
    clean_sentence = ''
    for word in sentence.replace('\n', '').replace('\r', '').lower().split():
        if "http" not in word and "@" not in word and word.isalpha() and word not in stopwords:
            clean_sentence = sentence + word + " "
    remove_digitals = str.maketrans(' ', ' ', digits)
    clean_sentence.translate(remove_digitals)
    clean_sentence = re.sub(r'[0-9]+]', ' ', clean_sentence)
    clean_sentence = re.sub(r'[^\w\s]', ' ', clean_sentence[:-1])
    return clean_sentence