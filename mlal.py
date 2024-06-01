import json
import re
from googletrans import Translator
import spacy


def extract_medical_texts_and_predictions(dataset):
    medical_texts = []
    predictions = []

    for data_element in dataset:
        medical_text = data_element['data']['text']
        medical_texts.append(medical_text)

        for result_element in data_element['predictions']:
            predictions.append(result_element['result'])

    return medical_texts, predictions


def detect_language(text):
    translator = Translator()

    return translator.detect(text).lang


def remove_useless_symbols(text):
    """
    Substitute all characters that are not letters, digits, whitespaces or allowed punctuation with a whitespace.
    """
    allowed_punctuation = r'\.,;:\"!'
    pattern = f'[^{allowed_punctuation}\\w\\s]'
    return re.sub(pattern, ' ', text)


def extract_word_positions(text):
    pattern = re.compile(r'\w+|[^\w\s]')
    matches = pattern.finditer(text)

    words_with_indices = {match.group(): {'start': match.start(), 'end': match.end()} for match in matches}

    return words_with_indices


def is_useless(word):
    """
    Check if the word is a punctuation mark or a whitespace.
    """
    pattern = re.compile(r"[a-zA-Z]|\d")
    return not pattern.search(word)


def preprocess_text(base_text):
    text = remove_useless_symbols(base_text)
    
    text_chunk = text[:1000]    # Google Translate API has a limit of 5000 characters per request
    
    print("Text chunk: ", text_chunk)
    
    lang = detect_language(text_chunk)

    nlp = spacy.load('es_core_news_md') if lang == 'es' else spacy.load('ca_core_news_md')

    doc = nlp(text)
    token_sent = [[token for token in sentence if not is_useless(token.text)] for sentence in doc.sents]

    token_sent = clear_processed_text(token_sent)

    for i, sentence in enumerate(token_sent):
        token_sent[i] = [extract_features(sentence, j) for j in range(len(sentence))]

    return token_sent


def extract_features(sentence, i):
    token = sentence[i]
    word = token.text

    features = {
        'word': word,
        'word_lower': word.lower(),
        'is_capitalized': word[0].isupper(),
        'is_all_caps': word.isupper(),
        'is_digit': word.isdigit(),
        'word_length': len(word),
        'contains_digits': bool(re.search(r'\d', word)),
        'pos': token.pos_,
        'lemma': token.lemma_,
        'start-end': {'start': token.idx, 'end': token.idx + len(word)}
    }

    features["prefix_2"] = word[:2]
    features["suffix_2"] = word[-2:]

    if i > 0:
        previous_tokens = [sentence[j - 1].text.lower() for j in range(max(0, i - 6), i)]
        features["previous_words"] = previous_tokens

    if i < len(sentence) - 1:
        next_token = sentence[i + 1].text
        features["next_word"] = next_token.lower()

    return features


def clear_sentence(sentence):
    """
    Remove punctuation marks and whitespaces from the sentence.
    """
    return [token for token in sentence if not is_useless(token.text)]


def clear_processed_text(processed_text):
    """
    Clear each sentence in the processed text and remove empty sentences.   
    """
    cleared_text = [clear_sentence(sentence) for sentence in processed_text]
    return [sentence for sentence in cleared_text if sentence]

def extract_cues_and_scopes(document):
    """
    Extract negations, uncertain cues, negation scopes and uncertain scopes from the document.
    """
    negations = [result_element for result_element in document["predictions"][0]["result"] if "NEG" in result_element["value"]["labels"]]
    uncertains = [result_element for result_element in document["predictions"][0]["result"] if "UNC" in result_element["value"]["labels"]]
    nscopes = [result_element for result_element in document["predictions"][0]["result"] if "NSCO" in result_element["value"]["labels"]]
    uscopes = [result_element for result_element in document["predictions"][0]["result"] if "USCO" in result_element["value"]["labels"]]

    # Sort the cues and scopes by their start position
    negations.sort(key=lambda x: x["value"]["start"])
    uncertains.sort(key=lambda x: x["value"]["start"])
    nscopes.sort(key=lambda x: x["value"]["start"])
    uscopes.sort(key=lambda x: x["value"]["start"])

    return negations, uncertains, nscopes, uscopes



# Set BIO tags for the tokens, based on the training negations
def BIO_tagging(tokens, labels, label_name, original_text):
    """
    Set BIO tags for the tokens based on the labels from the training data.
    
    Args:
    - tokens (list): A list of token dictionaries
    - labels (list): A list of labels from the training data
    - label_name (str): The name of the label to use for tagging (e.g., "NEG")
    - original_text (str): The original text of the document
    """
     
    negation_idx = 0

    for token in tokens:
        if 'tag' not in token:
            token['tag'] = 'O'

        if negation_idx >= len(labels):
            continue

        negation = labels[negation_idx]

        neg_start = negation['value']['start']
        neg_end = negation['value']['end']

        # Skip whitespace characters at the start and end positions
        if original_text[neg_start] == ' ':
            neg_start += 1
        if original_text[neg_end - 1] == ' ':
            neg_end -= 1

        token_start = token['start-end']['start']
        token_end = token['start-end']['end']

        if token_start == neg_start:
            token['tag'] = f'B-{label_name}'
        elif neg_start < token_start < neg_end:
            token['tag'] = f'I-{label_name}'
        elif token_start > neg_end:
            # Move to the next negation that starts after the current token
            while negation_idx < len(labels) - 1 and token_start > labels[negation_idx]['value']['start']:
                negation_idx += 1
                negation = labels[negation_idx]
            
            if token_start == negation['value']['start']:
                token['tag'] = f'B-{label_name}'

    # # Print tokens with BIO tags
    # for token in tokens:
    #     print(token)
    
def BIO_tagging_X(X_train_texts, Y_train, medical_texts_train):
    for text_idx, (train_text, train_labels, original_text) in enumerate(list(zip(X_train_texts, Y_train, medical_texts_train))):
        negs, uncs, nscs, uscs = train_labels
        for i in range(len(train_text)):
            BIO_tagging(train_text[i], negs, "NEG", original_text)
            BIO_tagging(train_text[i], uncs, "UNC", original_text)
            BIO_tagging(train_text[i], nscs, "NSCO", original_text)
            BIO_tagging(train_text[i], uscs, "USCO", original_text)

        print("Text: ", text_idx, " : ", original_text[:100], "...")
        for i, sentence in enumerate(train_text[:2]):
            print("Sentence: ", i)
            for token in sentence[:5]:
                print(token)
        print("<--------------------------------------->")    



def main():
    with open('train_data.json', 'r', encoding='utf-8') as train_file:
        train_dataset = json.load(train_file)

    # with open('test.json', 'r', encoding='utf-8') as test_file:
    #     test_dataset = json.load(test_file)

    medical_texts_train, predictions_train = extract_medical_texts_and_predictions(train_dataset)
    # medical_texts_test, predictions_test = extract_medical_texts_and_predictions(test_dataset)

    limit = 4  # Limit the number of documents to process (set to -1 to process all documents)

    print("Number of training documents:", len(medical_texts_train))
    print("Limiting to", limit, "documents...")

    print("Preprocessing text and extracting cues...")

    X_train_texts = [preprocess_text(text) for text in medical_texts_train[:limit]]
    Y_train = [extract_cues_and_scopes(document) for document in train_dataset[:limit]]

    print("Tagging tokens...")

    BIO_tagging_X(X_train_texts, Y_train, medical_texts_train)
    
    # Now in X_train_texts we have the tokens with the BIO tags for each sentence in each document
    # [ [ [ {token1}, {token2}, ...], [ {token1}, {token2}, ...], ...], ...]
    # Each token has the following structure:
    # {
    #     'word': word,
    #     'start-end': {'start': token.idx, 'end': token.idx + len(word)},
    #     ...
    # }


if __name__ == '__main__':
    main()
