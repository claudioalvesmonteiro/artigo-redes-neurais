

import pandas as pd
import nltk

fakes = pd.read_csv('v2/data/Fake.csv')
trues = pd.read_csv('v2/data/True.csv')

#====================
# preprocessamento
#===================

def cleanTextToken(text, tokenization = True):
    ''' standardize text to extract words
    '''
    # text to lowercase
    text = text.lower()
    # remove numbers
    #text = ''.join([i for i in text if not i.isdigit()]) 
    # remove punctuation
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+') # preserve words and alphanumeric
    text = tokenizer.tokenize(text)
    # remove stopwords
    from nltk.corpus import stopwords
    stop = set(stopwords.words('english'))
    text = [w for w in text if not w in stop] 
    # lemmatization
    from nltk.stem import WordNetLemmatizer 
    lemmatizer = WordNetLemmatizer() 
    text = [lemmatizer.lemmatize(word) for word in text]
    # return clean token
    return(text)

cleanTextToken(text, tokenization = True)

fakes['title_token'] = [cleanTextToken(text) for text in fakes['title']]
fakes['text_token'] = [cleanTextToken(text) for text in fakes['text'].values]

# one jhot encoder

# split
