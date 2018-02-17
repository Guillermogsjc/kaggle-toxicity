import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string


STOPWORDS_SET = set(stopwords.words('english'))

def get_upper_case_prop(s):
    aux_list = [char.isupper() for char in s if char.isalpha()]
    n_upper_cases = sum(aux_list)
    n = len(aux_list)

    if n > 0:
        val = n_upper_cases / n
    else:
        val = 0.0

    return val


def get_punctuation_prop(s):
    aux_list = [char for char in s if char in string.punctuation]

    n_puncts = len(aux_list)
    n = len(s)

    if n > 0:
        val = n_puncts / n
    else:
        val = 0.0

    return val

def process_text(text, stem_words=False, remove_stop_words=True):

    # Convert words to lower case and split them
    text = text.lower()
    
    #Removes \n
    


    # Clean the text
    text=re.sub("\\n","",text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub("\d+", "", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    
    # Remove leaky elements like ip,user
    text=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",text)
    # Removing usernames
    text=re.sub("\[\[.*\]","",text)
    
    # Remove links
    text = re.sub("(f|ht)tp(s?)://\\S+", " ", text)
    text = re.sub("http\\S+", "", text)
    text = re.sub("xml\\S+", "", text)
    
    # Transform short forms
    text = re.sub("'ll", " will", text)
    text = re.sub("i'm", "i am", text)
    text = re.sub("'re", " are", text)
    text = re.sub("'s", " is", text)
    text = re.sub("'ve", " have", text)
    text = re.sub("'d", " would", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"i`m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"havn't", " have not ", text)
    
    # Some "shittext"
    #text = re.sub("n ig ger", "nigger", text)
    #text = re.sub("s hit", "shit", text)
    #text = re.sub("g ay", "gay", text)
    #text = re.sub("f ag got", "faggot", text)
    #text = re.sub("c ock", "cock", text)
    #text = re.sub("cu nt", "cunt", text)
    #text = re.sub("idi ot", "idiot", text)
    text = re.sub("(?<=\\b(fu|su|di|co|li))\\s(?=(ck)\\b)", "", text)
    text = re.sub("(?<=\\w(ck))\\s(?=(ing)\\b)", "", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"j k", "jk", text)    
    text = re.sub(r" u ", " you ", text)

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in string.punctuation])

    #Split the sentences into words
    


    # Optionally, remove stop words
    if remove_stop_words:
        words=tokenizer.tokenize(comment)
        words=[lem.lemmatize(word, "v") for word in words]
        text = [w for w in words if not w in eng_stopwords]
        #text = text.split()
        #text = [w for w in text if not w in STOPWORDS_SET]
        text = " ".join(text)
    


    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    return text
    