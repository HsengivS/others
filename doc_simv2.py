from nltk.corpus import wordnet
from nltk import word_tokenize
import pandas as pd
from scipy import sparse
import time
import pickle
import re
from cleantext import clean
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


start = time.time()

stop_words = list(set(stopwords.words('english')))

blacklist_stopwords = ['not',"haven't","unable","to","break","down"]

stop_words = [i for i in stop_words if i not in blacklist_stopwords]

porter_stemmer = PorterStemmer()

noisy_tokens = ['from','to','sent','rgds','subject','sub','re','mailto','Timestamp','CC','Cc','cc','timestamp']
stop_lst = noisy_tokens + stop_words +['teammy','hello','helloi','himy','sirmam','hi','team','thanks','regards','dear','ril','vice president','navi mumbai','Vice president','chief','reliance','navi mumbai', 'reliance','mumbai','Bombay','bombay','jio','jio','jio infocomm','please', 'location','Ext','manager','chembur','Contact no:','phone no','Administrator','administrator','Please','Kindly', 'kindly','teampls','teamplease','hii','sir','teamkindly','team','teamrequest','madammy','madamplease', 'MadamRequest','Madamplz','plz','teamplz','teampls','hikindly','i','my','sirplease','mam','teami','helloplease','pls','siri','am','hiplease','good','morning']

# contraction_patterns = [ (r'won't', 'will not'), (r'can't', 'cannot'), (r'i'm', 'i am'), (r'ain't', 'is not'), (r'(\w+)'ll', '\g<1> will'), (r'(\w+)n't', '\g<1> not'), (r'(\w+)'ve', '\g<1> have'), (r'(\w+)'s', '\g<1> is'), (r'(\w+)'re', '\g<1> are'), (r'(\w+)'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]
contraction_patterns = [ (r"won't", "will not"), (r"can't", "cannot"), (r"i'm", "i am"), (r"ain't", "is not"), (r"(\w+)'ll", "\g<1> will"), (r"(\w+)n't", "\g<1> not"), (r"(\w+)'ve", "\g<1> have"), (r"(\w+)'s", "\g<1> is"), (r"(\w+)'re", "\g<1> are"), (r"(\w+)'d", "\g<1> would"), (r"&", "and"), (r"dammit", "damn it"), (r"dont", "do not"), (r"wont", "will not") ]



def pre_process_1(txt): # replace contractions
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (txt, count) = re.subn(pattern, repl, txt)
    txt = re.sub(r"(Sent|sent): \d\d +[A-Za-z]+ +\d\d\d\d +\d\d:\d\d"," TIMESTAMP ",txt)
    txt = re.sub(r"\d+[.]\d+[.]\d+[.]\d+"," IP_ADDR ",txt)
    txt = re.sub(r"[[cid:image\d+.png@[A-Z0-9]+.[A-Z0-9]+]"," IMAGE ",txt)
    txt = re.sub(r"regards+(.)|Regards+(.)|regard+(.)|Warm+\s+Regards+(.)"," ",txt)
    txt = re.sub(r"Thank+\s+|Thanks+\s+|thanks+\s+|thanks+\s+"," ",txt)
    txt = re.sub(r"[^\w\s]+|[^\w\s]"," ", txt)
    for j in [1,2,3]:
        for i in txt.split():
            m = re.findall(r"[a-z][A-Z][a-z]+",i)
            for j in m:
                txt = (re.sub(j,j[0]+" "+j[1:],txt))
    return txt

def pre_procee_2(txt):
#     print(type(txt),txt)
    txt = re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});'," ",txt)
    txt = clean(pre_process_1(txt),
                fix_unicode=True, # fix various unicode errors
                to_ascii=True, # transliterate to closest ASCII representation
                lower=True, # lowercase text
                no_line_breaks=True, # fully strip line breaks as opposed to only normalizing them
                no_urls=True, # replace all URLs with a special token
                no_emails=True, # replace all email addresses with a special token
                no_phone_numbers=True, # replace all phone numbers with a special token
                no_numbers=True, # replace all numbers with a special token
                no_digits=False, # replace all digits with a special token
                no_currency_symbols=True, # replace all currency symbols with a special token
                no_punct=False, # fully remove punctuation
                replace_with_url="URL>",
                replace_with_email="EMAIL",
                replace_with_phone_number="PHONE",
                replace_with_number="NUMBER",
                replace_with_digit="0",
                replace_with_currency_symbol="CURR",
                lang="en") # set to 'de' for German special handling)
#     txt = [porter_stemmer.stem(i) for i in word_tokenize(str(txt).lower())]
    txt = [i for i in word_tokenize(str(txt).lower()) if i not in stop_lst]
    txt = [w for w in word_tokenize(str(txt).lower()) if not w in stop_words]
    txt = ' '.join(txt)
    txt.translate(str.maketrans('', '', string.punctuation))
    txt = re.sub(r'[^\w\s]','',txt)
    txt = re.sub(r'_+','',txt)
    txt = re.sub(r"\s+"," ",txt)
    txt = ''.join(txt)
    txt = txt.strip()
    return txt


def time_taken(start):
    return time.time()-start


def get_tf_idf_query_similarity(documents, query):
    allDocs = []
    start = time.time()
    for document in documents:
        allDocs.append((document))
    print("1 appending all docs:", time_taken(start))
#     docTFIDF = TfidfVectorizer().fit_transform(allDocs)
    docTFIDF = sparse.load_npz("doc_tfidf.npz") ###################### here
    print("2 docTfidf:",time_taken(start))
    queryTFIDF = pickle.load(open("queryTFIDF.pkl", 'rb')) ########################## here
#     queryTFIDF = TfidfVectorizer().fit(allDocs)
    queryTFIDF = TfidfVectorizer(vocabulary = queryTFIDF.vocabulary_)
    print("3 query tfidf:",time_taken(start))
    queryTFIDF = queryTFIDF.fit_transform([query])
    print('4 query TFIDF:',time_taken(start))
    cosineSimilarities = cosine_similarity(queryTFIDF, docTFIDF).flatten()
    print('5 cosine similarity:',time_taken(start))
    return cosineSimilarities


def get_similar_doc_term(text, df, df_cleaned, threshold):
    targets = ['category','sub_category','area','sub_area','assignment_group']
    text = pre_procee_2(text)
    cosine_values = list(get_tf_idf_query_similarity(df_cleaned,text))
    arr_of_obj = []
    print('cosine values done')
    for i, c in enumerate(cosine_values):
        arr_of_obj.append({"index":i,"cosine":c})
    sorted_top_cosines = sorted(arr_of_obj, key=lambda x: x['cosine'])[::-1][0:3]
    print(sorted_top_cosines)
    for i in sorted_top_cosines:
        if i.get('cosine') <= threshold:
            return [{'category': 'un-identified'},
                    {'sub_category': 'un-identified'},
                    {'area': 'un-identified'},
                    {'sub_area': 'un-identified'},
                    {'assignment_group': 'un-identified'}]
        break
    predictions = [{k:v} for k, v in dict(df.iloc[sorted_top_cosines[0].get('index'),:]).items() if k in targets]
    return predictions



# def get_tf_idf_query_similarity(documents, query):
#     allDocs = []
#     for document in documents:
#         allDocs.append((document))
#     docTFIDF = TfidfVectorizer().fit_transform(allDocs)
#     queryTFIDF = TfidfVectorizer().fit(allDocs)
#     queryTFIDF = queryTFIDF.transform([query])
#
#     cosineSimilarities = cosine_similarity(queryTFIDF, docTFIDF).flatten()
#     return cosineSimilarities
#
#
# def get_similar_doc_term(text, df, threshold):
#     targets = ['category','sub_category','area','sub_area','assignment_group']
#     text = pre_procee_2(text)
#     cosine_values = list(get_tf_idf_query_similarity(df['mail_description'],text))
#     arr_of_obj = []
#     for i, c in enumerate(cosine_values):
#         arr_of_obj.append({"index":i,"cosine":c})
#     # print(arr_of_obj)
#     sorted_top_cosines = sorted(arr_of_obj, key=lambda x: x['cosine'])[::-1][0:3]
#     print(sorted_top_cosines)
#     for i in sorted_top_cosines:
#         if i.get('cosine') <= threshold:
#             return [{'category': 'un-identified'},
#                     {'sub_category': 'un-identified'},
#                     {'area': 'un-identified'},
#                     {'sub_area': 'un-identified'},
#                     {'assignment_group': 'un-identified'}]
#         break
#     predictions = [{k:v} for k, v in dict(df.iloc[sorted_top_cosines[0].get('index'),:]).items() if k in targets]
#     return predictions

# text = """Hi TeamKindly raise the ticket to clear port security at 3FFC and assign it to Enterprise LAN & Wi-Fi -Suraj Jadhav RegardsSuraj JadhavEnterprise Network Operations - LAN & Wi-Fi RCP-2B First Floor 2FFCA13Direct Landline: - +91-22-796 72200/ 447 72200Board Number: - +91-22- 796 70000/447 70000 Extension 72200Email Id: - Enterprise.LANWifi@ril.com ----------------------------------------------------------------------------------------------------------------------------------- Important note:For Enterprise LAN & Wi-Fi access please register your Service request through GETIT portal only. GETIT: https://getit.ril.com Our email id has been migrated from rcp.network@zmail.ril.com to Enterprise.LANWifi@ril.com Henceforth kindly forward all your concern email to Enterprise.LANWifi@ril.com. ------------------------------------------------------------------------------------------------------------------------------------"""
# # df.iloc[3,:][-3]
# get_similar_doc_term(text, df, 0.7)
