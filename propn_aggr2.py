import spacy
import os, re
import base64
from nltk import sent_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode

stop = list(set(stopwords.words('english')))

nlp_old = spacy.load(r"C:\Users\0016040\Documents\VH\sps_projects\proper_noun_ner_training\ner_trainer_v2\models\propn\PROPN_MODEL_thr_3_DATED_2019-08-16")
nlp_latest = spacy.load(r"C:\Users\0016040\Documents\VH\sps_projects\proper_noun_ner_training\ner_trainer_v2\models\propn\PROPN_MODEL_G_SHEET_thr_2_itr_30_DATED_2019-09-19")


text = "The ELF was developed by the deck Aerospace Exploration Agency japanese and installed in the KIBO deck in the ISS."

decoded_file_path = r"C:\Users\0016040\Documents\VH\sps_projects\proper_noun_ner_training\proper_noun_casing_aggregation\temp\Sample50_input_encode.txt"


punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~‘’“”'

def pre_process(text, punc):
    for i in punc:
        if i in text:
            text = text.replace(i," ")
    return text



def get_propn_corrections_v2(decoded_file_path):
    try:
        leebm_lst = open(decoded_file_path, "r", encoding='utf-8-sig').readlines()
    except:
        leebm_lst = open(decoded_file_path, "r", encoding='utf-8').readlines()
    leebm_lst_filter = list(map(lambda x:x.strip(),leebm_lst))
    out_file = open("output.txt","w")
    i = 1
    output = []
    file_sentences = []
    for le in leebm_lst_filter:
        if le:# != '\n' and le != " ":
            # print(le)
            file_sentences.append(le)
            pattern1 = "<LEBookMark{}>".format(i)
            pattern2 = "</LEBookMark{}>".format(i)
            # print(pattern1,"-------------------------------------------",pattern2)
            text = re.sub(pattern1, "", le)#le.replace(pattern1,"")
            text = re.sub(pattern2, "", text)#le.replace(pattern1,"")
            # print(text)
            out_list = []
            # print(len(sent_tokenize(text)))
            for sent in sent_tokenize(text):
                print(pattern1,"========",sent)
                try:
                    skip_start_end = [re.search(r"<skip>(.*?)</skip>",sent).span()[0]+6, re.search(r"<skip>(.*?)</skip>",sent).span()[1]-7]
                    difference_between_sent = skip_start_end[-1]-skip_start_end[0]
                    # print(difference_between_sent)
                    text_without_skip = re.sub(r"<skip>(.*?)</skip>","",sent) # flaw 1
                    print("-----",sent)
                    print("text_without_skip:", text_without_skip)
                    lower_txt = text_without_skip.lower()
                    doc = nlp_latest(pre_process(lower_txt, punc)) # flaw 2
                    # result = [{"text":ent.text,"position":[ent.start_char, ent.end_char]} for ent in doc.ents]
                    # print(result)
                    should_be_corrected_propns = []
                    for j in range(len(doc.ents)):
                        try:
                            if re.search(doc.ents[j].text,text_without_skip).group() and doc.ents[j].text not in stop:
                                # print(re.search(ent.text, text_without_skip).group())
                                # print("-----",re.search(doc.ents[j].text, text_without_skip))
                                should_be_corrected_propns.extend(list(re.finditer(doc.ents[j].text,text_without_skip)))
                        except:
                            pass
                    result_proper_nouns = [{"text":k.group(),"position":list(k.span())} for k in should_be_corrected_propns]
                    # print(result_proper_nouns)
                    corrections = []
                    for propn in result_proper_nouns:
                        if propn['position'][0] >= skip_start_end[0]:
                            positions = [propn['position'][0]+difference_between_sent+1,propn['position'][1]+difference_between_sent+1]
                            # print("..........",{"text":propn['text'],"position":[positions[0], positions[1]]})
                            out_list.append("<word>{}</word><start>{}</start><end>{}</end><message>PNC: Change the capitalization for the word {}</message>".format(propn['text'], positions[0], positions[1], propn['text']))
                            corrections.append({"sentence":sent,"index":positions})
                        else:
                            positions = [propn['position'][0], propn['position'][1]]
                            # print("..........",{"text":propn['text'],"position":[positions[0], positions[1]]})
                            out_list.append("<word>{}</word><start>{}</start><end>{}</end><message>PNC: Change the capitalization for the word {}</message>".format(propn['text'], positions[0], positions[1], propn['text']))
                            corrections.append({"sentence":sent,"index":positions})
                    # print(corrections)
                except Exception as e:
                    # print(e)
                    corrections = []
                    pass
            if out_list:
                out_result = "{}<correction>".format(pattern1)+ "".join(out_list) +"</correction>{}".format(pattern2)
            else:
                out_result = "{}{}".format(pattern1, pattern2)
                # out_file.write(str(out_result))
            i+=1
            output.append(out_result)
        total_corrections = ("".join(output)).count("PNC: Change the capitalization for the word")
    final_output = unidecode("".join(output))
    return [final_output, total_corrections, file_sentences]


print(get_propn_corrections_v2(r"C:\Users\0016040\Documents\VH\sps_projects\proper_noun_ner_training\proper_noun_casing_aggregation\temp\pnc_Sample3_input_encode.txt")[0])





















#
