import spacy
import os, re
from sample import lst
from bs4 import BeautifulSoup
from nltk import sent_tokenize

nlp = spacy.load("en_core_web_sm")

# lst = ["<Leebookmark1>I like Apple <skip>because</skip> from spain his pichai ceo post in google. He also studied in <skip>IIT</skip> usa.</Leebookmark1>","<Leebookmark2>the <skip>from</skip> america will fload in wind against Mangalore. And famous for dogs and <skip>is</skip> cats.</Leebookmark2>","<Leebookmark3>India has the chemical formula <skip>C2H5OH</skip> and it is famous for Ethanol.</Leebookmark3>"]

out_file = open("output.txt","w")
i = 1
for le in lst:
    pattern1 = "<Leebookmark{}>".format(i)
    pattern2 = "</Leebookmark{}>".format(i)
    print(pattern1,"-------------------------------------------",pattern2)
    text = le.replace(pattern1,"")
    text = text.replace(pattern2,"")
    # print(text)
    out_list = []
    for sent in sent_tokenize(text):
        print(sent)
        skip_start_end = [re.search(r"<skip>(.*?)</skip>",sent).span()[0]+6, re.search(r"<skip>(.*?)</skip>",sent).span()[1]-7]
        difference_between_sent = skip_start_end[-1]-skip_start_end[0]
        print(difference_between_sent)
        text_without_skip = re.sub(r"<skip>(.*?)</skip> ","",sent) # flaw 1
        print(text_without_skip)
        doc = nlp(text_without_skip) # flaw 2
        # result = [{"text":ent.text,"position":[ent.start_char, ent.end_char]} for ent in doc.ents]
        # print(result)
        for ent in doc.ents:
            if ent.start_char >= skip_start_end[0]:
                position = [ent.start_char+difference_between_sent+1, ent.end_char+difference_between_sent+1]
                print("..........",{"text":ent.text,"position":[ent.start_char+difference_between_sent+1, ent.end_char+difference_between_sent+1]})
                #----------------------------------------------------------------------------------------------------------------------------------------------flaw 3
                out_list.append("<word>{}</word><start>{}</start><end>{}</end><message>PNA: Change the capitalization of the word</message>".format(ent.text,position[0],position[1]))
                # out_result = "{}<correction>".format(pattern1)+ "".join(out_list) +"</correction>{}".format(pattern2)
                # print(out_result)
                # out_file.write(out_result)
            else:
                print("..........",{"text":ent.text,"position":[ent.start_char, ent.end_char]})
                position = [ent.start_char, ent.end_char]
                # print("..........",{"text":ent.text,"position":[ent.start_char+difference_between_sent+1, ent.end_char+difference_between_sent+1]})
                out_list.append("<word>{}</word><start>{}</start><end>{}</end><message>PNA: Change the capitalization of the word</message>".format(ent.text,position[0],position[1]))
                # out_result = "{}<correction>".format(pattern1)+ "".join(out_list) +"</correction>{}".format(pattern2)
                # print(out_result)
                # out_file.write(out_result)
        print(out_list)
        out_result = "{}<correction>".format(pattern1)+ "".join(out_list) +"</correction>{}".format(pattern2)
    out_file.write(out_result)
    i+=1








# i = 1
# for le in lst:
#     pattern = r"<Leebookmark{}>(.*?)</Leebookmark{}>".format(i,i)
#     print(re.search(pattern, le).group())
#     i+=1
#     # for sent in sent_tokenize()







































































































#
