import csv
import sys
import string
from math import log10, sqrt, ceil
import pandas as pd
import os
from math import log10, sqrt
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import mean_squared_error



#Read all csv files
train_csv = open('C:/Users/Hardik/Downloads/Sarika/train.csv',encoding='charmap')
test_csv = open('C:/Users/Hardik/Downloads/Sarika/test.csv',encoding='charmap')
attributes_csv = open('C:/Users/Hardik/Downloads/Sarika/attributes.csv',encoding='charmap')
product_descriptions_csv = open('C:/Users/Hardik/Downloads/Sarika/product_descriptions.csv',encoding='charmap')

#dictionary variables
title_dict={}
brand_dict={}
material_dict={}
function_dict={}
vocab_dict={}
product_dict={}
product_description_dict={}
relevance_dict={}
id_uid_dict={}
title_dict={}
initial_dictionary_all={}
initial_list_all=[]
search_term_dict={}
product_list=[]
vocab=0
search_term_vocab=[]
product_title_vocab=[]
product_descriptions_vocab=[]
brand_vocab=[]
material_vocab=[]
function_vocab=[]
sent=0
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
sw=stopwords.words("english")
stemmer = PorterStemmer()

frequency_matrix={}
search_term_freq_matrix={}


train_csv_row_count=sum(1 for row in train_csv)
train_csv.seek(0)
print(train_csv_row_count)
training_set_row_count=ceil((2/3)*train_csv_row_count)
print(training_set_row_count)


f_tokens=[]

def tokenize_stem(sent):
    tokens = tokenizer.tokenize(sent)
    lowercase = [token.lower() for token in tokens]
    f_tokens=[stemmer.stem(token) for token in lowercase if not token in sw]
    return f_tokens


#creaing test dataframe from train.csv
f1=pd.read_csv('C:/Users/Hardik/Downloads/Sarika/train.csv',encoding='charmap')
test_df=pd.DataFrame(f1[training_set_row_count:train_csv_row_count])

#creating train dataframe from train.csv
f2=pd.read_csv('C:/Users/Hardik/Downloads/Sarika/train.csv',encoding='charmap')
train_df=pd.DataFrame(f2[0:training_set_row_count])

#creating attribute dataframe from attribute.csv
f3=pd.read_csv('C:/Users/Hardik/Downloads/Sarika/attributes.csv',encoding='charmap')
attributes_df=pd.DataFrame(f3)

#creating pdesc dataframe from product_descriptions.csv
f4=pd.read_csv('C:/Users/Hardik/Downloads/Sarika/product_descriptions.csv',encoding='charmap')
product_descriptions_df=pd.DataFrame(f4)

#creating full train.csv dataframe
f5=pd.read_csv('C:/Users/Hardik/Downloads/Sarika/train.csv',encoding='charmap')
train_full_df=pd.DataFrame(f5)

#creating full test.csv dataframe
f6=pd.read_csv('C:/Users/Hardik/Downloads/Sarika/test.csv',encoding='charmap')
test_full_df=pd.DataFrame(f6)

print("file reading done pandas")
# ID-RELEVANCE and ID-PRODUCT UID dictionary
train_csv.seek(0)
for i in train_df.itertuples():
    relevance_dict[i[1]]=str(i[5])
    id_uid_dict[i[1]]=i[2]



# BRAND, MATERIAL, FUNCTION DICT
for i in attributes_df.itertuples():
    if i[2] == '"MFG Brand Name"':
        brand_dict[i[1]]=i[3]
    if str(i[2]).__contains__('Material'):
       material_dict[i[1]]=i[3]
    if str(i[2]).__contains__('Function') and not str(i[3]).__contains__('No'):
        function_dict[i[1]]= str(i[2]) + str(i[3])

#PRODUCT DESCREPTION DICT
for i in product_descriptions_df.itertuples():
    product_description_dict[i[1]]= i[2]
product_descriptions_csv.seek(0)

#PRODUCT UID LIST
product_uid_list=[]
product_uid_list.extend(i.split('0') for i in product_descriptions_csv)
product_descriptions_csv.seek(0)


#PRODUCT ID TITLE DICT
train_csv.seek(0)
for i in train_df.itertuples():
    title_dict[i[2]]=i[3]



#SEARCH TERM DICT (ID-SEACRH TERM)
train_csv.seek(0)
for i in train_df.itertuples():
    search_term_dict[i[1]]=i[4]


for i in train_df._getitem_column('id'):
    initial_list_all=[tokenize_stem(search_term_dict[i]),tokenize_stem(str(title_dict[id_uid_dict[i]])),tokenize_stem(str(product_description_dict[id_uid_dict[i]])) ,tokenize_stem(str(brand_dict[id_uid_dict[i]])) if id_uid_dict[i] in brand_dict.keys() else tokenize_stem('None'),tokenize_stem(str(material_dict[id_uid_dict[i]])) if id_uid_dict[i] in material_dict.keys() else tokenize_stem('None'),tokenize_stem(str(function_dict[id_uid_dict[i]])) if id_uid_dict[i] in function_dict.keys() else tokenize_stem('None'),relevance_dict[i]]
    initial_dictionary_all[i]=initial_list_all

model_dict_all={}
model_dict_all[0]=[0]


for i in initial_dictionary_all:
    count_ptitle=0
    count_pdesc=0
    brand_flag=0
    material_flag=0
    function_flag=0
    counter_search_term = Counter(initial_dictionary_all[i][0])
    counter_ptitle=Counter(initial_dictionary_all[i][1])
    counter_pdesc=Counter(initial_dictionary_all[i][2])
    #print(counter_search_term)
    for word in initial_dictionary_all[i][0]:
        if word in counter_ptitle:
            count_ptitle+=counter_ptitle[word]
        if word in counter_pdesc:
            count_pdesc+=counter_pdesc[word]
        if word in initial_dictionary_all[i][3]:
            brand_flag=1
        if word in initial_dictionary_all[i][4]:
            material_flag=1
        if word in initial_dictionary_all[i][5]:
            function_flag=1
    if count_ptitle>=len(initial_dictionary_all[i][0]):
        model_dict_all[i]=['Perfect']
    elif count_ptitle<len(initial_dictionary_all[i][0]) and count_ptitle!=0:
        model_dict_all[i]=['Partial']
    else:
        model_dict_all[i]=['Irrelevant']
    if count_pdesc>=len(initial_dictionary_all[i][0]):
        model_dict_all[i].append('Perfect')
    elif count_pdesc<len(initial_dictionary_all[i][0]):
        model_dict_all[i].append('Partial')
    else:
        model_dict_all[i].append('Irrelevant')
    if brand_flag==1:
        model_dict_all[i].append('Yes')
    else:
        model_dict_all[i].append('No')
    if material_flag==1:
        model_dict_all[i].append('Yes')
    else:
        model_dict_all[i].append('No')
    if function_flag==1:
        model_dict_all[i].append('Yes')
    else:
        model_dict_all[i].append('No')
    model_dict_all[i].append(initial_dictionary_all[i][6])

del(model_dict_all[0])
#print(model_dict_all)


count_rel_1=0
count_rel_1_25=0
count_rel_1_33=0
count_rel_1_5=0
count_rel_1_67=0
count_rel_1_75=0
count_rel_2=0
count_rel_2_25=0
count_rel_2_33=0
count_rel_2_5=0
count_rel_2_67=0
count_rel_2_75=0
count_rel_3=0

count_title_perfect=0
count_title_partial=0
count_title_irrelevant=0

count_title_perfect_1=0
count_title_perfect_1_25=0
count_title_perfect_1_33=0
count_title_perfect_1_5=0
count_title_perfect_1_67=0
count_title_perfect_1_75=0
count_title_perfect_2=0
count_titlel_perfect_2_25=0
count_title_perfect_2_33=0
count_title_perfect_2_5=0
count_title_perfect_2_67=0
count_title_perfect_2_75=0
count_title_perfect_3=0
count_title_partial_1=0
count_title_partial_1_25=0
count_title_partial_1_33=0
count_title_partial_1_5=0
count_title_partial_1_67=0
count_title_partial_1_75=0
count_title_partial_2=0
count_titlel_partial_2_25=0
count_title_partial_2_33=0
count_title_partial_2_5=0
count_title_partial_2_67=0
count_title_partial_2_75=0
count_title_partial_3=0
count_title_irrelevant_1=0
count_title_irrelevant_1_25=0
count_title_irrelevant_1_33=0
count_title_irrelevant_1_5=0
count_title_irrelevant_1_67=0
count_title_irrelevant_1_75=0
count_title_irrelevant_2=0
count_titlel_irrelevant_2_25=0
count_title_irrelevant_2_33=0
count_title_irrelevant_2_5=0
count_title_irrelevant_2_67=0
count_title_irrelevant_2_75=0
count_title_irrelevant_3=0

count_desc_perfect=0
count_desc_partial=0
count_desc_irrelevant=0

count_desc_perfect_1=0
count_desc_perfect_1_25=0
count_desc_perfect_1_33=0
count_desc_perfect_1_5=0
count_desc_perfect_1_67=0
count_desc_perfect_1_75=0
count_desc_perfect_2=0
count_descl_perfect_2_25=0
count_desc_perfect_2_33=0
count_desc_perfect_2_5=0
count_desc_perfect_2_67=0
count_desc_perfect_2_75=0
count_desc_perfect_3=0
count_desc_partial_1=0
count_desc_partial_1_25=0
count_desc_partial_1_33=0
count_desc_partial_1_5=0
count_desc_partial_1_67=0
count_desc_partial_1_75=0
count_desc_partial_2=0
count_descl_partial_2_25=0
count_desc_partial_2_33=0
count_desc_partial_2_5=0
count_desc_partial_2_67=0
count_desc_partial_2_75=0
count_desc_partial_3=0
count_desc_irrelevant_1=0
count_desc_irrelevant_1_25=0
count_desc_irrelevant_1_33=0
count_desc_irrelevant_1_5=0
count_desc_irrelevant_1_67=0
count_desc_irrelevant_1_75=0
count_desc_irrelevant_2=0
count_descl_irrelevant_2_25=0
count_desc_irrelevant_2_33=0
count_desc_irrelevant_2_5=0
count_desc_irrelevant_2_67=0
count_desc_irrelevant_2_75=0
count_desc_irrelevant_3=0

count_brand_yes_1=0
count_brand_yes_1_25=0
count_brand_yes_1_33=0
count_brand_yes_1_5=0
count_brand_yes_1_67=0
count_brand_yes_1_75=0
count_brand_yes_2=0
count_brandl_yes_2_25=0
count_brand_yes_2_33=0
count_brand_yes_2_5=0
count_brand_yes_2_67=0
count_brand_yes_2_75=0
count_brand_yes_3=0

count_brand_no_1=0
count_brand_no_1_25=0
count_brand_no_1_33=0
count_brand_no_1_5=0
count_brand_no_1_67=0
count_brand_no_1_75=0
count_brand_no_2=0
count_brandl_no_2_25=0
count_brand_no_2_33=0
count_brand_no_2_5=0
count_brand_no_2_67=0
count_brand_no_2_75=0
count_brand_no_3=0

count_material_yes_1=0
count_material_yes_1_25=0
count_material_yes_1_33=0
count_material_yes_1_5=0
count_material_yes_1_67=0
count_material_yes_1_75=0
count_material_yes_2=0
count_materiall_yes_2_25=0
count_material_yes_2_33=0
count_material_yes_2_5=0
count_material_yes_2_67=0
count_material_yes_2_75=0
count_material_yes_3=0

count_material_no_1=0
count_material_no_1_25=0
count_material_no_1_33=0
count_material_no_1_5=0
count_material_no_1_67=0
count_material_no_1_75=0
count_material_no_2=0
count_materiall_no_2_25=0
count_material_no_2_33=0
count_material_no_2_5=0
count_material_no_2_67=0
count_material_no_2_75=0
count_material_no_3=0

count_function_yes_1=0
count_function_yes_1_25=0
count_function_yes_1_33=0
count_function_yes_1_5=0
count_function_yes_1_67=0
count_function_yes_1_75=0
count_function_yes_2=0
count_functionl_yes_2_25=0
count_function_yes_2_33=0
count_function_yes_2_5=0
count_function_yes_2_67=0
count_function_yes_2_75=0
count_function_yes_3=0

count_function_no_1=0
count_function_no_1_25=0
count_function_no_1_33=0
count_function_no_1_5=0
count_function_no_1_67=0
count_function_no_1_75=0
count_function_no_2=0
count_functionl_no_2_25=0
count_function_no_2_33=0
count_function_no_2_5=0
count_function_no_2_67=0
count_function_no_2_75=0
count_function_no_3=0

p={}
N=len(model_dict_all)
for i in model_dict_all:
    if model_dict_all[i][5]=='1.0':
        count_rel_1+=1
    p['1.0']=((count_rel_1+1)/(N+13))
    if model_dict_all[i][5]=='1.25':
        count_rel_1_25+=1
    p['1.25']=((count_rel_1_25+1)/(N+13))
    if model_dict_all[i][5]=='1.33':
        count_rel_1_33+=1
    p['1.33']=((count_rel_1_33+1)/(N+13))
    if model_dict_all[i][5]=='1.5':
        count_rel_1_5+=1
    p['1.5']=((count_rel_1_5+1)/(N+13))
    if model_dict_all[i][5]=='1.67':
        count_rel_1_67+=1
    p['1.67']=((count_rel_1_67+1)/(N+13))
    if model_dict_all[i][5]=='1.75':
        count_rel_1_75+=1
    p['1.75']=((count_rel_1_75+1)/(N+13))
    if model_dict_all[i][5]=='2.0':
        count_rel_2+=1
    p['2.0']=((count_rel_2+1)/(N+13))
    if model_dict_all[i][5]=='2.25':
        count_rel_2_25+=1
    p['2.25']=((count_rel_2_25+1)/(N+13))
    if model_dict_all[i][5]=='2.33':
        count_rel_2_33+=1
    p['2.33']=((count_rel_2_33+1)/(N+13))
    if model_dict_all[i][5]=='2.5':
        count_rel_2_5+=1
    p['2.5']=((count_rel_2_5+1)/(N+13))
    if model_dict_all[i][5]=='2.67':
        count_rel_2_67+=1
    p['2.67']=((count_rel_2_67+1)/(N+13))
    if model_dict_all[i][5]=='2.75':
        count_rel_2_75+=1
    p['2.75']=((count_rel_2_75+1)/(N+13))
    if model_dict_all[i][5]=='3.0':
        count_rel_3+=1
    p['3.0']=((count_rel_3+1)/(N+13))



    if model_dict_all[i][5]=='1.0' and model_dict_all[i][1]=='Perfect':
        count_title_perfect_1+=1
    p['titlePerfect1.0']=((count_title_perfect_1+1)/(count_rel_1+13))
    if model_dict_all[i][5]=='1.25' and model_dict_all[i][1]=='Perfect':
        count_title_perfect_1_25+=1
    p['titlePerfect1.25']=((count_title_perfect_1_25+1)/(count_rel_1_25+13))
    if model_dict_all[i][5]=='1.33' and model_dict_all[i][1]=='Perfect':
        count_title_perfect_1_33+=1
    p['titlePerfect1.33']=((count_title_perfect_1_33+1)/(count_rel_1_33+13))
    if model_dict_all[i][5]=='1.5' and model_dict_all[i][1]=='Perfect':
        count_title_perfect_1_5+=1
    p['titlePerfect1.5']=((count_title_perfect_1_5+1)/(count_rel_1_5+13))
    if model_dict_all[i][5]=='1.67' and model_dict_all[i][1]=='Perfect':
        count_title_perfect_1_67+=1
    p['titlePerfect1.67']=((count_title_perfect_1_67+1)/(count_rel_1_67+13))
    if model_dict_all[i][5]=='1.75' and model_dict_all[i][1]=='Perfect':
        count_title_perfect_1_75+=1
    p['titlePerfect1.75']=((count_title_perfect_1_75+1)/(count_rel_1_75+13))
    if model_dict_all[i][5]=='2.0' and model_dict_all[i][1]=='Perfect':
        count_title_perfect_2+=1
    p['titlePerfect2.0']=((count_title_perfect_2+1)/(count_rel_2+13))
    if model_dict_all[i][5]=='2.25' and model_dict_all[i][1]=='Perfect':
        count_titlel_perfect_2_25+=1
    p['titlePerfect2.25']=((count_titlel_perfect_2_25+1)/(count_rel_2_25+13))
    if model_dict_all[i][5]=='2.33' and model_dict_all[i][1]=='Perfect':
        count_title_perfect_2_33+=1
    p['titlePerfect2.33']=((count_title_perfect_2_33+1)/(count_rel_2_33+13))
    if model_dict_all[i][5]=='2.5' and model_dict_all[i][1]=='Perfect':
        count_title_perfect_2_5+=1
    p['titlePerfect2.5']=((count_title_perfect_2_5+1)/(count_rel_2_5+13))
    if model_dict_all[i][5]=='2.67' and model_dict_all[i][1]=='Perfect':
        count_title_perfect_2_67+=1
    p['titlePerfect2.67']=((count_title_perfect_2_67+1)/(count_rel_2_67+13))
    if model_dict_all[i][5]=='2.75' and model_dict_all[i][1]=='Perfect':
        count_title_perfect_2_75+=1
    p['titlePerfect2.75']=((count_title_perfect_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all[i][5]=='3.0' and model_dict_all[i][1]=='Perfect':
        count_title_perfect_3+=1
    p['titlePerfect3.0']=((count_title_perfect_3+1)/(count_rel_3+13))


    if model_dict_all[i][5]=='1.0' and model_dict_all[i][1]=='Partial':
        count_title_partial_1+=1
    p['titlePartial1.0']=((count_title_partial_1+1)/(count_rel_1+13))
    if model_dict_all[i][5]=='1.25' and model_dict_all[i][1]=='Partial':
        count_title_partial_1_25+=1
    p['titlePartial1.25']=((count_title_partial_1_25+1)/(count_rel_1_25+13))
    if model_dict_all[i][5]=='1.33' and model_dict_all[i][1]=='Partial':
        count_title_partial_1_33+=1
    p['titlePartial1.33']=((count_title_partial_1_33+1)/(count_rel_1_33+13))
    if model_dict_all[i][5]=='1.5' and model_dict_all[i][1]=='Partial':
        count_title_partial_1_5+=1
    p['titlePartial1.5']=((count_title_partial_1_5+1)/(count_rel_1_5+13))
    if model_dict_all[i][5]=='1.67' and model_dict_all[i][1]=='Partial':
        count_title_partial_1_67+=1
    p['titlePartial1.67']=((count_title_partial_1_67+1)/(count_rel_1_67+13))
    if model_dict_all[i][5]=='1.75' and model_dict_all[i][1]=='Partial':
        count_title_partial_1_75+=1
    p['titlePartial1.75']=((count_title_partial_1_75+1)/(count_rel_1_75+13))
    if model_dict_all[i][5]=='2.0' and model_dict_all[i][1]=='Partial':
        count_title_partial_2+=1
    p['titlePartial2.0']=((count_title_partial_2+1)/(count_rel_2+13))
    if model_dict_all[i][5]=='2.25' and model_dict_all[i][1]=='Partial':
        count_titlel_partial_2_25+=1
    p['titlePartial2.25']=((count_titlel_partial_2_25+1)/(count_rel_2_25+13))
    if model_dict_all[i][5]=='2.33' and model_dict_all[i][1]=='Partial':
        count_title_partial_2_33+=1
    p['titlePartial2.33']=((count_title_partial_2_33+1)/(count_rel_2_33+13))
    if model_dict_all[i][5]=='2.5' and model_dict_all[i][1]=='Partial':
        count_title_partial_2_5+=1
    p['titlePartial2.5']=((count_title_partial_2_5+1)/(count_rel_2_5+13))
    if model_dict_all[i][5]=='2.67' and model_dict_all[i][1]=='Partial':
        count_title_partial_2_67+=1
    p['titlePartial2.67']=((count_title_partial_2_67+1)/(count_rel_2_67+13))
    if model_dict_all[i][5]=='2.75' and model_dict_all[i][1]=='Partial':
        count_title_partial_2_75+=1
    p['titlePartial2.75']=((count_title_partial_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all[i][5]=='3.0' and model_dict_all[i][1]=='Partial':
        count_title_partial_3+=1
    p['titlePartial3.0']=((count_title_partial_3+1)/(count_rel_3+13))


    if model_dict_all[i][5]=='1.0' and model_dict_all[i][1]=='Irrelevant':
        count_title_irrelevant_1+=1
    p['titleIrrelevant1.0']=((count_title_irrelevant_1+1)/(count_rel_1+13))
    if model_dict_all[i][5]=='1.25' and model_dict_all[i][1]=='Irrelevant':
        count_title_irrelevant_1_25+=1
    p['titleIrrelevant1.25']=((count_title_irrelevant_1_25+1)/(count_rel_1_25+13))
    if model_dict_all[i][5]=='1.33' and model_dict_all[i][1]=='Irrelevant':
        count_title_irrelevant_1_33+=1
    p['titleIrrelevant1.33']=((count_title_irrelevant_1_33+1)/(count_rel_1_33+13))
    if model_dict_all[i][5]=='1.5' and model_dict_all[i][1]=='Irrelevant':
        count_title_irrelevant_1_5+=1
    p['titleIrrelevant1.5']=((count_title_irrelevant_1_5+1)/(count_rel_1_5+13))
    if model_dict_all[i][5]=='1.67' and model_dict_all[i][1]=='Irrelevant':
        count_title_irrelevant_1_67+=1
    p['titleIrrelevant1.67']=((count_title_irrelevant_1_67+1)/(count_rel_1_67+13))
    if model_dict_all[i][5]=='1.75' and model_dict_all[i][1]=='Irrelevant':
        count_title_irrelevant_1_75+=1
    p['titleIrrelevant1.75']=((count_title_irrelevant_1_75+1)/(count_rel_1_75+13))
    if model_dict_all[i][5]=='2.0' and model_dict_all[i][1]=='Irrelevant':
        count_title_irrelevant_2+=1
    p['titleIrrelevant2.0']=((count_title_irrelevant_2+1)/(count_rel_2+13))
    if model_dict_all[i][5]=='2.25' and model_dict_all[i][1]=='Irrelevant':
        count_titlel_irrelevant_2_25+=1
    p['titleIrrelevant2.25']=((count_titlel_irrelevant_2_25+1)/(count_rel_2_25+13))
    if model_dict_all[i][5]=='2.33' and model_dict_all[i][1]=='Irrelevant':
        count_title_irrelevant_2_33+=1
    p['titleIrrelevant2.33']=((count_title_irrelevant_2_33+1)/(count_rel_2_33+13))
    if model_dict_all[i][5]=='2.5' and model_dict_all[i][1]=='Irrelevant':
        count_title_irrelevant_2_5+=1
    p['titleIrrelevant2.5']=((count_title_irrelevant_2_5+1)/(count_rel_2_5+13))
    if model_dict_all[i][5]=='2.67' and model_dict_all[i][1]=='Irrelevant':
        count_title_irrelevant_2_67+=1
    p['titleIrrelevant2.67']=((count_title_irrelevant_2_67+1)/(count_rel_2_67+13))
    if model_dict_all[i][5]=='2.75' and model_dict_all[i][1]=='Irrelevant':
        count_title_irrelevant_2_75+=1
    p['titleIrrelevant2.75']=((count_title_irrelevant_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all[i][5]=='3.0' and model_dict_all[i][1]=='Irrelevant':
        count_title_irrelevant_3+=1
    p['titleIrrelevant3.0']=((count_title_irrelevant_3+1)/(count_rel_3+13))



    if model_dict_all[i][5]=='1.0' and model_dict_all[i][2]=='Perfect':
        count_desc_perfect_1+=1
    p['descPerfect1.0']=((count_desc_perfect_1+1)/(count_rel_1+13))
    if model_dict_all[i][5]=='1.25' and model_dict_all[i][2]=='Perfect':
        count_desc_perfect_1_25+=1
    p['descPerfect1.25']=((count_desc_perfect_1_25+1)/(count_rel_1_25+13))
    if model_dict_all[i][5]=='1.33' and model_dict_all[i][2]=='Perfect':
        count_desc_perfect_1_33+=1
    p['descPerfect1.33']=((count_desc_perfect_1_33+1)/(count_rel_1_33+13))
    if model_dict_all[i][5]=='1.5' and model_dict_all[i][2]=='Perfect':
        count_desc_perfect_1_5+=1
    p['descPerfect1.5']=((count_desc_perfect_1_5+1)/(count_rel_1_5+13))
    if model_dict_all[i][5]=='1.67' and model_dict_all[i][2]=='Perfect':
        count_desc_perfect_1_67+=1
    p['descPerfect1.67']=((count_desc_perfect_1_67+1)/(count_rel_1_67+13))
    if model_dict_all[i][5]=='1.75' and model_dict_all[i][2]=='Perfect':
        count_desc_perfect_1_75+=1
    p['descPerfect1.75']=((count_desc_perfect_1_75+1)/(count_rel_1_75+13))
    if model_dict_all[i][5]=='2.0' and model_dict_all[i][2]=='Perfect':
        count_desc_perfect_2+=1
    p['descPerfect2.0']=((count_desc_perfect_2+1)/(count_rel_2+13))
    if model_dict_all[i][5]=='2.25' and model_dict_all[i][2]=='Perfect':
        count_descl_perfect_2_25+=1
    p['descPerfect2.25']=((count_descl_perfect_2_25+1)/(count_rel_2_25+13))
    if model_dict_all[i][5]=='2.33' and model_dict_all[i][2]=='Perfect':
        count_desc_perfect_2_33+=1
    p['descPerfect2.33']=((count_desc_perfect_2_33+1)/(count_rel_2_33+13))
    if model_dict_all[i][5]=='2.5' and model_dict_all[i][2]=='Perfect':
        count_desc_perfect_2_5+=1
    p['descPerfect2.5']=((count_desc_perfect_2_5+1)/(count_rel_2_5+13))
    if model_dict_all[i][5]=='2.67' and model_dict_all[i][2]=='Perfect':
        count_desc_perfect_2_67+=1
    p['descPerfect2.67']=((count_desc_perfect_2_67+1)/(count_rel_2_67+13))
    if model_dict_all[i][5]=='2.75' and model_dict_all[i][2]=='Perfect':
        count_desc_perfect_2_75+=1
    p['descPerfect2.75']=((count_desc_perfect_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all[i][5]=='3.0' and model_dict_all[i][2]=='Perfect':
        count_desc_perfect_3+=1
    p['descPerfect3.0']=((count_desc_perfect_3+1)/(count_rel_3+13))


    if model_dict_all[i][5]=='1.0' and model_dict_all[i][2]=='Partial':
        count_desc_partial_1+=1
    p['descPartial1.0']=((count_desc_partial_1+1)/(count_rel_1+13))
    if model_dict_all[i][5]=='1.25' and model_dict_all[i][2]=='Partial':
        count_desc_partial_1_25+=1
    p['descPartial1.25']=((count_desc_partial_1_25+1)/(count_rel_1_25+13))
    if model_dict_all[i][5]=='1.33' and model_dict_all[i][2]=='Partial':
        count_desc_partial_1_33+=1
    p['descPartial1.33']=((count_desc_partial_1_33+1)/(count_rel_1_33+13))
    if model_dict_all[i][5]=='1.5' and model_dict_all[i][2]=='Partial':
        count_desc_partial_1_5+=1
    p['descPartial1.5']=((count_desc_partial_1_5+1)/(count_rel_1_5+13))
    if model_dict_all[i][5]=='1.67' and model_dict_all[i][2]=='Partial':
        count_desc_partial_1_67+=1
    p['descPartial1.67']=((count_desc_partial_1_67+1)/(count_rel_1_67+13))
    if model_dict_all[i][5]=='1.75' and model_dict_all[i][2]=='Partial':
        count_desc_partial_1_75+=1
    p['descPartial1.75']=((count_desc_partial_1_75+1)/(count_rel_1_75+13))
    if model_dict_all[i][5]=='2.0' and model_dict_all[i][2]=='Partial':
        count_desc_partial_2+=1
    p['descPartial2.0']=((count_desc_partial_2+1)/(count_rel_2+13))
    if model_dict_all[i][5]=='2.25' and model_dict_all[i][2]=='Partial':
        count_descl_partial_2_25+=1
    p['descPartial2.25']=((count_descl_partial_2_25+1)/(count_rel_2_25+13))
    if model_dict_all[i][5]=='2.33' and model_dict_all[i][2]=='Partial':
        count_desc_partial_2_33+=1
    p['descPartial2.33']=((count_desc_partial_2_33+1)/(count_rel_2_33+13))
    if model_dict_all[i][5]=='2.5' and model_dict_all[i][2]=='Partial':
        count_desc_partial_2_5+=1
    p['descPartial2.5']=((count_desc_partial_2_5+1)/(count_rel_2_5+13))
    if model_dict_all[i][5]=='2.67' and model_dict_all[i][2]=='Partial':
        count_desc_partial_2_67+=1
    p['descPartial2.67']=((count_desc_partial_2_67+1)/(count_rel_2_67+13))
    if model_dict_all[i][5]=='2.75' and model_dict_all[i][2]=='Partial':
        count_desc_partial_2_75+=1
    p['descPartial2.75']=((count_desc_partial_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all[i][5]=='3.0' and model_dict_all[i][2]=='Partial':
        count_desc_partial_3+=1
    p['descPartial3.0']=((count_desc_partial_3+1)/(count_rel_3+13))


    if model_dict_all[i][5]=='1.0' and model_dict_all[i][2]=='Irrelevant':
        count_desc_irrelevant_1+=1
    p['descIrrelevant1.0']=((count_desc_irrelevant_1+1)/(count_rel_1+13))
    if model_dict_all[i][5]=='1.25' and model_dict_all[i][2]=='Irrelevant':
        count_desc_irrelevant_1_25+=1
    p['descIrrelevant1.25']=((count_desc_irrelevant_1_25+1)/(count_rel_1_25+13))
    if model_dict_all[i][5]=='1.33' and model_dict_all[i][2]=='Irrelevant':
        count_desc_irrelevant_1_33+=1
    p['descIrrelevant1.33']=((count_desc_irrelevant_1_33+1)/(count_rel_1_33+13))
    if model_dict_all[i][5]=='1.5' and model_dict_all[i][2]=='Irrelevant':
        count_desc_irrelevant_1_5+=1
    p['descIrrelevant1.5']=((count_desc_irrelevant_1_5+1)/(count_rel_1_5+13))
    if model_dict_all[i][5]=='1.67' and model_dict_all[i][2]=='Irrelevant':
        count_desc_irrelevant_1_67+=1
    p['descIrrelevant1.67']=((count_desc_irrelevant_1_67+1)/(count_rel_1_67+13))
    if model_dict_all[i][5]=='1.75' and model_dict_all[i][2]=='Irrelevant':
        count_desc_irrelevant_1_75+=1
    p['descIrrelevant1.75']=((count_desc_irrelevant_1_75+1)/(count_rel_1_75+13))
    if model_dict_all[i][5]=='2.0' and model_dict_all[i][2]=='Irrelevant':
        count_desc_irrelevant_2+=1
    p['descIrrelevant2.0']=((count_desc_irrelevant_2+1)/(count_rel_2+13))
    if model_dict_all[i][5]=='2.25' and model_dict_all[i][2]=='Irrelevant':
        count_descl_irrelevant_2_25+=1
    p['descIrrelevant2.25']=((count_descl_irrelevant_2_25+1)/(count_rel_2_25+13))
    if model_dict_all[i][5]=='2.33' and model_dict_all[i][2]=='Irrelevant':
        count_desc_irrelevant_2_33+=1
    p['descIrrelevant2.33']=((count_desc_irrelevant_2_33+1)/(count_rel_2_33+13))
    if model_dict_all[i][5]=='2.5' and model_dict_all[i][2]=='Irrelevant':
        count_desc_irrelevant_2_5+=1
    p['descIrrelevant2.5']=((count_desc_irrelevant_2_5+1)/(count_rel_2_5+13))
    if model_dict_all[i][5]=='2.67' and model_dict_all[i][2]=='Irrelevant':
        count_desc_irrelevant_2_67+=1
    p['descIrrelevant2.67']=((count_desc_irrelevant_2_67+1)/(count_rel_2_67+13))
    if model_dict_all[i][5]=='2.75' and model_dict_all[i][2]=='Irrelevant':
        count_desc_irrelevant_2_75+=1
    p['descIrrelevant2.75']=((count_desc_irrelevant_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all[i][5]=='3.0' and model_dict_all[i][2]=='Irrelevant':
        count_desc_irrelevant_3+=1
    p['descIrrelevant3.0']=((count_desc_irrelevant_3+1)/(count_rel_3+13))


    if model_dict_all[i][5]=='1.0' and model_dict_all[i][3]=='Yes':
        count_brand_yes_1+=1
    p['brandYes1.0']=((count_brand_yes_1+1)/(count_rel_1+13))
    if model_dict_all[i][5]=='1.25' and model_dict_all[i][3]=='Yes':
        count_brand_yes_1_25+=1
    p['brandYes1.25']=((count_brand_yes_1_25+1)/(count_rel_1_25+13))
    if model_dict_all[i][5]=='1.33' and model_dict_all[i][3]=='Yes':
        count_brand_yes_1_33+=1
    p['brandYes1.33']=((count_brand_yes_1_33+1)/(count_rel_1_33+13))
    if model_dict_all[i][5]=='1.5' and model_dict_all[i][3]=='Yes':
        count_brand_yes_1_5+=1
    p['brandYes1.5']=((count_brand_yes_1_5+1)/(count_rel_1_5+13))
    if model_dict_all[i][5]=='1.67' and model_dict_all[i][3]=='Yes':
        count_brand_yes_1_67+=1
    p['brandYes1.67']=((count_brand_yes_1_67+1)/(count_rel_1_67+13))
    if model_dict_all[i][5]=='1.75' and model_dict_all[i][3]=='Yes':
        count_brand_yes_1_75+=1
    p['brandYes1.75']=((count_brand_yes_1_75+1)/(count_rel_1_75+13))
    if model_dict_all[i][5]=='2.0' and model_dict_all[i][3]=='Yes':
        count_brand_yes_2+=1
    p['brandYes2.0']=((count_brand_yes_2+1)/(count_rel_2+13))
    if model_dict_all[i][5]=='2.25' and model_dict_all[i][3]=='Yes':
        count_brandl_yes_2_25+=1
    p['brandYes2.25']=((count_brandl_yes_2_25+1)/(count_rel_2_25+13))
    if model_dict_all[i][5]=='2.33' and model_dict_all[i][3]=='Yes':
        count_brand_yes_2_33+=1
    p['brandYes2.33']=((count_brand_yes_2_33+1)/(count_rel_2_33+13))
    if model_dict_all[i][5]=='2.5' and model_dict_all[i][3]=='Yes':
        count_brand_yes_2_5+=1
    p['brandYes2.5']=((count_brand_yes_2_5+1)/(count_rel_2_5+13))
    if model_dict_all[i][5]=='2.67' and model_dict_all[i][3]=='Yes':
        count_brand_yes_2_67+=1
    p['brandYes2.67']=((count_brand_yes_2_67+1)/(count_rel_2_67+13))
    if model_dict_all[i][5]=='2.75' and model_dict_all[i][3]=='Yes':
        count_brand_yes_2_75+=1
    p['brandYes2.75']=((count_brand_yes_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all[i][5]=='3.0' and model_dict_all[i][3]=='Yes':
        count_brand_yes_3+=1
    p['brandYes3.0']=((count_brand_yes_3+1)/(count_rel_3+13))


    if model_dict_all[i][5]=='1.0' and model_dict_all[i][3]=='No':
        count_brand_no_1+=1
    p['brandNo1.0']=((count_brand_no_1+1)/(count_rel_1+13))
    if model_dict_all[i][5]=='1.25' and model_dict_all[i][3]=='No':
        count_brand_no_1_25+=1
    p['brandNo1.25']=((count_brand_no_1_25+1)/(count_rel_1_25+13))
    if model_dict_all[i][5]=='1.33' and model_dict_all[i][3]=='No':
        count_brand_no_1_33+=1
    p['brandNo1.33']=((count_brand_no_1_33+1)/(count_rel_1_33+13))
    if model_dict_all[i][5]=='1.5' and model_dict_all[i][3]=='No':
        count_brand_no_1_5+=1
    p['brandNo1.5']=((count_brand_no_1_5+1)/(count_rel_1_5+13))
    if model_dict_all[i][5]=='1.67' and model_dict_all[i][3]=='No':
        count_brand_no_1_67+=1
    p['brandNo1.67']=((count_brand_no_1_67+1)/(count_rel_1_67+13))
    if model_dict_all[i][5]=='1.75' and model_dict_all[i][3]=='No':
        count_brand_no_1_75+=1
    p['brandNo1.75']=((count_brand_no_1_75+1)/(count_rel_1_75+13))
    if model_dict_all[i][5]=='2.0' and model_dict_all[i][3]=='No':
        count_brand_no_2+=1
    p['brandNo2.0']=((count_brand_no_2+1)/(count_rel_2+13))
    if model_dict_all[i][5]=='2.25' and model_dict_all[i][3]=='No':
        count_brandl_no_2_25+=1
    p['brandNo2.25']=((count_brandl_no_2_25+1)/(count_rel_2_25+13))
    if model_dict_all[i][5]=='2.33' and model_dict_all[i][3]=='No':
        count_brand_no_2_33+=1
    p['brandNo2.33']=((count_brand_no_2_33+1)/(count_rel_2_33+13))
    if model_dict_all[i][5]=='2.5' and model_dict_all[i][3]=='No':
        count_brand_no_2_5+=1
    p['brandNo2.5']=((count_brand_no_2_5+1)/(count_rel_2_5+13))
    if model_dict_all[i][5]=='2.67' and model_dict_all[i][3]=='No':
        count_brand_no_2_67+=1
    p['brandNo2.67']=((count_brand_no_2_67+1)/(count_rel_2_67+13))
    if model_dict_all[i][5]=='2.75' and model_dict_all[i][3]=='No':
        count_brand_no_2_75+=1
    p['brandNo2.75']=((count_brand_no_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all[i][5]=='3.0' and model_dict_all[i][3]=='No':
        count_brand_no_3+=1
    p['brandNo3.0']=((count_brand_no_3+1)/(count_rel_3+13))


    if model_dict_all[i][5]=='1.0' and model_dict_all[i][3]=='Yes':
        count_material_yes_1+=1
    p['materialYes1.0']=((count_material_yes_1+1)/(count_rel_1+13))
    if model_dict_all[i][5]=='1.25' and model_dict_all[i][3]=='Yes':
        count_material_yes_1_25+=1
    p['materialYes1.25']=((count_material_yes_1_25+1)/(count_rel_1_25+13))
    if model_dict_all[i][5]=='1.33' and model_dict_all[i][3]=='Yes':
        count_material_yes_1_33+=1
    p['materialYes1.33']=((count_material_yes_1_33+1)/(count_rel_1_33+13))
    if model_dict_all[i][5]=='1.5' and model_dict_all[i][3]=='Yes':
        count_material_yes_1_5+=1
    p['materialYes1.5']=((count_material_yes_1_5+1)/(count_rel_1_5+13))
    if model_dict_all[i][5]=='1.67' and model_dict_all[i][3]=='Yes':
        count_material_yes_1_67+=1
    p['materialYes1.67']=((count_material_yes_1_67+1)/(count_rel_1_67+13))
    if model_dict_all[i][5]=='1.75' and model_dict_all[i][3]=='Yes':
        count_material_yes_1_75+=1
    p['materialYes1.75']=((count_material_yes_1_75+1)/(count_rel_1_75+13))
    if model_dict_all[i][5]=='2.0' and model_dict_all[i][3]=='Yes':
        count_material_yes_2+=1
    p['materialYes2.0']=((count_material_yes_2+1)/(count_rel_2+13))
    if model_dict_all[i][5]=='2.25' and model_dict_all[i][3]=='Yes':
        count_materiall_yes_2_25+=1
    p['materialYes2.25']=((count_materiall_yes_2_25+1)/(count_rel_2_25+13))
    if model_dict_all[i][5]=='2.33' and model_dict_all[i][3]=='Yes':
        count_material_yes_2_33+=1
    p['materialYes2.33']=((count_material_yes_2_33+1)/(count_rel_2_33+13))
    if model_dict_all[i][5]=='2.5' and model_dict_all[i][3]=='Yes':
        count_material_yes_2_5+=1
    p['materialYes2.5']=((count_material_yes_2_5+1)/(count_rel_2_5+13))
    if model_dict_all[i][5]=='2.67' and model_dict_all[i][3]=='Yes':
        count_material_yes_2_67+=1
    p['materialYes2.67']=((count_material_yes_2_67+1)/(count_rel_2_67+13))
    if model_dict_all[i][5]=='2.75' and model_dict_all[i][3]=='Yes':
        count_material_yes_2_75+=1
    p['materialYes2.75']=((count_material_yes_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all[i][5]=='3.0' and model_dict_all[i][3]=='Yes':
        count_material_yes_3+=1
    p['materialYes3.0']=((count_material_yes_3+1)/(count_rel_3+13))


    if model_dict_all[i][5]=='1.0' and model_dict_all[i][3]=='No':
        count_material_no_1+=1
    p['materialNo1.0']=((count_material_no_1+1)/(count_rel_1+13))
    if model_dict_all[i][5]=='1.25' and model_dict_all[i][3]=='No':
        count_material_no_1_25+=1
    p['materialNo1.25']=((count_material_no_1_25+1)/(count_rel_1_25+13))
    if model_dict_all[i][5]=='1.33' and model_dict_all[i][3]=='No':
        count_material_no_1_33+=1
    p['materialNo1.33']=((count_material_no_1_33+1)/(count_rel_1_33+13))
    if model_dict_all[i][5]=='1.5' and model_dict_all[i][3]=='No':
        count_material_no_1_5+=1
    p['materialNo1.5']=((count_material_no_1_5+1)/(count_rel_1_5+13))
    if model_dict_all[i][5]=='1.67' and model_dict_all[i][3]=='No':
        count_material_no_1_67+=1
    p['materialNo1.67']=((count_material_no_1_67+1)/(count_rel_1_67+13))
    if model_dict_all[i][5]=='1.75' and model_dict_all[i][3]=='No':
        count_material_no_1_75+=1
    p['materialNo1.75']=((count_material_no_1_75+1)/(count_rel_1_75+13))
    if model_dict_all[i][5]=='2.0' and model_dict_all[i][3]=='No':
        count_material_no_2+=1
    p['materialNo2.0']=((count_material_no_2+1)/(count_rel_2+13))
    if model_dict_all[i][5]=='2.25' and model_dict_all[i][3]=='No':
        count_materiall_no_2_25+=1
    p['materialNo2.25']=((count_materiall_no_2_25+1)/(count_rel_2_25+13))
    if model_dict_all[i][5]=='2.33' and model_dict_all[i][3]=='No':
        count_material_no_2_33+=1
    p['materialNo2.33']=((count_material_no_2_33+1)/(count_rel_2_33+13))
    if model_dict_all[i][5]=='2.5' and model_dict_all[i][3]=='No':
        count_material_no_2_5+=1
    p['materialNo2.5']=((count_material_no_2_5+1)/(count_rel_2_5+13))
    if model_dict_all[i][5]=='2.67' and model_dict_all[i][3]=='No':
        count_material_no_2_67+=1
    p['materialNo2.67']=((count_material_no_2_67+1)/(count_rel_2_67+13))
    if model_dict_all[i][5]=='2.75' and model_dict_all[i][3]=='No':
        count_material_no_2_75+=1
    p['materialNo2.75']=((count_material_no_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all[i][5]=='3.0' and model_dict_all[i][3]=='No':
        count_material_no_3+=1
    p['materialNo3.0']=((count_material_no_3+1)/(count_rel_3+13))


    if model_dict_all[i][5]=='1.0' and model_dict_all[i][3]=='Yes':
        count_function_yes_1+=1
    p['functionYes1.0']=((count_function_yes_1+1)/(count_rel_1+13))
    if model_dict_all[i][5]=='1.25' and model_dict_all[i][3]=='Yes':
        count_function_yes_1_25+=1
    p['functionYes1.25']=((count_function_yes_1_25+1)/(count_rel_1_25+13))
    if model_dict_all[i][5]=='1.33' and model_dict_all[i][3]=='Yes':
        count_function_yes_1_33+=1
    p['functionYes1.33']=((count_function_yes_1_33+1)/(count_rel_1_33+13))
    if model_dict_all[i][5]=='1.5' and model_dict_all[i][3]=='Yes':
        count_function_yes_1_5+=1
    p['functionYes1.5']=((count_function_yes_1_5+1)/(count_rel_1_5+13))
    if model_dict_all[i][5]=='1.67' and model_dict_all[i][3]=='Yes':
        count_function_yes_1_67+=1
    p['functionYes1.67']=((count_function_yes_1_67+1)/(count_rel_1_67+13))
    if model_dict_all[i][5]=='1.75' and model_dict_all[i][3]=='Yes':
        count_function_yes_1_75+=1
    p['functionYes1.75']=((count_function_yes_1_75+1)/(count_rel_1_75+13))
    if model_dict_all[i][5]=='2.0' and model_dict_all[i][3]=='Yes':
        count_function_yes_2+=1
    p['functionYes2.0']=((count_function_yes_2+1)/(count_rel_2+13))
    if model_dict_all[i][5]=='2.25' and model_dict_all[i][3]=='Yes':
        count_functionl_yes_2_25+=1
    p['functionYes2.25']=((count_functionl_yes_2_25+1)/(count_rel_2_25+13))
    if model_dict_all[i][5]=='2.33' and model_dict_all[i][3]=='Yes':
        count_function_yes_2_33+=1
    p['functionYes2.33']=((count_function_yes_2_33+1)/(count_rel_2_33+13))
    if model_dict_all[i][5]=='2.5' and model_dict_all[i][3]=='Yes':
        count_function_yes_2_5+=1
    p['functionYes2.5']=((count_function_yes_2_5+1)/(count_rel_2_5+13))
    if model_dict_all[i][5]=='2.67' and model_dict_all[i][3]=='Yes':
        count_function_yes_2_67+=1
    p['functionYes2.67']=((count_function_yes_2_67+1)/(count_rel_2_67+13))
    if model_dict_all[i][5]=='2.75' and model_dict_all[i][3]=='Yes':
        count_function_yes_2_75+=1
    p['functionYes2.75']=((count_function_yes_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all[i][5]=='3.0' and model_dict_all[i][3]=='Yes':
        count_function_yes_3+=1
    p['functionYes3.0']=((count_function_yes_3+1)/(count_rel_3+13))


    if model_dict_all[i][5]=='1.0' and model_dict_all[i][3]=='No':
        count_function_no_1+=1
    p['functionNo1.0']=((count_function_no_1+1)/(count_rel_1+13))
    if model_dict_all[i][5]=='1.25' and model_dict_all[i][3]=='No':
        count_function_no_1_25+=1
    p['functionNo1.25']=((count_function_no_1_25+1)/(count_rel_1_25+13))
    if model_dict_all[i][5]=='1.33' and model_dict_all[i][3]=='No':
        count_function_no_1_33+=1
    p['functionNo1.33']=((count_function_no_1_33+1)/(count_rel_1_33+13))
    if model_dict_all[i][5]=='1.5' and model_dict_all[i][3]=='No':
        count_function_no_1_5+=1
    p['functionNo1.5']=((count_function_no_1_5+1)/(count_rel_1_5+13))
    if model_dict_all[i][5]=='1.67' and model_dict_all[i][3]=='No':
        count_function_no_1_67+=1
    p['functionNo1.67']=((count_function_no_1_67+1)/(count_rel_1_67+13))
    if model_dict_all[i][5]=='1.75' and model_dict_all[i][3]=='No':
        count_function_no_1_75+=1
    p['functionNo1.75']=((count_function_no_1_75+1)/(count_rel_1_75+13))
    if model_dict_all[i][5]=='2.0' and model_dict_all[i][3]=='No':
        count_function_no_2+=1
    p['functionNo2.0']=((count_function_no_2+1)/(count_rel_2+13))
    if model_dict_all[i][5]=='2.25' and model_dict_all[i][3]=='No':
        count_functionl_no_2_25+=1
    p['functionNo2.25']=((count_functionl_no_2_25+1)/(count_rel_2_25+13))
    if model_dict_all[i][5]=='2.33' and model_dict_all[i][3]=='No':
        count_function_no_2_33+=1
    p['functionNo2.33']=((count_function_no_2_33+1)/(count_rel_2_33+13))
    if model_dict_all[i][5]=='2.5' and model_dict_all[i][3]=='No':
        count_function_no_2_5+=1
    p['functionNo2.5']=((count_function_no_2_5+1)/(count_rel_2_5+13))
    if model_dict_all[i][5]=='2.67' and model_dict_all[i][3]=='No':
        count_function_no_2_67+=1
    p['functionNo2.67']=((count_function_no_2_67+1)/(count_rel_2_67+13))
    if model_dict_all[i][5]=='2.75' and model_dict_all[i][3]=='No':
        count_function_no_2_75+=1
    p['functionNo2.75']=((count_function_no_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all[i][5]=='3.0' and model_dict_all[i][3]=='No':
        count_function_no_3+=1
    p['functionNo3.0']=((count_function_no_3+1)/(count_rel_3+13))



relevance_dict_test={}
id_uid_dict_test={}
title_dict_test={}
search_term_dict_test={}
initial_list_all_test=[]
initial_dictionary_all_test={}


# ID-RELEVANCE and ID-PRODUCT UID dictionary
test_csv.seek(0)
for i in test_df.itertuples():
    relevance_dict_test[i[1]]=str(i[5])
    id_uid_dict_test[i[1]]=i[2]


#PRODUCT ID TITLE DICT
test_csv.seek(0)
for i in test_df.itertuples():
    title_dict_test[i[2]]=i[3]


#SEARCH TERM DICT (ID-SEACRH TERM)
test_csv.seek(0)
for i in test_df.itertuples():
    search_term_dict_test[i[1]]=i[4]
print("ghgvcg")


for i in test_df._getitem_column('id'):
    initial_list_all_test=[tokenize_stem(search_term_dict_test[i]),tokenize_stem(str(title_dict_test[id_uid_dict_test[i]])),tokenize_stem(str(product_description_dict[id_uid_dict_test[i]])) ,tokenize_stem(str(brand_dict[id_uid_dict_test[i]])) if id_uid_dict_test[i] in brand_dict.keys() else tokenize_stem('None'),tokenize_stem(str(material_dict[id_uid_dict_test[i]])) if id_uid_dict_test[i] in material_dict.keys() else tokenize_stem('None'),tokenize_stem(str(function_dict[id_uid_dict_test[i]])) if id_uid_dict_test[i] in function_dict.keys() else tokenize_stem('None'),relevance_dict_test[i]]
    initial_dictionary_all_test[i]=initial_list_all_test


model_dict_all_test={}

for i in initial_dictionary_all_test:
    count_ptitle_test=0
    count_pdesc_test=0
    brand_flag_test=0
    material_flag_test=0
    function_flag_test=0
    counter_search_term_test = Counter(initial_dictionary_all_test[i][0])
    counter_ptitle_test=Counter(initial_dictionary_all_test[i][1])
    counter_pdesc_test=Counter(initial_dictionary_all_test[i][2])
    #print(counter_search_term_test)
    for word in initial_dictionary_all_test[i][0]:
        if word in counter_ptitle_test:
            count_ptitle_test+=counter_ptitle_test[word]
        if word in counter_pdesc_test:
            count_pdesc_test+=counter_pdesc_test[word]
        if word in initial_dictionary_all_test[i][3]:
            brand_flag_test=1
        if word in initial_dictionary_all_test[i][4]:
            material_flag_test=1
        if word in initial_dictionary_all_test[i][5]:
            function_flag_test=1
    if count_ptitle_test>=len(initial_dictionary_all_test[i][0]):
        model_dict_all_test[i]=['Perfect']
    elif count_ptitle_test<len(initial_dictionary_all_test[i][0]) and count_ptitle_test!=0:
        model_dict_all_test[i]=['Partial']
    else:
        model_dict_all_test[i]=['Irrelevant']
    if count_pdesc_test>=len(initial_dictionary_all_test[i][0]):
        model_dict_all_test[i].append('Perfect')
    elif count_pdesc_test<len(initial_dictionary_all_test[i][0]):
        model_dict_all_test[i].append('Partial')
    else:
        model_dict_all_test[i].append('Irrelevant')
    if brand_flag_test==1:
        model_dict_all_test[i].append('Yes')
    else:
        model_dict_all_test[i].append('No')
    if material_flag_test==1:
        model_dict_all_test[i].append('Yes')
    else:
        model_dict_all_test[i].append('No')
    if function_flag_test==1:
        model_dict_all_test[i].append('Yes')
    else:
        model_dict_all_test[i].append('No')
    model_dict_all_test[i].append(initial_dictionary_all_test[i][6])

#print(model_dict_all_test)


P={}
rel_before_list=[]
rel_after_list=[]
for i in model_dict_all_test:
    title=model_dict_all_test[i][0]
    desc=model_dict_all_test[i][1]
    brand=model_dict_all_test[i][2]
    material=model_dict_all_test[i][3]
    function=model_dict_all_test[i][4]
    relevance=model_dict_all_test[i][5]
    P['1.0']=p['title'+title+'1.0']*p['desc'+desc+'1.0']*p['brand'+brand+'1.0']*p['material'+material+'1.0']*p['function'+function+'1.0']
    P['1.25']=p['title'+title+'1.25']*p['desc'+desc+'1.25']*p['brand'+brand+'1.25']*p['material'+material+'1.25']*p['function'+function+'1.25']
    P['1.33']=p['title'+title+'1.33']*p['desc'+desc+'1.33']*p['brand'+brand+'1.33']*p['material'+material+'1.33']*p['function'+function+'1.33']
    P['1.5']=p['title'+title+'1.5']*p['desc'+desc+'1.5']*p['brand'+brand+'1.5']*p['material'+material+'1.5']*p['function'+function+'1.5']
    P['1.67']=p['title'+title+'1.67']*p['desc'+desc+'1.67']*p['brand'+brand+'1.67']*p['material'+material+'1.67']*p['function'+function+'1.67']
    P['1.75']=p['title'+title+'1.75']*p['desc'+desc+'1.75']*p['brand'+brand+'1.75']*p['material'+material+'1.75']*p['function'+function+'1.75']
    P['2.0']=p['title'+title+'2.0']*p['desc'+desc+'2.0']*p['brand'+brand+'2.0']*p['material'+material+'2.0']*p['function'+function+'2.0']
    P['2.25']=p['title'+title+'2.25']*p['desc'+desc+'2.25']*p['brand'+brand+'2.25']*p['material'+material+'2.25']*p['function'+function+'2.25']
    P['2.33']=p['title'+title+'2.33']*p['desc'+desc+'2.33']*p['brand'+brand+'2.33']*p['material'+material+'2.33']*p['function'+function+'2.33']
    P['2.5']=p['title'+title+'2.5']*p['desc'+desc+'2.5']*p['brand'+brand+'2.5']*p['material'+material+'2.5']*p['function'+function+'2.5']
    P['2.67']=p['title'+title+'2.67']*p['desc'+desc+'2.67']*p['brand'+brand+'2.67']*p['material'+material+'2.67']*p['function'+function+'2.67']
    P['3.0']=p['title'+title+'3.0']*p['desc'+desc+'3.0']*p['brand'+brand+'3.0']*p['material'+material+'3.0']*p['function'+function+'3.0']
    max_prob1=max(P['1.0'],P['1.25'],P['1.33'],P['1.5'],P['1.67'],P['1.75'],P['2.0'],P['2.25'],P['2.33'],P['2.5'],P['2.67'],P['3.0'])

    for j in P:
        if P[j]==max_prob1:
            after_rel=j

    rel_before_list.append(float(relevance))
    rel_after_list.append(float(after_rel))

rmse=mean_squared_error(rel_before_list, rel_after_list)**0.5
print('RMSE is:')
print(rmse)


relevance_dict_full_train={}
id_uid_dict_full_train={}
title_dict_full_train={}
search_term_dict_full_train={}

# ID-RELEVANCE and ID-PRODUCT UID dictionary
for i in train_full_df.itertuples():
    relevance_dict_full_train[i[1]]=str(i[5])
    id_uid_dict_full_train[i[1]]=i[2]

#PRODUCT ID TITLE DICT
for i in train_full_df.itertuples():
    title_dict_full_train[i[2]]=i[3]

#SEARCH TERM DICT (ID-SEACRH TERM)
for i in train_full_df.itertuples():
    search_term_dict_full_train[i[1]]=i[4]


initial_list_all_full_train=[]
initial_dictionary_all_full_train={}
for i in train_full_df._getitem_column('id'):
    initial_list_all_full_train=[tokenize_stem(search_term_dict_full_train[i]),tokenize_stem(str(title_dict_full_train[id_uid_dict_full_train[i]])),tokenize_stem(str(product_description_dict[id_uid_dict_full_train[i]])) ,tokenize_stem(str(brand_dict[id_uid_dict_full_train[i]])) if id_uid_dict_full_train[i] in brand_dict.keys() else tokenize_stem('None'),tokenize_stem(str(material_dict[id_uid_dict_full_train[i]])) if id_uid_dict_full_train[i] in material_dict.keys() else tokenize_stem('None'),tokenize_stem(str(function_dict[id_uid_dict_full_train[i]])) if id_uid_dict_full_train[i] in function_dict.keys() else tokenize_stem('None'),relevance_dict_full_train[i]]
    initial_dictionary_all_full_train[i]=initial_list_all_full_train

model_dict_all_full_train={}

#FORMING YES NO TABLE of FULL TRAIN.CSV
for i in initial_dictionary_all_full_train:
    count_ptitle=0
    count_pdesc=0
    brand_flag=0
    material_flag=0
    function_flag=0
    counter_search_term = Counter(initial_dictionary_all_full_train[i][0])
    counter_ptitle=Counter(initial_dictionary_all_full_train[i][1])
    counter_pdesc=Counter(initial_dictionary_all_full_train[i][2])
    #print(counter_search_term)
    for word in initial_dictionary_all_full_train[i][0]:
        if word in counter_ptitle:
            count_ptitle+=counter_ptitle[word]
        if word in counter_pdesc:
            count_pdesc+=counter_pdesc[word]
        if word in initial_dictionary_all_full_train[i][3]:
            brand_flag=1
        if word in initial_dictionary_all_full_train[i][4]:
            material_flag=1
        if word in initial_dictionary_all_full_train[i][5]:
            function_flag=1
    if count_ptitle>=len(initial_dictionary_all_full_train[i][0]):
        model_dict_all_full_train[i]=['Perfect']
    elif count_ptitle<len(initial_dictionary_all_full_train[i][0]) and count_ptitle!=0:
        model_dict_all_full_train[i]=['Partial']
    else:
        model_dict_all_full_train[i]=['Irrelevant']
    if count_pdesc>=len(initial_dictionary_all_full_train[i][0]):
        model_dict_all_full_train[i].append('Perfect')
    elif count_pdesc<len(initial_dictionary_all_full_train[i][0]):
        model_dict_all_full_train[i].append('Partial')
    else:
        model_dict_all_full_train[i].append('Irrelevant')
    if brand_flag==1:
        model_dict_all_full_train[i].append('Yes')
    else:
        model_dict_all_full_train[i].append('No')
    if material_flag==1:
        model_dict_all_full_train[i].append('Yes')
    else:
        model_dict_all_full_train[i].append('No')
    if function_flag==1:
        model_dict_all_full_train[i].append('Yes')
    else:
        model_dict_all_full_train[i].append('No')
    model_dict_all_full_train[i].append(initial_dictionary_all_full_train[i][6])

print("Train table formed")

#NAIVE BAYES PROB CALCULATION ON FULL TRAIN.CSV

count_rel_1=0
count_rel_1_25=0
count_rel_1_33=0
count_rel_1_5=0
count_rel_1_67=0
count_rel_1_75=0
count_rel_2=0
count_rel_2_25=0
count_rel_2_33=0
count_rel_2_5=0
count_rel_2_67=0
count_rel_2_75=0
count_rel_3=0

count_title_perfect=0
count_title_partial=0
count_title_irrelevant=0

count_title_perfect_1=0
count_title_perfect_1_25=0
count_title_perfect_1_33=0
count_title_perfect_1_5=0
count_title_perfect_1_67=0
count_title_perfect_1_75=0
count_title_perfect_2=0
count_titlel_perfect_2_25=0
count_title_perfect_2_33=0
count_title_perfect_2_5=0
count_title_perfect_2_67=0
count_title_perfect_2_75=0
count_title_perfect_3=0
count_title_partial_1=0
count_title_partial_1_25=0
count_title_partial_1_33=0
count_title_partial_1_5=0
count_title_partial_1_67=0
count_title_partial_1_75=0
count_title_partial_2=0
count_titlel_partial_2_25=0
count_title_partial_2_33=0
count_title_partial_2_5=0
count_title_partial_2_67=0
count_title_partial_2_75=0
count_title_partial_3=0
count_title_irrelevant_1=0
count_title_irrelevant_1_25=0
count_title_irrelevant_1_33=0
count_title_irrelevant_1_5=0
count_title_irrelevant_1_67=0
count_title_irrelevant_1_75=0
count_title_irrelevant_2=0
count_titlel_irrelevant_2_25=0
count_title_irrelevant_2_33=0
count_title_irrelevant_2_5=0
count_title_irrelevant_2_67=0
count_title_irrelevant_2_75=0
count_title_irrelevant_3=0

count_desc_perfect=0
count_desc_partial=0
count_desc_irrelevant=0

count_desc_perfect_1=0
count_desc_perfect_1_25=0
count_desc_perfect_1_33=0
count_desc_perfect_1_5=0
count_desc_perfect_1_67=0
count_desc_perfect_1_75=0
count_desc_perfect_2=0
count_descl_perfect_2_25=0
count_desc_perfect_2_33=0
count_desc_perfect_2_5=0
count_desc_perfect_2_67=0
count_desc_perfect_2_75=0
count_desc_perfect_3=0
count_desc_partial_1=0
count_desc_partial_1_25=0
count_desc_partial_1_33=0
count_desc_partial_1_5=0
count_desc_partial_1_67=0
count_desc_partial_1_75=0
count_desc_partial_2=0
count_descl_partial_2_25=0
count_desc_partial_2_33=0
count_desc_partial_2_5=0
count_desc_partial_2_67=0
count_desc_partial_2_75=0
count_desc_partial_3=0
count_desc_irrelevant_1=0
count_desc_irrelevant_1_25=0
count_desc_irrelevant_1_33=0
count_desc_irrelevant_1_5=0
count_desc_irrelevant_1_67=0
count_desc_irrelevant_1_75=0
count_desc_irrelevant_2=0
count_descl_irrelevant_2_25=0
count_desc_irrelevant_2_33=0
count_desc_irrelevant_2_5=0
count_desc_irrelevant_2_67=0
count_desc_irrelevant_2_75=0
count_desc_irrelevant_3=0

count_brand_yes_1=0
count_brand_yes_1_25=0
count_brand_yes_1_33=0
count_brand_yes_1_5=0
count_brand_yes_1_67=0
count_brand_yes_1_75=0
count_brand_yes_2=0
count_brandl_yes_2_25=0
count_brand_yes_2_33=0
count_brand_yes_2_5=0
count_brand_yes_2_67=0
count_brand_yes_2_75=0
count_brand_yes_3=0

count_brand_no_1=0
count_brand_no_1_25=0
count_brand_no_1_33=0
count_brand_no_1_5=0
count_brand_no_1_67=0
count_brand_no_1_75=0
count_brand_no_2=0
count_brandl_no_2_25=0
count_brand_no_2_33=0
count_brand_no_2_5=0
count_brand_no_2_67=0
count_brand_no_2_75=0
count_brand_no_3=0

count_material_yes_1=0
count_material_yes_1_25=0
count_material_yes_1_33=0
count_material_yes_1_5=0
count_material_yes_1_67=0
count_material_yes_1_75=0
count_material_yes_2=0
count_materiall_yes_2_25=0
count_material_yes_2_33=0
count_material_yes_2_5=0
count_material_yes_2_67=0
count_material_yes_2_75=0
count_material_yes_3=0

count_material_no_1=0
count_material_no_1_25=0
count_material_no_1_33=0
count_material_no_1_5=0
count_material_no_1_67=0
count_material_no_1_75=0
count_material_no_2=0
count_materiall_no_2_25=0
count_material_no_2_33=0
count_material_no_2_5=0
count_material_no_2_67=0
count_material_no_2_75=0
count_material_no_3=0

count_function_yes_1=0
count_function_yes_1_25=0
count_function_yes_1_33=0
count_function_yes_1_5=0
count_function_yes_1_67=0
count_function_yes_1_75=0
count_function_yes_2=0
count_functionl_yes_2_25=0
count_function_yes_2_33=0
count_function_yes_2_5=0
count_function_yes_2_67=0
count_function_yes_2_75=0
count_function_yes_3=0

count_function_no_1=0
count_function_no_1_25=0
count_function_no_1_33=0
count_function_no_1_5=0
count_function_no_1_67=0
count_function_no_1_75=0
count_function_no_2=0
count_functionl_no_2_25=0
count_function_no_2_33=0
count_function_no_2_5=0
count_function_no_2_67=0
count_function_no_2_75=0
count_function_no_3=0

print(count_function_no_3)

p1={}
N=len(model_dict_all_full_train)
for i in model_dict_all_full_train:
    if model_dict_all_full_train[i][5]=='1.0':
        count_rel_1+=1
    p1['1.0']=((count_rel_1+1)/(N+13))
    if model_dict_all_full_train[i][5]=='1.25':
        count_rel_1_25+=1
    p1['1.25']=((count_rel_1_25+1)/(N+13))
    if model_dict_all_full_train[i][5]=='1.33':
        count_rel_1_33+=1
    p1['1.33']=((count_rel_1_33+1)/(N+13))
    if model_dict_all_full_train[i][5]=='1.5':
        count_rel_1_5+=1
    p1['1.5']=((count_rel_1_5+1)/(N+13))
    if model_dict_all_full_train[i][5]=='1.67':
        count_rel_1_67+=1
    p1['1.67']=((count_rel_1_67+1)/(N+13))
    if model_dict_all_full_train[i][5]=='1.75':
        count_rel_1_75+=1
    p1['1.75']=((count_rel_1_75+1)/(N+13))
    if model_dict_all_full_train[i][5]=='2.0':
        count_rel_2+=1
    p1['2.0']=((count_rel_2+1)/(N+13))
    if model_dict_all_full_train[i][5]=='2.25':
        count_rel_2_25+=1
    p1['2.25']=((count_rel_2_25+1)/(N+13))
    if model_dict_all_full_train[i][5]=='2.33':
        count_rel_2_33+=1
    p1['2.33']=((count_rel_2_33+1)/(N+13))
    if model_dict_all_full_train[i][5]=='2.5':
        count_rel_2_5+=1
    p1['2.5']=((count_rel_2_5+1)/(N+13))
    if model_dict_all_full_train[i][5]=='2.67':
        count_rel_2_67+=1
    p1['2.67']=((count_rel_2_67+1)/(N+13))
    if model_dict_all_full_train[i][5]=='2.75':
        count_rel_2_75+=1
    p1['2.75']=((count_rel_2_75+1)/(N+13))
    if model_dict_all_full_train[i][5]=='3.0':
        count_rel_3+=1
    p1['3.0']=((count_rel_3+1)/(N+13))



    if model_dict_all_full_train[i][5]=='1.0' and model_dict_all_full_train[i][1]=='Perfect':
        count_title_perfect_1+=1
    p1['titlePerfect1.0']=((count_title_perfect_1+1)/(count_rel_1+13))
    if model_dict_all_full_train[i][5]=='1.25' and model_dict_all_full_train[i][1]=='Perfect':
        count_title_perfect_1_25+=1
    p1['titlePerfect1.25']=((count_title_perfect_1_25+1)/(count_rel_1_25+13))
    if model_dict_all_full_train[i][5]=='1.33' and model_dict_all_full_train[i][1]=='Perfect':
        count_title_perfect_1_33+=1
    p1['titlePerfect1.33']=((count_title_perfect_1_33+1)/(count_rel_1_33+13))
    if model_dict_all_full_train[i][5]=='1.5' and model_dict_all_full_train[i][1]=='Perfect':
        count_title_perfect_1_5+=1
    p1['titlePerfect1.5']=((count_title_perfect_1_5+1)/(count_rel_1_5+13))
    if model_dict_all_full_train[i][5]=='1.67' and model_dict_all_full_train[i][1]=='Perfect':
        count_title_perfect_1_67+=1
    p1['titlePerfect1.67']=((count_title_perfect_1_67+1)/(count_rel_1_67+13))
    if model_dict_all_full_train[i][5]=='1.75' and model_dict_all_full_train[i][1]=='Perfect':
        count_title_perfect_1_75+=1
    p1['titlePerfect1.75']=((count_title_perfect_1_75+1)/(count_rel_1_75+13))
    if model_dict_all_full_train[i][5]=='2.0' and model_dict_all_full_train[i][1]=='Perfect':
        count_title_perfect_2+=1
    p1['titlePerfect2.0']=((count_title_perfect_2+1)/(count_rel_2+13))
    if model_dict_all_full_train[i][5]=='2.25' and model_dict_all_full_train[i][1]=='Perfect':
        count_titlel_perfect_2_25+=1
    p1['titlePerfect2.25']=((count_titlel_perfect_2_25+1)/(count_rel_2_25+13))
    if model_dict_all_full_train[i][5]=='2.33' and model_dict_all_full_train[i][1]=='Perfect':
        count_title_perfect_2_33+=1
    p1['titlePerfect2.33']=((count_title_perfect_2_33+1)/(count_rel_2_33+13))
    if model_dict_all_full_train[i][5]=='2.5' and model_dict_all_full_train[i][1]=='Perfect':
        count_title_perfect_2_5+=1
    p1['titlePerfect2.5']=((count_title_perfect_2_5+1)/(count_rel_2_5+13))
    if model_dict_all_full_train[i][5]=='2.67' and model_dict_all_full_train[i][1]=='Perfect':
        count_title_perfect_2_67+=1
    p1['titlePerfect2.67']=((count_title_perfect_2_67+1)/(count_rel_2_67+13))
    if model_dict_all_full_train[i][5]=='2.75' and model_dict_all_full_train[i][1]=='Perfect':
        count_title_perfect_2_75+=1
    p1['titlePerfect2.75']=((count_title_perfect_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all_full_train[i][5]=='3.0' and model_dict_all_full_train[i][1]=='Perfect':
        count_title_perfect_3+=1
    p1['titlePerfect3.0']=((count_title_perfect_3+1)/(count_rel_3+13))


    if model_dict_all_full_train[i][5]=='1.0' and model_dict_all_full_train[i][1]=='Partial':
        count_title_partial_1+=1
    p1['titlePartial1.0']=((count_title_partial_1+1)/(count_rel_1+13))
    if model_dict_all_full_train[i][5]=='1.25' and model_dict_all_full_train[i][1]=='Partial':
        count_title_partial_1_25+=1
    p1['titlePartial1.25']=((count_title_partial_1_25+1)/(count_rel_1_25+13))
    if model_dict_all_full_train[i][5]=='1.33' and model_dict_all_full_train[i][1]=='Partial':
        count_title_partial_1_33+=1
    p1['titlePartial1.33']=((count_title_partial_1_33+1)/(count_rel_1_33+13))
    if model_dict_all_full_train[i][5]=='1.5' and model_dict_all_full_train[i][1]=='Partial':
        count_title_partial_1_5+=1
    p1['titlePartial1.5']=((count_title_partial_1_5+1)/(count_rel_1_5+13))
    if model_dict_all_full_train[i][5]=='1.67' and model_dict_all_full_train[i][1]=='Partial':
        count_title_partial_1_67+=1
    p1['titlePartial1.67']=((count_title_partial_1_67+1)/(count_rel_1_67+13))
    if model_dict_all_full_train[i][5]=='1.75' and model_dict_all_full_train[i][1]=='Partial':
        count_title_partial_1_75+=1
    p1['titlePartial1.75']=((count_title_partial_1_75+1)/(count_rel_1_75+13))
    if model_dict_all_full_train[i][5]=='2.0' and model_dict_all_full_train[i][1]=='Partial':
        count_title_partial_2+=1
    p1['titlePartial2.0']=((count_title_partial_2+1)/(count_rel_2+13))
    if model_dict_all_full_train[i][5]=='2.25' and model_dict_all_full_train[i][1]=='Partial':
        count_titlel_partial_2_25+=1
    p1['titlePartial2.25']=((count_titlel_partial_2_25+1)/(count_rel_2_25+13))
    if model_dict_all_full_train[i][5]=='2.33' and model_dict_all_full_train[i][1]=='Partial':
        count_title_partial_2_33+=1
    p1['titlePartial2.33']=((count_title_partial_2_33+1)/(count_rel_2_33+13))
    if model_dict_all_full_train[i][5]=='2.5' and model_dict_all_full_train[i][1]=='Partial':
        count_title_partial_2_5+=1
    p1['titlePartial2.5']=((count_title_partial_2_5+1)/(count_rel_2_5+13))
    if model_dict_all_full_train[i][5]=='2.67' and model_dict_all_full_train[i][1]=='Partial':
        count_title_partial_2_67+=1
    p1['titlePartial2.67']=((count_title_partial_2_67+1)/(count_rel_2_67+13))
    if model_dict_all_full_train[i][5]=='2.75' and model_dict_all_full_train[i][1]=='Partial':
        count_title_partial_2_75+=1
    p1['titlePartial2.75']=((count_title_partial_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all_full_train[i][5]=='3.0' and model_dict_all_full_train[i][1]=='Partial':
        count_title_partial_3+=1
    p1['titlePartial3.0']=((count_title_partial_3+1)/(count_rel_3+13))


    if model_dict_all_full_train[i][5]=='1.0' and model_dict_all_full_train[i][1]=='Irrelevant':
        count_title_irrelevant_1+=1
    p1['titleIrrelevant1.0']=((count_title_irrelevant_1+1)/(count_rel_1+13))
    if model_dict_all_full_train[i][5]=='1.25' and model_dict_all_full_train[i][1]=='Irrelevant':
        count_title_irrelevant_1_25+=1
    p1['titleIrrelevant1.25']=((count_title_irrelevant_1_25+1)/(count_rel_1_25+13))
    if model_dict_all_full_train[i][5]=='1.33' and model_dict_all_full_train[i][1]=='Irrelevant':
        count_title_irrelevant_1_33+=1
    p1['titleIrrelevant1.33']=((count_title_irrelevant_1_33+1)/(count_rel_1_33+13))
    if model_dict_all_full_train[i][5]=='1.5' and model_dict_all_full_train[i][1]=='Irrelevant':
        count_title_irrelevant_1_5+=1
    p1['titleIrrelevant1.5']=((count_title_irrelevant_1_5+1)/(count_rel_1_5+13))
    if model_dict_all_full_train[i][5]=='1.67' and model_dict_all_full_train[i][1]=='Irrelevant':
        count_title_irrelevant_1_67+=1
    p1['titleIrrelevant1.67']=((count_title_irrelevant_1_67+1)/(count_rel_1_67+13))
    if model_dict_all_full_train[i][5]=='1.75' and model_dict_all_full_train[i][1]=='Irrelevant':
        count_title_irrelevant_1_75+=1
    p1['titleIrrelevant1.75']=((count_title_irrelevant_1_75+1)/(count_rel_1_75+13))
    if model_dict_all_full_train[i][5]=='2.0' and model_dict_all_full_train[i][1]=='Irrelevant':
        count_title_irrelevant_2+=1
    p1['titleIrrelevant2.0']=((count_title_irrelevant_2+1)/(count_rel_2+13))
    if model_dict_all_full_train[i][5]=='2.25' and model_dict_all_full_train[i][1]=='Irrelevant':
        count_titlel_irrelevant_2_25+=1
    p1['titleIrrelevant2.25']=((count_titlel_irrelevant_2_25+1)/(count_rel_2_25+13))
    if model_dict_all_full_train[i][5]=='2.33' and model_dict_all_full_train[i][1]=='Irrelevant':
        count_title_irrelevant_2_33+=1
    p1['titleIrrelevant2.33']=((count_title_irrelevant_2_33+1)/(count_rel_2_33+13))
    if model_dict_all_full_train[i][5]=='2.5' and model_dict_all_full_train[i][1]=='Irrelevant':
        count_title_irrelevant_2_5+=1
    p1['titleIrrelevant2.5']=((count_title_irrelevant_2_5+1)/(count_rel_2_5+13))
    if model_dict_all_full_train[i][5]=='2.67' and model_dict_all_full_train[i][1]=='Irrelevant':
        count_title_irrelevant_2_67+=1
    p1['titleIrrelevant2.67']=((count_title_irrelevant_2_67+1)/(count_rel_2_67+13))
    if model_dict_all_full_train[i][5]=='2.75' and model_dict_all_full_train[i][1]=='Irrelevant':
        count_title_irrelevant_2_75+=1
    p1['titleIrrelevant2.75']=((count_title_irrelevant_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all_full_train[i][5]=='3.0' and model_dict_all_full_train[i][1]=='Irrelevant':
        count_title_irrelevant_3+=1
    p1['titleIrrelevant3.0']=((count_title_irrelevant_3+1)/(count_rel_3+13))



    if model_dict_all_full_train[i][5]=='1.0' and model_dict_all_full_train[i][2]=='Perfect':
        count_desc_perfect_1+=1
    p1['descPerfect1.0']=((count_desc_perfect_1+1)/(count_rel_1+13))
    if model_dict_all_full_train[i][5]=='1.25' and model_dict_all_full_train[i][2]=='Perfect':
        count_desc_perfect_1_25+=1
    p1['descPerfect1.25']=((count_desc_perfect_1_25+1)/(count_rel_1_25+13))
    if model_dict_all_full_train[i][5]=='1.33' and model_dict_all_full_train[i][2]=='Perfect':
        count_desc_perfect_1_33+=1
    p1['descPerfect1.33']=((count_desc_perfect_1_33+1)/(count_rel_1_33+13))
    if model_dict_all_full_train[i][5]=='1.5' and model_dict_all_full_train[i][2]=='Perfect':
        count_desc_perfect_1_5+=1
    p1['descPerfect1.5']=((count_desc_perfect_1_5+1)/(count_rel_1_5+13))
    if model_dict_all_full_train[i][5]=='1.67' and model_dict_all_full_train[i][2]=='Perfect':
        count_desc_perfect_1_67+=1
    p1['descPerfect1.67']=((count_desc_perfect_1_67+1)/(count_rel_1_67+13))
    if model_dict_all_full_train[i][5]=='1.75' and model_dict_all_full_train[i][2]=='Perfect':
        count_desc_perfect_1_75+=1
    p1['descPerfect1.75']=((count_desc_perfect_1_75+1)/(count_rel_1_75+13))
    if model_dict_all_full_train[i][5]=='2.0' and model_dict_all_full_train[i][2]=='Perfect':
        count_desc_perfect_2+=1
    p1['descPerfect2.0']=((count_desc_perfect_2+1)/(count_rel_2+13))
    if model_dict_all_full_train[i][5]=='2.25' and model_dict_all_full_train[i][2]=='Perfect':
        count_descl_perfect_2_25+=1
    p1['descPerfect2.25']=((count_descl_perfect_2_25+1)/(count_rel_2_25+13))
    if model_dict_all_full_train[i][5]=='2.33' and model_dict_all_full_train[i][2]=='Perfect':
        count_desc_perfect_2_33+=1
    p1['descPerfect2.33']=((count_desc_perfect_2_33+1)/(count_rel_2_33+13))
    if model_dict_all_full_train[i][5]=='2.5' and model_dict_all_full_train[i][2]=='Perfect':
        count_desc_perfect_2_5+=1
    p1['descPerfect2.5']=((count_desc_perfect_2_5+1)/(count_rel_2_5+13))
    if model_dict_all_full_train[i][5]=='2.67' and model_dict_all_full_train[i][2]=='Perfect':
        count_desc_perfect_2_67+=1
    p1['descPerfect2.67']=((count_desc_perfect_2_67+1)/(count_rel_2_67+13))
    if model_dict_all_full_train[i][5]=='2.75' and model_dict_all_full_train[i][2]=='Perfect':
        count_desc_perfect_2_75+=1
    p1['descPerfect2.75']=((count_desc_perfect_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all_full_train[i][5]=='3.0' and model_dict_all_full_train[i][2]=='Perfect':
        count_desc_perfect_3+=1
    p1['descPerfect3.0']=((count_desc_perfect_3+1)/(count_rel_3+13))


    if model_dict_all_full_train[i][5]=='1.0' and model_dict_all_full_train[i][2]=='Partial':
        count_desc_partial_1+=1
    p1['descPartial1.0']=((count_desc_partial_1+1)/(count_rel_1+13))
    if model_dict_all_full_train[i][5]=='1.25' and model_dict_all_full_train[i][2]=='Partial':
        count_desc_partial_1_25+=1
    p1['descPartial1.25']=((count_desc_partial_1_25+1)/(count_rel_1_25+13))
    if model_dict_all_full_train[i][5]=='1.33' and model_dict_all_full_train[i][2]=='Partial':
        count_desc_partial_1_33+=1
    p1['descPartial1.33']=((count_desc_partial_1_33+1)/(count_rel_1_33+13))
    if model_dict_all_full_train[i][5]=='1.5' and model_dict_all_full_train[i][2]=='Partial':
        count_desc_partial_1_5+=1
    p1['descPartial1.5']=((count_desc_partial_1_5+1)/(count_rel_1_5+13))
    if model_dict_all_full_train[i][5]=='1.67' and model_dict_all_full_train[i][2]=='Partial':
        count_desc_partial_1_67+=1
    p1['descPartial1.67']=((count_desc_partial_1_67+1)/(count_rel_1_67+13))
    if model_dict_all_full_train[i][5]=='1.75' and model_dict_all_full_train[i][2]=='Partial':
        count_desc_partial_1_75+=1
    p1['descPartial1.75']=((count_desc_partial_1_75+1)/(count_rel_1_75+13))
    if model_dict_all_full_train[i][5]=='2.0' and model_dict_all_full_train[i][2]=='Partial':
        count_desc_partial_2+=1
    p1['descPartial2.0']=((count_desc_partial_2+1)/(count_rel_2+13))
    if model_dict_all_full_train[i][5]=='2.25' and model_dict_all_full_train[i][2]=='Partial':
        count_descl_partial_2_25+=1
    p1['descPartial2.25']=((count_descl_partial_2_25+1)/(count_rel_2_25+13))
    if model_dict_all_full_train[i][5]=='2.33' and model_dict_all_full_train[i][2]=='Partial':
        count_desc_partial_2_33+=1
    p1['descPartial2.33']=((count_desc_partial_2_33+1)/(count_rel_2_33+13))
    if model_dict_all_full_train[i][5]=='2.5' and model_dict_all_full_train[i][2]=='Partial':
        count_desc_partial_2_5+=1
    p1['descPartial2.5']=((count_desc_partial_2_5+1)/(count_rel_2_5+13))
    if model_dict_all_full_train[i][5]=='2.67' and model_dict_all_full_train[i][2]=='Partial':
        count_desc_partial_2_67+=1
    p1['descPartial2.67']=((count_desc_partial_2_67+1)/(count_rel_2_67+13))
    if model_dict_all_full_train[i][5]=='2.75' and model_dict_all_full_train[i][2]=='Partial':
        count_desc_partial_2_75+=1
    p1['descPartial2.75']=((count_desc_partial_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all_full_train[i][5]=='3.0' and model_dict_all_full_train[i][2]=='Partial':
        count_desc_partial_3+=1
    p1['descPartial3.0']=((count_desc_partial_3+1)/(count_rel_3+13))


    if model_dict_all_full_train[i][5]=='1.0' and model_dict_all_full_train[i][2]=='Irrelevant':
        count_desc_irrelevant_1+=1
    p1['descIrrelevant1.0']=((count_desc_irrelevant_1+1)/(count_rel_1+13))
    if model_dict_all_full_train[i][5]=='1.25' and model_dict_all_full_train[i][2]=='Irrelevant':
        count_desc_irrelevant_1_25+=1
    p1['descIrrelevant1.25']=((count_desc_irrelevant_1_25+1)/(count_rel_1_25+13))
    if model_dict_all_full_train[i][5]=='1.33' and model_dict_all_full_train[i][2]=='Irrelevant':
        count_desc_irrelevant_1_33+=1
    p1['descIrrelevant1.33']=((count_desc_irrelevant_1_33+1)/(count_rel_1_33+13))
    if model_dict_all_full_train[i][5]=='1.5' and model_dict_all_full_train[i][2]=='Irrelevant':
        count_desc_irrelevant_1_5+=1
    p1['descIrrelevant1.5']=((count_desc_irrelevant_1_5+1)/(count_rel_1_5+13))
    if model_dict_all_full_train[i][5]=='1.67' and model_dict_all_full_train[i][2]=='Irrelevant':
        count_desc_irrelevant_1_67+=1
    p1['descIrrelevant1.67']=((count_desc_irrelevant_1_67+1)/(count_rel_1_67+13))
    if model_dict_all_full_train[i][5]=='1.75' and model_dict_all_full_train[i][2]=='Irrelevant':
        count_desc_irrelevant_1_75+=1
    p1['descIrrelevant1.75']=((count_desc_irrelevant_1_75+1)/(count_rel_1_75+13))
    if model_dict_all_full_train[i][5]=='2.0' and model_dict_all_full_train[i][2]=='Irrelevant':
        count_desc_irrelevant_2+=1
    p1['descIrrelevant2.0']=((count_desc_irrelevant_2+1)/(count_rel_2+13))
    if model_dict_all_full_train[i][5]=='2.25' and model_dict_all_full_train[i][2]=='Irrelevant':
        count_descl_irrelevant_2_25+=1
    p1['descIrrelevant2.25']=((count_descl_irrelevant_2_25+1)/(count_rel_2_25+13))
    if model_dict_all_full_train[i][5]=='2.33' and model_dict_all_full_train[i][2]=='Irrelevant':
        count_desc_irrelevant_2_33+=1
    p1['descIrrelevant2.33']=((count_desc_irrelevant_2_33+1)/(count_rel_2_33+13))
    if model_dict_all_full_train[i][5]=='2.5' and model_dict_all_full_train[i][2]=='Irrelevant':
        count_desc_irrelevant_2_5+=1
    p1['descIrrelevant2.5']=((count_desc_irrelevant_2_5+1)/(count_rel_2_5+13))
    if model_dict_all_full_train[i][5]=='2.67' and model_dict_all_full_train[i][2]=='Irrelevant':
        count_desc_irrelevant_2_67+=1
    p1['descIrrelevant2.67']=((count_desc_irrelevant_2_67+1)/(count_rel_2_67+13))
    if model_dict_all_full_train[i][5]=='2.75' and model_dict_all_full_train[i][2]=='Irrelevant':
        count_desc_irrelevant_2_75+=1
    p1['descIrrelevant2.75']=((count_desc_irrelevant_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all_full_train[i][5]=='3.0' and model_dict_all_full_train[i][2]=='Irrelevant':
        count_desc_irrelevant_3+=1
    p1['descIrrelevant3.0']=((count_desc_irrelevant_3+1)/(count_rel_3+13))


    if model_dict_all_full_train[i][5]=='1.0' and model_dict_all_full_train[i][3]=='Yes':
        count_brand_yes_1+=1
    p1['brandYes1.0']=((count_brand_yes_1+1)/(count_rel_1+13))
    if model_dict_all_full_train[i][5]=='1.25' and model_dict_all_full_train[i][3]=='Yes':
        count_brand_yes_1_25+=1
    p1['brandYes1.25']=((count_brand_yes_1_25+1)/(count_rel_1_25+13))
    if model_dict_all_full_train[i][5]=='1.33' and model_dict_all_full_train[i][3]=='Yes':
        count_brand_yes_1_33+=1
    p1['brandYes1.33']=((count_brand_yes_1_33+1)/(count_rel_1_33+13))
    if model_dict_all_full_train[i][5]=='1.5' and model_dict_all_full_train[i][3]=='Yes':
        count_brand_yes_1_5+=1
    p1['brandYes1.5']=((count_brand_yes_1_5+1)/(count_rel_1_5+13))
    if model_dict_all_full_train[i][5]=='1.67' and model_dict_all_full_train[i][3]=='Yes':
        count_brand_yes_1_67+=1
    p1['brandYes1.67']=((count_brand_yes_1_67+1)/(count_rel_1_67+13))
    if model_dict_all_full_train[i][5]=='1.75' and model_dict_all_full_train[i][3]=='Yes':
        count_brand_yes_1_75+=1
    p1['brandYes1.75']=((count_brand_yes_1_75+1)/(count_rel_1_75+13))
    if model_dict_all_full_train[i][5]=='2.0' and model_dict_all_full_train[i][3]=='Yes':
        count_brand_yes_2+=1
    p1['brandYes2.0']=((count_brand_yes_2+1)/(count_rel_2+13))
    if model_dict_all_full_train[i][5]=='2.25' and model_dict_all_full_train[i][3]=='Yes':
        count_brandl_yes_2_25+=1
    p1['brandYes2.25']=((count_brandl_yes_2_25+1)/(count_rel_2_25+13))
    if model_dict_all_full_train[i][5]=='2.33' and model_dict_all_full_train[i][3]=='Yes':
        count_brand_yes_2_33+=1
    p1['brandYes2.33']=((count_brand_yes_2_33+1)/(count_rel_2_33+13))
    if model_dict_all_full_train[i][5]=='2.5' and model_dict_all_full_train[i][3]=='Yes':
        count_brand_yes_2_5+=1
    p1['brandYes2.5']=((count_brand_yes_2_5+1)/(count_rel_2_5+13))
    if model_dict_all_full_train[i][5]=='2.67' and model_dict_all_full_train[i][3]=='Yes':
        count_brand_yes_2_67+=1
    p1['brandYes2.67']=((count_brand_yes_2_67+1)/(count_rel_2_67+13))
    if model_dict_all_full_train[i][5]=='2.75' and model_dict_all_full_train[i][3]=='Yes':
        count_brand_yes_2_75+=1
    p1['brandYes2.75']=((count_brand_yes_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all_full_train[i][5]=='3.0' and model_dict_all_full_train[i][3]=='Yes':
        count_brand_yes_3+=1
    p1['brandYes3.0']=((count_brand_yes_3+1)/(count_rel_3+13))


    if model_dict_all_full_train[i][5]=='1.0' and model_dict_all_full_train[i][3]=='No':
        count_brand_no_1+=1
    p1['brandNo1.0']=((count_brand_no_1+1)/(count_rel_1+13))
    if model_dict_all_full_train[i][5]=='1.25' and model_dict_all_full_train[i][3]=='No':
        count_brand_no_1_25+=1
    p1['brandNo1.25']=((count_brand_no_1_25+1)/(count_rel_1_25+13))
    if model_dict_all_full_train[i][5]=='1.33' and model_dict_all_full_train[i][3]=='No':
        count_brand_no_1_33+=1
    p1['brandNo1.33']=((count_brand_no_1_33+1)/(count_rel_1_33+13))
    if model_dict_all_full_train[i][5]=='1.5' and model_dict_all_full_train[i][3]=='No':
        count_brand_no_1_5+=1
    p1['brandNo1.5']=((count_brand_no_1_5+1)/(count_rel_1_5+13))
    if model_dict_all_full_train[i][5]=='1.67' and model_dict_all_full_train[i][3]=='No':
        count_brand_no_1_67+=1
    p1['brandNo1.67']=((count_brand_no_1_67+1)/(count_rel_1_67+13))
    if model_dict_all_full_train[i][5]=='1.75' and model_dict_all_full_train[i][3]=='No':
        count_brand_no_1_75+=1
    p1['brandNo1.75']=((count_brand_no_1_75+1)/(count_rel_1_75+13))
    if model_dict_all_full_train[i][5]=='2.0' and model_dict_all_full_train[i][3]=='No':
        count_brand_no_2+=1
    p1['brandNo2.0']=((count_brand_no_2+1)/(count_rel_2+13))
    if model_dict_all_full_train[i][5]=='2.25' and model_dict_all_full_train[i][3]=='No':
        count_brandl_no_2_25+=1
    p1['brandNo2.25']=((count_brandl_no_2_25+1)/(count_rel_2_25+13))
    if model_dict_all_full_train[i][5]=='2.33' and model_dict_all_full_train[i][3]=='No':
        count_brand_no_2_33+=1
    p1['brandNo2.33']=((count_brand_no_2_33+1)/(count_rel_2_33+13))
    if model_dict_all_full_train[i][5]=='2.5' and model_dict_all_full_train[i][3]=='No':
        count_brand_no_2_5+=1
    p1['brandNo2.5']=((count_brand_no_2_5+1)/(count_rel_2_5+13))
    if model_dict_all_full_train[i][5]=='2.67' and model_dict_all_full_train[i][3]=='No':
        count_brand_no_2_67+=1
    p1['brandNo2.67']=((count_brand_no_2_67+1)/(count_rel_2_67+13))
    if model_dict_all_full_train[i][5]=='2.75' and model_dict_all_full_train[i][3]=='No':
        count_brand_no_2_75+=1
    p1['brandNo2.75']=((count_brand_no_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all_full_train[i][5]=='3.0' and model_dict_all_full_train[i][3]=='No':
        count_brand_no_3+=1
    p1['brandNo3.0']=((count_brand_no_3+1)/(count_rel_3+13))


    if model_dict_all_full_train[i][5]=='1.0' and model_dict_all_full_train[i][3]=='Yes':
        count_material_yes_1+=1
    p1['materialYes1.0']=((count_material_yes_1+1)/(count_rel_1+13))
    if model_dict_all_full_train[i][5]=='1.25' and model_dict_all_full_train[i][3]=='Yes':
        count_material_yes_1_25+=1
    p1['materialYes1.25']=((count_material_yes_1_25+1)/(count_rel_1_25+13))
    if model_dict_all_full_train[i][5]=='1.33' and model_dict_all_full_train[i][3]=='Yes':
        count_material_yes_1_33+=1
    p1['materialYes1.33']=((count_material_yes_1_33+1)/(count_rel_1_33+13))
    if model_dict_all_full_train[i][5]=='1.5' and model_dict_all_full_train[i][3]=='Yes':
        count_material_yes_1_5+=1
    p1['materialYes1.5']=((count_material_yes_1_5+1)/(count_rel_1_5+13))
    if model_dict_all_full_train[i][5]=='1.67' and model_dict_all_full_train[i][3]=='Yes':
        count_material_yes_1_67+=1
    p1['materialYes1.67']=((count_material_yes_1_67+1)/(count_rel_1_67+13))
    if model_dict_all_full_train[i][5]=='1.75' and model_dict_all_full_train[i][3]=='Yes':
        count_material_yes_1_75+=1
    p1['materialYes1.75']=((count_material_yes_1_75+1)/(count_rel_1_75+13))
    if model_dict_all_full_train[i][5]=='2.0' and model_dict_all_full_train[i][3]=='Yes':
        count_material_yes_2+=1
    p1['materialYes2.0']=((count_material_yes_2+1)/(count_rel_2+13))
    if model_dict_all_full_train[i][5]=='2.25' and model_dict_all_full_train[i][3]=='Yes':
        count_materiall_yes_2_25+=1
    p1['materialYes2.25']=((count_materiall_yes_2_25+1)/(count_rel_2_25+13))
    if model_dict_all_full_train[i][5]=='2.33' and model_dict_all_full_train[i][3]=='Yes':
        count_material_yes_2_33+=1
    p1['materialYes2.33']=((count_material_yes_2_33+1)/(count_rel_2_33+13))
    if model_dict_all_full_train[i][5]=='2.5' and model_dict_all_full_train[i][3]=='Yes':
        count_material_yes_2_5+=1
    p1['materialYes2.5']=((count_material_yes_2_5+1)/(count_rel_2_5+13))
    if model_dict_all_full_train[i][5]=='2.67' and model_dict_all_full_train[i][3]=='Yes':
        count_material_yes_2_67+=1
    p1['materialYes2.67']=((count_material_yes_2_67+1)/(count_rel_2_67+13))
    if model_dict_all_full_train[i][5]=='2.75' and model_dict_all_full_train[i][3]=='Yes':
        count_material_yes_2_75+=1
    p1['materialYes2.75']=((count_material_yes_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all_full_train[i][5]=='3.0' and model_dict_all_full_train[i][3]=='Yes':
        count_material_yes_3+=1
    p1['materialYes3.0']=((count_material_yes_3+1)/(count_rel_3+13))


    if model_dict_all_full_train[i][5]=='1.0' and model_dict_all_full_train[i][3]=='No':
        count_material_no_1+=1
    p1['materialNo1.0']=((count_material_no_1+1)/(count_rel_1+13))
    if model_dict_all_full_train[i][5]=='1.25' and model_dict_all_full_train[i][3]=='No':
        count_material_no_1_25+=1
    p1['materialNo1.25']=((count_material_no_1_25+1)/(count_rel_1_25+13))
    if model_dict_all_full_train[i][5]=='1.33' and model_dict_all_full_train[i][3]=='No':
        count_material_no_1_33+=1
    p1['materialNo1.33']=((count_material_no_1_33+1)/(count_rel_1_33+13))
    if model_dict_all_full_train[i][5]=='1.5' and model_dict_all_full_train[i][3]=='No':
        count_material_no_1_5+=1
    p1['materialNo1.5']=((count_material_no_1_5+1)/(count_rel_1_5+13))
    if model_dict_all_full_train[i][5]=='1.67' and model_dict_all_full_train[i][3]=='No':
        count_material_no_1_67+=1
    p1['materialNo1.67']=((count_material_no_1_67+1)/(count_rel_1_67+13))
    if model_dict_all_full_train[i][5]=='1.75' and model_dict_all_full_train[i][3]=='No':
        count_material_no_1_75+=1
    p1['materialNo1.75']=((count_material_no_1_75+1)/(count_rel_1_75+13))
    if model_dict_all_full_train[i][5]=='2.0' and model_dict_all_full_train[i][3]=='No':
        count_material_no_2+=1
    p1['materialNo2.0']=((count_material_no_2+1)/(count_rel_2+13))
    if model_dict_all_full_train[i][5]=='2.25' and model_dict_all_full_train[i][3]=='No':
        count_materiall_no_2_25+=1
    p1['materialNo2.25']=((count_materiall_no_2_25+1)/(count_rel_2_25+13))
    if model_dict_all_full_train[i][5]=='2.33' and model_dict_all_full_train[i][3]=='No':
        count_material_no_2_33+=1
    p1['materialNo2.33']=((count_material_no_2_33+1)/(count_rel_2_33+13))
    if model_dict_all_full_train[i][5]=='2.5' and model_dict_all_full_train[i][3]=='No':
        count_material_no_2_5+=1
    p1['materialNo2.5']=((count_material_no_2_5+1)/(count_rel_2_5+13))
    if model_dict_all_full_train[i][5]=='2.67' and model_dict_all_full_train[i][3]=='No':
        count_material_no_2_67+=1
    p1['materialNo2.67']=((count_material_no_2_67+1)/(count_rel_2_67+13))
    if model_dict_all_full_train[i][5]=='2.75' and model_dict_all_full_train[i][3]=='No':
        count_material_no_2_75+=1
    p1['materialNo2.75']=((count_material_no_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all_full_train[i][5]=='3.0' and model_dict_all_full_train[i][3]=='No':
        count_material_no_3+=1
    p1['materialNo3.0']=((count_material_no_3+1)/(count_rel_3+13))


    if model_dict_all_full_train[i][5]=='1.0' and model_dict_all_full_train[i][3]=='Yes':
        count_function_yes_1+=1
    p1['functionYes1.0']=((count_function_yes_1+1)/(count_rel_1+13))
    if model_dict_all_full_train[i][5]=='1.25' and model_dict_all_full_train[i][3]=='Yes':
        count_function_yes_1_25+=1
    p1['functionYes1.25']=((count_function_yes_1_25+1)/(count_rel_1_25+13))
    if model_dict_all_full_train[i][5]=='1.33' and model_dict_all_full_train[i][3]=='Yes':
        count_function_yes_1_33+=1
    p1['functionYes1.33']=((count_function_yes_1_33+1)/(count_rel_1_33+13))
    if model_dict_all_full_train[i][5]=='1.5' and model_dict_all_full_train[i][3]=='Yes':
        count_function_yes_1_5+=1
    p1['functionYes1.5']=((count_function_yes_1_5+1)/(count_rel_1_5+13))
    if model_dict_all_full_train[i][5]=='1.67' and model_dict_all_full_train[i][3]=='Yes':
        count_function_yes_1_67+=1
    p1['functionYes1.67']=((count_function_yes_1_67+1)/(count_rel_1_67+13))
    if model_dict_all_full_train[i][5]=='1.75' and model_dict_all_full_train[i][3]=='Yes':
        count_function_yes_1_75+=1
    p1['functionYes1.75']=((count_function_yes_1_75+1)/(count_rel_1_75+13))
    if model_dict_all_full_train[i][5]=='2.0' and model_dict_all_full_train[i][3]=='Yes':
        count_function_yes_2+=1
    p1['functionYes2.0']=((count_function_yes_2+1)/(count_rel_2+13))
    if model_dict_all_full_train[i][5]=='2.25' and model_dict_all_full_train[i][3]=='Yes':
        count_functionl_yes_2_25+=1
    p1['functionYes2.25']=((count_functionl_yes_2_25+1)/(count_rel_2_25+13))
    if model_dict_all_full_train[i][5]=='2.33' and model_dict_all_full_train[i][3]=='Yes':
        count_function_yes_2_33+=1
    p1['functionYes2.33']=((count_function_yes_2_33+1)/(count_rel_2_33+13))
    if model_dict_all_full_train[i][5]=='2.5' and model_dict_all_full_train[i][3]=='Yes':
        count_function_yes_2_5+=1
    p1['functionYes2.5']=((count_function_yes_2_5+1)/(count_rel_2_5+13))
    if model_dict_all_full_train[i][5]=='2.67' and model_dict_all_full_train[i][3]=='Yes':
        count_function_yes_2_67+=1
    p1['functionYes2.67']=((count_function_yes_2_67+1)/(count_rel_2_67+13))
    if model_dict_all_full_train[i][5]=='2.75' and model_dict_all_full_train[i][3]=='Yes':
        count_function_yes_2_75+=1
    p1['functionYes2.75']=((count_function_yes_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all_full_train[i][5]=='3.0' and model_dict_all_full_train[i][3]=='Yes':
        count_function_yes_3+=1
    p1['functionYes3.0']=((count_function_yes_3+1)/(count_rel_3+13))


    if model_dict_all_full_train[i][5]=='1.0' and model_dict_all_full_train[i][3]=='No':
        count_function_no_1+=1
    p1['functionNo1.0']=((count_function_no_1+1)/(count_rel_1+13))
    if model_dict_all_full_train[i][5]=='1.25' and model_dict_all_full_train[i][3]=='No':
        count_function_no_1_25+=1
    p1['functionNo1.25']=((count_function_no_1_25+1)/(count_rel_1_25+13))
    if model_dict_all_full_train[i][5]=='1.33' and model_dict_all_full_train[i][3]=='No':
        count_function_no_1_33+=1
    p1['functionNo1.33']=((count_function_no_1_33+1)/(count_rel_1_33+13))
    if model_dict_all_full_train[i][5]=='1.5' and model_dict_all_full_train[i][3]=='No':
        count_function_no_1_5+=1
    p1['functionNo1.5']=((count_function_no_1_5+1)/(count_rel_1_5+13))
    if model_dict_all_full_train[i][5]=='1.67' and model_dict_all_full_train[i][3]=='No':
        count_function_no_1_67+=1
    p1['functionNo1.67']=((count_function_no_1_67+1)/(count_rel_1_67+13))
    if model_dict_all_full_train[i][5]=='1.75' and model_dict_all_full_train[i][3]=='No':
        count_function_no_1_75+=1
    p1['functionNo1.75']=((count_function_no_1_75+1)/(count_rel_1_75+13))
    if model_dict_all_full_train[i][5]=='2.0' and model_dict_all_full_train[i][3]=='No':
        count_function_no_2+=1
    p1['functionNo2.0']=((count_function_no_2+1)/(count_rel_2+13))
    if model_dict_all_full_train[i][5]=='2.25' and model_dict_all_full_train[i][3]=='No':
        count_functionl_no_2_25+=1
    p1['functionNo2.25']=((count_functionl_no_2_25+1)/(count_rel_2_25+13))
    if model_dict_all_full_train[i][5]=='2.33' and model_dict_all_full_train[i][3]=='No':
        count_function_no_2_33+=1
    p1['functionNo2.33']=((count_function_no_2_33+1)/(count_rel_2_33+13))
    if model_dict_all_full_train[i][5]=='2.5' and model_dict_all_full_train[i][3]=='No':
        count_function_no_2_5+=1
    p1['functionNo2.5']=((count_function_no_2_5+1)/(count_rel_2_5+13))
    if model_dict_all_full_train[i][5]=='2.67' and model_dict_all_full_train[i][3]=='No':
        count_function_no_2_67+=1
    p1['functionNo2.67']=((count_function_no_2_67+1)/(count_rel_2_67+13))
    if model_dict_all_full_train[i][5]=='2.75' and model_dict_all_full_train[i][3]=='No':
        count_function_no_2_75+=1
    p1['functionNo2.75']=((count_function_no_2_75+1)/(count_rel_2_75+13))
    if  model_dict_all_full_train[i][5]=='3.0' and model_dict_all_full_train[i][3]=='No':
        count_function_no_3+=1
    p1['functionNo3.0']=((count_function_no_3+1)/(count_rel_3+13))

print("probabilities calculated")
# BELOW IS FOR FULL TEST.CSV
relevance_dict_full_test={}
id_uid_dict_full_test={}
title_dict_full_test={}
search_term_dict_full_test={}
initial_list_all_full_test=[]
initial_dictionary_all_full_test={}
model_dict_all_full_test={}


# ID-RELEVANCE and ID-PRODUCT UID dictionary
for i in test_full_df.itertuples():
    #relevance_dict_full_test[i[1]]=str(i[5])
    id_uid_dict_full_test[i[1]]=i[2]


#PRODUCT ID TITLE DICT
for i in test_full_df.itertuples():
    title_dict_full_test[i[2]]=i[3]


#SEARCH TERM DICT (ID-SEACRH TERM)
for i in test_full_df.itertuples():
    search_term_dict_full_test[i[1]]=i[4]

id_list=[]
for i in test_full_df._getitem_column('id'):
    initial_list_all_full_test=[tokenize_stem(search_term_dict_full_test[i]),tokenize_stem(str(title_dict_full_test[id_uid_dict_full_test[i]])),tokenize_stem(str(product_description_dict[id_uid_dict_full_test[i]])) ,tokenize_stem(str(brand_dict[id_uid_dict_full_test[i]])) if id_uid_dict_full_test[i] in brand_dict.keys() else tokenize_stem('None'),tokenize_stem(str(material_dict[id_uid_dict_full_test[i]])) if id_uid_dict_full_test[i] in material_dict.keys() else tokenize_stem('None'),tokenize_stem(str(function_dict[id_uid_dict_full_test[i]])) if id_uid_dict_full_test[i] in function_dict.keys() else tokenize_stem('None')]
    initial_dictionary_all_full_test[i]=initial_list_all_full_test
    id_list.append(i)


#FORMING YES NO TABLE of FULL TEST.CSV
for i in initial_dictionary_all_full_test:
    count_ptitle_full_test=0
    count_pdesc_full_test=0
    brand_flag_full_test=0
    material_flag_full_test=0
    function_flag_full_test=0
    counter_search_term_full_test = Counter(initial_dictionary_all_full_test[i][0])
    counter_ptitle_full_test=Counter(initial_dictionary_all_full_test[i][1])
    counter_pdesc_full_test=Counter(initial_dictionary_all_full_test[i][2])
    #print(counter_search_term_full_test)
    for word in initial_dictionary_all_full_test[i][0]:
        if word in counter_ptitle_full_test:
            count_ptitle_full_test+=counter_ptitle_full_test[word]
        if word in counter_pdesc_full_test:
            count_pdesc_full_test+=counter_pdesc_full_test[word]
        if word in initial_dictionary_all_full_test[i][3]:
            brand_flag_full_test=1
        if word in initial_dictionary_all_full_test[i][4]:
            material_flag_full_test=1
        if word in initial_dictionary_all_full_test[i][5]:
            function_flag_full_test=1
    if count_ptitle_full_test>=len(initial_dictionary_all_full_test[i][0]):
        model_dict_all_full_test[i]=['Perfect']
    elif count_ptitle_full_test<len(initial_dictionary_all_full_test[i][0]) and count_ptitle_full_test!=0:
        model_dict_all_full_test[i]=['Partial']
    else:
        model_dict_all_full_test[i]=['Irrelevant']
    if count_pdesc_full_test>=len(initial_dictionary_all_full_test[i][0]):
        model_dict_all_full_test[i].append('Perfect')
    elif count_pdesc_full_test<len(initial_dictionary_all_full_test[i][0]):
        model_dict_all_full_test[i].append('Partial')
    else:
        model_dict_all_full_test[i].append('Irrelevant')
    if brand_flag_full_test==1:
        model_dict_all_full_test[i].append('Yes')
    else:
        model_dict_all_full_test[i].append('No')
    if material_flag_full_test==1:
        model_dict_all_full_test[i].append('Yes')
    else:
        model_dict_all_full_test[i].append('No')
    if function_flag_full_test==1:
        model_dict_all_full_test[i].append('Yes')
    else:
        model_dict_all_full_test[i].append('No')



#print("test table formed")

P1={}
after_rel=0
rel_after_list_full_test=[]
for i in model_dict_all_full_test:
    title=model_dict_all_full_test[i][0]
    desc=model_dict_all_full_test[i][1]
    brand=model_dict_all_full_test[i][2]
    material=model_dict_all_full_test[i][3]
    function=model_dict_all_full_test[i][4]
    P1['1.0']=p1['title'+title+'1.0']*p1['desc'+desc+'1.0']*p1['brand'+brand+'1.0']*p1['material'+material+'1.0']*p1['function'+function+'1.0']*p1['1.0']
    P1['1.25']=p1['title'+title+'1.25']*p1['desc'+desc+'1.25']*p1['brand'+brand+'1.25']*p1['material'+material+'1.25']*p1['function'+function+'1.25']*p1['1.25']
    P1['1.33']=p1['title'+title+'1.33']*p1['desc'+desc+'1.33']*p1['brand'+brand+'1.33']*p1['material'+material+'1.33']*p1['function'+function+'1.33']*p1['1.33']
    P1['1.5']=p1['title'+title+'1.5']*p1['desc'+desc+'1.5']*p1['brand'+brand+'1.5']*p1['material'+material+'1.5']*p1['function'+function+'1.5']*p1['1.5']
    P1['1.67']=p1['title'+title+'1.67']*p1['desc'+desc+'1.67']*p1['brand'+brand+'1.67']*p1['material'+material+'1.67']*p1['function'+function+'1.67']*p1['1.67']
    P1['1.75']=p1['title'+title+'1.75']*p1['desc'+desc+'1.75']*p1['brand'+brand+'1.75']*p1['material'+material+'1.75']*p1['function'+function+'1.75']*p1['1.75']
    P1['2.0']=p1['title'+title+'2.0']*p1['desc'+desc+'2.0']*p1['brand'+brand+'2.0']*p1['material'+material+'2.0']*p1['function'+function+'2.0']*p1['2.0']
    P1['2.25']=p1['title'+title+'2.25']*p1['desc'+desc+'2.25']*p1['brand'+brand+'2.25']*p1['material'+material+'2.25']*p1['function'+function+'2.25']*p1['2.25']
    P1['2.33']=p1['title'+title+'2.33']*p1['desc'+desc+'2.33']*p1['brand'+brand+'2.33']*p1['material'+material+'2.33']*p1['function'+function+'2.33']*p1['2.33']
    P1['2.5']=p1['title'+title+'2.5']*p1['desc'+desc+'2.5']*p1['brand'+brand+'2.5']*p1['material'+material+'2.5']*p1['function'+function+'2.5']*p1['2.5']
    P1['2.67']=p1['title'+title+'2.67']*p1['desc'+desc+'2.67']*p1['brand'+brand+'2.67']*p1['material'+material+'2.67']*p1['function'+function+'2.67']*p1['2.67']
    P1['3.0']=p1['title'+title+'3.0']*p1['desc'+desc+'3.0']*p1['brand'+brand+'3.0']*p1['material'+material+'3.0']*p1['function'+function+'3.0']*p1['3.0']

    max_prob2=max(P1['1.0'],P1['1.25'],P1['1.33'],P1['1.5'],P1['1.67'],P1['1.75'],P1['2.0'],P1['2.25'],P1['2.33'],P1['2.5'],P1['2.67'],P1['3.0'])


    for j in P1:
        if P1[j] ==max_prob2:
            after_rel=j

    rel_after_list_full_test.append(float(after_rel))

#print("prob found out")

rows=zip(id_list,rel_after_list_full_test)

with open('C:/Users/Hardik/Downloads/Sarika/result.csv', 'w', encoding='charmap') as file_df:
    a = csv.writer(file_df, delimiter=',',lineterminator='\n')
    a.writerow(["id","relevance"])
    for row in rows:
        a.writerow(row)







































































