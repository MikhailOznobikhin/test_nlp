# -*- coding: utf-8 -*-

import pandas as pd
import pymorphy2
import nltk
from nltk.corpus import stopwords
from string import punctuation
    
df = pd.read_csv('test_data.csv')
#оставляем только менеджерские сообщения
df = df[df.role == 'manager']


stop_words= stopwords.words('russian')
newStopWords = ['ага', 'угу', 'алло', 'ну', 'да', 'a']
stop_words.extend(newStopWords)

df_orig = pd.read_csv('test_data.csv')
df = df_orig.copy()
#оставляем только менеджерские сообщения
df = df[df.role == 'manager']

# фразы приветствия
greetings_phrase = ['добрый день','добрый'
                   ,'добрый вечер'
                   ,'здравствуйте'
                   ,'приветствую']

# фразы прощания 
parting_phrase = ['свидания', 'доброго'
                 ,'всего доброго'
                 ,'всего хорошего'
                 ,'хорошего дня'
                 ,'хорошего вечера']

#фразы при представлении
hello_phrase = ['это','зовут']

def preprocess(text):   
    lem = nltk.word_tokenize(text.lower())
    lem_clear = [w for w in lem if w not in stop_words
                 and w != ' '
                 and w.strip() not in punctuation]
    text = " ".join(lem_clear)
    return text

def get_greeting(text):
    for i in greetings_phrase:
        if text.find(i) >= 0:
            return True

def get_parting(text):
    for i in parting_phrase:
        
        if text.find(i) >= 0:
            return True

prob_thresh = 0.5
morph = pymorphy2.MorphAnalyzer()

def get_name(text):
    for i in hello_phrase:
      if text.find(i) >= 0:
          for word in nltk.word_tokenize(text):
              for p in morph.parse(word):
                  if 'Name' in p.tag and p.score >= prob_thresh:
                      return p.normal_form        

def get_company(text):
    name = ''
    if text.find('компания') >= 0:
        token_text = nltk.word_tokenize(text)
        for word in token_text[token_text.index('компания')+1:]:
            p = morph.parse(word)[0]
            #Не горжусь этим решеним, но не придумал как сделать иначе
            if 'ADJF' in p.tag or 'NOUN' in p.tag and 'gent' not in p.tag:
                name+=p.word + ' '
            else:
                break
    return name


df_orig['insight'] = None
df_orig['manager'] = None
df_orig['company'] = None

df['Greeting'] = None
df['Parting'] = None
df['Manager'] = None
df['Company'] = None

df['text'] = df['text'].apply(preprocess)

for i, row in df.iterrows():
    name = get_name(row.text)
    greting = get_greeting(row.text)
    parting = get_parting(row.text)
    company = get_company(row.text)
    
    if name:
        df_orig.loc[i,'manager'] = name
        df.loc[i,'Manager'] = name
        
    if len(company) > 1:
        df_orig.loc[i,'company'] = company
        df.loc[i,'Company'] = company
    
    if greting:
        df_orig.loc[i,'insight'] = 'greeting=True'
        df.loc[i,'Greeting'] = True
    
    if parting:
        df_orig.loc[i,'insight'] = 'parting=True'
        df.loc[i,'Parting'] = True

df_orig.to_csv('result.csv', encoding="utf-8")

#не указан формат вывода f требования, поэтому сделал так
for i in df.groupby(by = 'dlg_id'):
    if i[1]["Greeting"].isin([True]).any() and i[1]["Parting"].isin([True]).any():
        print('Вежливый менеджер в диалоге', i[0]) 
    else:
        print('Менеджер не поприветствовал/не попрощался с клиентом в диалоге', i[0])                
        

