# -*- coding: utf-8 -*-
import base64
import streamlit as st
import streamlit.components.v1 as components
from streamlit_chat import message
import pandas as pd
import numpy as np
# from googletrans import Translator

from pymongo import MongoClient
import gridfs
import certifi
ca = certifi.where()

from Answer.model import ExtractiveModel
from Answer.tfidf import TfidfVector
from Answer.document_selector import DocSelector
from Answer.pymu_extract import PyMuExtract

st.set_page_config(layout="wide", page_title='ACCIA Bot')

@st.cache
def load_data(file_name):
    path = f'data\{file_name}'
    return pd.read_csv(path)

@st.cache(allow_output_mutation=True)
def connect_mongo_db():
    try:
        cluster = MongoClient("mongodb+srv://accia_nuc:cxapJk82Zb4GZmdP@cluster0.mhmhq61.mongodb.net/?retryWrites=true&w=majority", tlsCAFile=ca)
        print("MOngo DB connected")
        db = cluster["accia"]
        return db
    except Exception as e:
        print("Error in mongo connection : ", e)

@st.cache(allow_output_mutation=True)
def load_qna_model():
    qna_model = ExtractiveModel('etalab-ia/camembert-base-squadFR-fquad-piaf', 'etalab-ia/camembert-base-squadFR-fquad-piaf')
    return qna_model

def get_qna_answer(model, q, doc_selector):
    answers = []
    scores = []
    paras = []
    selected_doc = doc_selector.get_selected_doc(q)
    #pdf = Pdf(f".\PDFs\\{selected_doc}")

    file = db.fs.files.find_one({'name': selected_doc})
    my_id = file['_id']
    outputdata = fs.get(my_id).read()

    pdf = PyMuExtract(outputdata)
    vectorizer = TfidfVector(pdf)
    paras_tfidf = vectorizer.get_top_paras(q, 5)
    for p in paras_tfidf:
        paras.append(pdf.paragraphs[p])
        scores.append(model.get_answer(q, pdf.paragraphs[p])['score'])
        answers.append(model.get_answer(q, pdf.paragraphs[p])['answer'])

    top_score_answer = answers[np.argmax(np.array(scores))]
    top_para_answer = answers[0]
    top_para = paras[0]
    top_para_top_score = paras[np.argmax(np.array(scores))]
    return top_score_answer, top_para_answer, selected_doc, top_para, top_para_top_score, outputdata

def get_def_sigle_nuc(sigle, sigles_nuc):
    df_temp = sigles_nuc[sigles_nuc['sigle'] == sigle]
    return df_temp['def_fr'], df_temp['def_ang']

def get_sentence_in_para(p, text):
    sentences = p.split('.')
    if text[-1] == '.':
        text = text[:-1]
    for s in sentences:
        if text in s:
            return s

def show_pdf_mongo(file_data, db, fs):
    base64_pdf = base64.b64encode(file_data).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)



# Laoding bot with transformer model*
with st.spinner("Chargement de la page ..."):
    qna_model = load_qna_model()
    #spell = Speller(lang='fr')
    # Test connection to extraction image API
    db = connect_mongo_db()
    fs = gridfs.GridFS(db)
    doc_selector = DocSelector(db, fs)

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = [['Bonjour, je suis un BOT qui peut vous aider si vous avez des questions sur le domaine du nucléaire.', False]]

for i,message_ in enumerate(st.session_state['message_history']):
    message(message_[0], is_user=message_[1], key=str(i)) 

placeholder_user = st.empty()
placeholder_bot = st.empty()  # placeholder for latest message
user_input = st.text_input("you:","", key="input")

if user_input :
    st.session_state['message_history'].append([user_input, True])
    with placeholder_user.container():
        message( st.session_state.message_history[-1], is_user=True, key='last_user') 
    
    _, top_para_a, selected_doc, top_para, _, selected_doc_data = get_qna_answer(qna_model, user_input, doc_selector)
    top_para_s = get_sentence_in_para(top_para, top_para_a)
    st.session_state['message_history'].append([top_para_s, False])

    # display the latest message
    with placeholder_bot.container():
        message(st.session_state.message_history[-1], is_user=False, key='last_answer')

