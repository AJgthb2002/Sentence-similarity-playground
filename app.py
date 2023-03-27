import streamlit as st
import requests
import json
from sentence_transformers import SentenceTransformer, util
from models_data import models

st.set_page_config(page_title="Sanvadita", page_icon=":sparkles:",  initial_sidebar_state="expanded", layout="wide")
st.header('Sentence Similarity')

# Define the custom CSS rules

custom_css = """
<style>
div.stTextInput > div > div > input {
    font-size: 20px;
    color: blue;
    background-color: lightyellow;
    border-radius: 10px;
    border: 2px solid black;
    padding: 10px;
}
label[for="source"] {
    font-size: 24px;
}
</style>
"""

# Add the custom CSS rules to the app
st.markdown(custom_css, unsafe_allow_html=True)

# Set the title
# st.markdown('<span style="font-size:32px;font-weight: bold;">Sentence Similarity </span>', unsafe_allow_html=True)
# st.title('Sentence Similarity')

# Create the sidebar inputs
st.sidebar.title('Select a language')
lang = st.sidebar.selectbox("",(models.keys()))
st.sidebar.title('Select a model')
modelopt = st.sidebar.selectbox('',(models[lang]['model_names'].keys()))
st.sidebar.title('Choose an example')
egs_list= list(models[lang]['examples'].keys())
egs_list.insert(0,'None')
exampleopt = st.sidebar.selectbox('',egs_list)

if exampleopt == 'None':
    # Create the text inputs
    st.markdown(f'<span style="font-size:24px">Source Sentence: </span>', unsafe_allow_html=True)
    sourcetext = st.text_input('', key='source', label_visibility='collapsed')
    st.markdown(f'<span style="font-size:24px">Sentences to compare to: </span>', unsafe_allow_html=True)
    text1 = st.text_input('', key='target1', label_visibility='collapsed')
    text2 = st.text_input('', key='target2', label_visibility='collapsed')
    text3 = st.text_input('', key='target3', label_visibility='collapsed')
else:
    st.markdown(f'<span style="font-size:24px">Source Sentence: </span>', unsafe_allow_html=True)
    sourcetext = st.text_input('', value=models[lang]['examples'][exampleopt]['source'],key='source', label_visibility='collapsed')
    st.markdown(f'<span style="font-size:24px">Sentences to compare to: </span>', unsafe_allow_html=True)
    text1 = st.text_input('', value=models[lang]['examples'][exampleopt]['targets'][0],key='target1', label_visibility='collapsed')
    text2 = st.text_input('', value=models[lang]['examples'][exampleopt]['targets'][1],key='target2', label_visibility='collapsed')
    text3 = st.text_input('', value=models[lang]['examples'][exampleopt]['targets'][2],key='target3', label_visibility='collapsed')



# Define the API endpoint and payload
API_TOKEN=st.secrets["API_KEY"]
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/"+models[lang]['model_names'][modelopt]

similarity_result=[0,0,0]
# Call the API and parse the response
def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

input_data={"inputs": {
            "source_sentence": sourcetext,
            "sentences": [text1,text2,text3],
        },
        "options":{"use_cache":True, "wait_for_model":True}}
  
def find_cosine_sim(model, text1, text2):
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    return util.cos_sim(embeddings1, embeddings2).item()


def model_scores(model_name, source, target1, target2, target3):
    # Load the pre-trained model
    model = SentenceTransformer(model_name)
    cosine_scores=[find_cosine_sim(model, source, sent) for sent in [target1,target2,target3]]
    return cosine_scores


if st.button('Compute', type="primary"):     

# Display the similarity score on the main page
    if modelopt in ['MahaSBERT-STS', 'MahaSBERT','HindSBERT-STS']:
        similarity_result=query(input_data)  
    else:    
        similarity_result= model_scores(models[lang]['model_names'][modelopt], sourcetext, text1, text2, text3)
    if sourcetext!="":
        st.markdown(f'<span style="font-size:24px">Similarity scores: </span>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])

        similarity_score1= "" if (sourcetext.strip()=="" or text1.strip()=="") else round(similarity_result[0],3)
        col1.markdown(f'<span style="font-size:20px">{text1.strip()} </span>', unsafe_allow_html=True)
        col2.markdown(f'<span style="font-size:20px;font-weight: bold">{similarity_score1} </span>', unsafe_allow_html=True)

        similarity_score2= "" if (sourcetext.strip()=="" or text2.strip()=="") else round(similarity_result[1],3)
        col1.markdown(f'<span style="font-size:20px">{text2.strip()} </span>', unsafe_allow_html=True)
        col2.markdown(f'<span style="font-size:20px;font-weight: bold">{similarity_score2} </span>', unsafe_allow_html=True)

        similarity_score3= "" if (sourcetext.strip()=="" or text3.strip()=="") else round(similarity_result[2],3)
        col1.markdown(f'<span style="font-size:20px">{text3.strip()} </span>', unsafe_allow_html=True)
        col2.markdown(f'<span style="font-size:20px;font-weight: bold">{similarity_score3} </span>', unsafe_allow_html=True)      


