import asyncio
import os

import streamlit as st
import openai


#os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
#openai.api_key = st.secrets['OPENAI_API_KEY']
openai.api_key = os.environ['OPENAI_API_KEY']


from utils import shorten, clean_string, get_job_summary, get_cv_summary_2, process_cv_file_single, get_overal_person_suggestion, ExtractionSchemaFromCV2, GPT_MODEL_SHORT, GPT_MODEL_LONG, MAX_TOKENS

st.set_page_config(page_title="applicant", page_icon="../logo.png", layout="wide", initial_sidebar_state="auto", menu_items=None)
    

async def main_function_gradio(job_description: str, pdf_file) -> str:
    cv_text = process_cv_file_single(pdf_file)
    tasks = [get_job_summary(job_description), get_cv_summary_2(cv_text, ExtractionSchemaFromCV2)]
    job_desc_summary, person_summary = await asyncio.gather(*tasks)
    overall_overview = get_overal_person_suggestion(job_desc_summary, person_summary)

    return overall_overview


st.title('Your are a job applicant:')
st.markdown('copy and paste the job description that you are applying for in the text field below. Then upload your resume, and GPT will tell you how to modify your resume to better match that job description.')



with st.form("my_form"):
    c1, c2 = st.columns(2)
    with c1:
        job_description_st = st.text_area('Copy and paste the full job posting description here:')
        uploaded_file = st.file_uploader('Upload your resume (PDF)')


    # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
    if submitted:
        with c2:
            with st.spinner('Generating Report....'):
                #pdf_files_texts = process_cv_files(uploaded_files)
                candidates_report = asyncio.run(main_function_gradio(job_description_st, uploaded_file))
                st.write(candidates_report)