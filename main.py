import asyncio
from typing import List
import os
import streamlit as st

import matplotlib.pyplot as plt

import openai
from pydantic import BaseModel

from pypdf import PdfReader
from utils import (clean_string, convert_elements_to_strings, get_best_candidate, 
                   get_cv_summary, get_job_summary, shorten, ExtractionSchemaFromCV, ExtractionPersonSkill, get_person_skill_matches, plot_people_skills)


#os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
openai.api_key = st.secrets['OPENAI_API_KEY']

st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

def process_cv_files(pdf_files):
    cv_texts = []
    for i in pdf_files:
        pdf = PdfReader(i)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        cv_texts.append(clean_string(text))
    return cv_texts


async def get_people_summaries(cv_texts: List[str]) -> List[ExtractionSchemaFromCV]:
    system_prompt_person_summary = "You are an AI assistant whose task is to extract and summarize information written on a resume of a job applicant"
    tasks = [get_cv_summary(system_prompt_person_summary, cv_text, ExtractionSchemaFromCV) for cv_text in cv_texts]
    return await asyncio.gather(*tasks)



async def main_function_gradio(job_description: str, pdf_files: List[str]) -> str:
    job_desc_summary = get_job_summary(job_description)
    cv_texts = process_cv_files(pdf_files)
    people_summaries = await get_people_summaries(cv_texts)
    overall_overview = get_best_candidate(people_summaries, job_desc_summary)

    return overall_overview


st.title('ResumeGPT')
with st.sidebar:
    st.title('ResumeGPT')
    st.markdown("<small>:money_with_wings: Every time you use this app, I am being charged by the OpenAI API. Please consider donating [here](https://donate.stripe.com/cN2dUMe379mNcLu9AA) :moneybag: :pray:", unsafe_allow_html=True)


    st.markdown("<small>:zap: Powered by the OpenAI GPT models, this app serves as your personal assistant in the recruitment process! Simply copy and paste the job description into the text field, then upload the applicants' resumes in pdf format, and press submit! :clipboard: :file_folder:", unsafe_allow_html=True)

    st.write("<small>:bar_chart: You'll receive a report, along with an illustrative plot that compares the applicants' strengths in the required skills. This way, you can quickly identify the top talent for the job!", unsafe_allow_html=True)

    st.markdown("<small>Created by Amirarsalan Rajabi :pencil2:</small>", unsafe_allow_html=True)
    st.markdown("[GitHub](https://github.com/amirarsalan90)")
    st.markdown("[Website](https://amirarsalan90.github.io)")



with st.form("my_form"):
    c1, c2 = st.columns(2)
    with c1:
        job_description_st = st.text_area('Copy and paste the full job posting description here:')
        uploaded_files = st.file_uploader('Upload up to 10 resume PDFs', accept_multiple_files=True)


    # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
    if submitted:
        with c2:
            with st.spinner('Generating Report....'):
                #pdf_files_texts = process_cv_files(uploaded_files)
                candidates_report = asyncio.run(main_function_gradio(job_description_st, uploaded_files))
                st.write(candidates_report)
            with st.spinner('Plotting people skills....'):
                res = get_person_skill_matches(candidates_report, ExtractionPersonSkill)
                st.write("---")
                #st.write(res)
                fig = plot_people_skills(res)

                st.pyplot(fig)