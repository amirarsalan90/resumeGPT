import asyncio
from typing import List
import os
import streamlit as st

import matplotlib.pyplot as plt

import openai
from pydantic import BaseModel

from pypdf import PdfReader
from utils import (clean_string, convert_elements_to_strings, get_best_candidate, 
                   get_cv_summary, get_job_summary, shorten, ExtractionSchema, ExtractionPersonSkill, get_person_skill_matches, plot_people_skills)


openai.api_key = os.environ["OPENAI_API_KEY"]

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


async def get_people_summaries(cv_texts: List[str]) -> List[ExtractionSchema]:
    system_prompt_person_summary = "You are an AI assistant whose task is to extract and summarize information written on a resume of a job seeker"
    tasks = [get_cv_summary(system_prompt_person_summary, cv_text, ExtractionSchema) for cv_text in cv_texts]
    return await asyncio.gather(*tasks)



async def main_function_gradio(job_description: str, pdf_files: List[str]) -> str:
    job_desc_summary = get_job_summary(job_description)
    cv_texts = process_cv_files(pdf_files)
    people_summaries = await get_people_summaries(cv_texts)
    overall_overview = get_best_candidate(people_summaries, job_desc_summary)

    return overall_overview



with st.form("my_form"):
    c1, c2 = st.columns(2)
    with c1:
        st.write("Inside the form")
        job_description_st = st.text_area('Copy and paste the job description here:')
        uploaded_files = st.file_uploader('Upload up to 10 resume PDFs', accept_multiple_files=True)


    # Every form must have a submit button.
        submitted = st.form_submit_button("Get Best Applicant")
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