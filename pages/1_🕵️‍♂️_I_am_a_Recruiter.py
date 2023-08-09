import asyncio
import os
from typing import List

import streamlit as st
import openai



#os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
openai.api_key = st.secrets['OPENAI_API_KEY']
#openai.api_key = os.environ['OPENAI_API_KEY']



from utils import shorten, clean_string, get_job_summary_synchronous, get_cv_summary_1, process_cv_files, ExtractionSchemaFromCV, ExtractionPerson, ExtractionPersonSkill, GPT_MODEL_SHORT, GPT_MODEL_LONG, MAX_TOKENS, convert_elements_to_strings, get_best_candidate, get_person_skill_matches, wrap_legend_text, plot_people_skills, get_people_summaries_1



async def main_function_gradio(job_description: str, pdf_files: List[str]) -> str:
    job_desc_summary = get_job_summary_synchronous(job_description)
    cv_texts = process_cv_files(pdf_files)
    people_summaries = await get_people_summaries_1(cv_texts)
    overall_overview = get_best_candidate(people_summaries, job_desc_summary)

    return overall_overview

st.set_page_config(page_title="recruiter", page_icon="../logo.png", layout="wide", initial_sidebar_state="auto", menu_items=None)

st.title('You are a recruiter: finding the best candiate:')
st.markdown('Copy and paste the job description in the text field below, and upload up to 10 resume pdfs of applicants. GPT will extract information from those resumes and tell you for each required skill for that job description, which applicant is the strongest.')



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