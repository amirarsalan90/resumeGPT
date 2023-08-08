import asyncio
import os
import json
from collections import defaultdict
from typing import Dict, List, Type, Union

import streamlit as st
from pydantic import BaseModel
from pypdf import PdfReader
import openai
import tiktoken
import textwrap
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator


#os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
#openai.api_key = st.secrets['OPENAI_API_KEY']
openai.api_key = os.environ['OPENAI_API_KEY']


# Constants
GPT_MODEL_SHORT = "gpt-3.5-turbo-0613"
GPT_MODEL_LONG = "gpt-3.5-turbo-16k"
MAX_TOKENS = 3500


class ExtractionSchemaFromCV(BaseModel):
    person_name: str
    person_education: str
    person_work_experience: str
    person_skills: str
    person_projects: str
    person_software: str


def shorten(message_text: str, gpt_model: str = GPT_MODEL_LONG, max_tokens: int = MAX_TOKENS) -> str:
    encoding = tiktoken.encoding_for_model(GPT_MODEL_SHORT)
    tokens = encoding.encode(message_text)
    return encoding.decode(tokens[:MAX_TOKENS])


def clean_string(input_string: str) -> str:
    # Remove non-ASCII characters and control characters from the input string
    cleaned_string = "".join(char for char in input_string if 32 <= ord(char) < 128 or char in '\t\n\r')
    return cleaned_string



async def get_job_summary(job_description: str) -> str:
    messages=[]
    system_prompt = "you are an AI assistant that is supposed to help user summarize a job description, pointing out its most important requirements and information. The user provides you with a job description, you return a shorter summary of that job description, including all important requirements and information"
    job_description = shorten(job_description, GPT_MODEL_LONG, 14000)
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": job_description})
    response = await openai.ChatCompletion.acreate(
        temperature = 0,
        model=GPT_MODEL_LONG,
        messages=messages)
    return response.choices[0].message.content


async def get_cv_summary(cv_text: str, extraction_pydantic_object: Type[BaseModel]) -> Dict:
    messages=[]
    system_prompt = "You are an AI assistant whose task is to extract and summarize information written on a resume of a job applicant"
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": cv_text})

    response = await openai.ChatCompletion.acreate(
        temperature = 0,
        model=GPT_MODEL_SHORT,
        messages=messages,
        functions=[
            {
            "name": "extract_resume_information",
            "description": "Get the information of a person",
            "parameters": extraction_pydantic_object.schema()
            }
        ],
        function_call={"name": "extract_resume_information"}
    )
    try:
        return json.loads(response.choices[0].message.function_call.arguments)
    except json.JSONDecodeError:
        return response.choices[0].message.function_call.arguments


def process_cv_file(pdf_file):
    pdf = PdfReader(pdf_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    
    return text

def get_overal_person_suggestion(job_desc_summary, person_summary):
    messages=[]
    system_prompt = "You are an AI assistant. A user gives you a job description and their resume. You have to analyze the provided job description and resume. Using markdown formatting, give concise suggestions for the most important changes the user should make to their resume to match the job. Bold key points."

    user_message = f""" 
    Job description:
    {job_desc_summary}

    person resume summary:
    {person_summary}
    """
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    response = openai.ChatCompletion.create(
        temperature = 0,
        model=GPT_MODEL_LONG,
        messages=messages,
    )
    return response.choices[0].message['content']



st.set_page_config(page_title="recruiter", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
    

async def main_function_gradio(job_description: str, pdf_file) -> str:
    cv_text = process_cv_file(pdf_file)
    #job_desc_summary = await get_job_summary(job_description)
    #person_summary = await get_cv_summary(cv_text, ExtractionSchemaFromCV)
    tasks = [get_job_summary(job_description), get_cv_summary(cv_text, ExtractionSchemaFromCV)]
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