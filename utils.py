import json
from typing import Dict, List, Type, Union

import openai
from pydantic import BaseModel
import tiktoken

from matplotlib.figure import Figure

import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from matplotlib.patches import Patch
import textwrap
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator


# Constants
GPT_MODEL_SHORT = "gpt-3.5-turbo-0613"
GPT_MODEL_LONG = "gpt-3.5-turbo-16k"
MAX_TOKENS = 3500


class ExtractionSchemaFromCV(BaseModel):
    person_name: str
    person_education: str
    person_work_experience: str
    person_skills: str


class ExtractionPerson(BaseModel):
    person: str
    skill: str

class ExtractionPersonSkill(BaseModel):
    get_skill: List[ExtractionPerson]


def shorten(message_text: str, gpt_model: str = GPT_MODEL_LONG, max_tokens: int = MAX_TOKENS) -> str:
    encoding = tiktoken.encoding_for_model(GPT_MODEL_SHORT)
    tokens = encoding.encode(message_text)
    return encoding.decode(tokens[:MAX_TOKENS])


def clean_string(input_string: str) -> str:
    # Remove non-ASCII characters and control characters from the input string
    cleaned_string = "".join(char for char in input_string if 32 <= ord(char) < 128 or char in '\t\n\r')
    return cleaned_string


def convert_elements_to_strings(input_list: List[Union[Dict, str]]) -> List[str]:
    output_list = [json.dumps(element, indent=2) if isinstance(element, dict) else str(element)
                   for element in input_list]
    return output_list


def get_job_summary(job_description: str) -> str:
    messages=[]
    system_prompt = "you are an AI assistant that is supposed to help user summarize a job description, pointing out its most important requirements and information. The user provides you with a job description, you return a shorter summary of that job description, including all important requirements and information"
    job_description = shorten(job_description, GPT_MODEL_LONG, 14000)
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": job_description})
    response = openai.ChatCompletion.create(
        model=GPT_MODEL_LONG,
        messages=messages)
    return response.choices[0].message.content


async def get_cv_summary(system_prompt: str, cv_text: str, extraction_pydantic_object: Type[BaseModel]) -> Dict:
    messages=[]
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": cv_text})

    response = await openai.ChatCompletion.acreate(
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


def get_best_candidate(persons_result: List[Dict], job_desc_summary: str) -> str:
    persons_result_strings = convert_elements_to_strings(persons_result)
    messages=[]
    system_prompt = """
    You are an AI assistant helping a HR recruiter find the best candidate for a job. After reviewing the job description, determine the essential characteristics and skills for the job, regardless of the number. Then identify the best candidate for each skill. Summarize this in a brief, concise manner. Do not make up any information. Here is the format:

    Essential Skill 1 (e.g., "Software Engineering skills"): Best candidate is Person X because...
    Essential Skill 2 (e.g., "Education"): Best candidate is Person X because...
    Essential Skill 3 (e.g., "Knowledge of programming languages"): Best candidate is Person Y because...
    Essential Skill 4 (e.g., "Leadership"): Best candidate is Person Z because...

    Overall: Person X is the best candidate because...
    """
    user_prompt = f"The job description: \n{job_desc_summary}"
    user_prompt += "\n\n".join(f"{person}\n" for person in persons_result_strings)
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    response = openai.ChatCompletion.create(
        model=GPT_MODEL_LONG,
        messages=messages)
    
    return response.choices[0].message.content


def get_person_skill_matches(job_evaluation_text: str, extraction_pydantic_object: Type[BaseModel]) -> Dict:
    system_prompt = "You are an AI assistant. A text containing the evaluation of applicant for a job description is given to you. Your task is to extract all the skill titles and the best person for those skills from the prompt given by user."
    
    messages=[]
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": job_evaluation_text})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=[
            {
            "name": "extract_skill_person_match",
            "description": "Get skills and best person for that skill",
            "parameters": extraction_pydantic_object.schema()
            }
        ],
        function_call={"name": "extract_skill_person_match"}
    )
    try:
        return json.loads(response.choices[0].message.function_call.arguments)
    except json.JSONDecodeError:
        return response.choices[0].message.function_call.arguments


def wrap_legend_text(text: str, width : int =15) -> str:
    return '\n'.join(textwrap.wrap(text, width))

def plot_people_skills(data: Dict[str, List[Dict[str, str]]]) -> Figure:
    # gather the skills each person has
    skill_dict = defaultdict(list)
    all_skills = []
    for person_skill in data['get_skill']:
        skill_dict[person_skill['person']].append(person_skill['skill'])
        all_skills.append(person_skill['skill'])

    # unique set of skills
    unique_skills = list(set(all_skills))

    # create a color palette for each unique skill
    colors = sns.color_palette('hsv', len(unique_skills))

    # map each skill to a color
    color_dict = dict(zip(unique_skills, colors))

    people = list(skill_dict.keys())
    people_indices = np.arange(len(people))

    # use a different style
    plt.style.use('ggplot')

    # create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = np.zeros(len(people))
    for i, (person, skills) in enumerate(skill_dict.items()):
        for j, skill in enumerate(skills):
            ax.bar(people_indices[i], 1, bottom=bottom[i], color=color_dict[skill], edgecolor='white')
            # wrap text to next line if it's too long
            wrapped_skill = textwrap.fill(skill, 15)
            ax.text(people_indices[i], bottom[i] + 0.5, wrapped_skill, ha='center', va='center', fontsize=8, color='black')
            bottom[i] += 1

    # use FontProperties to make x-tick labels bold
    font = FontProperties()
    font.set_weight('bold')

    # wrap x-tick labels
    wrapped_people = [textwrap.fill(person, 15) for person in people]
    ax.set_xticks(people_indices)
    ax.set_xticklabels(wrapped_people, rotation=0, fontsize=8)
    ax.set_ylabel('Number of Skills')
    ax.set_title('Comparison of Skills per Person')

    # set y-ticks to only include whole numbers
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # create a legend
    legend_elements = [Patch(facecolor=color_dict[skill], edgecolor='white', label=skill) for skill in unique_skills]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    # add gridlines
    ax.grid(True)
    plt.tight_layout()
    return fig
