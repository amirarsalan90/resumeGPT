import streamlit as st

st.set_page_config(page_title="recruiter", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
st.title('resumeGPT')

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
<style>
.bigger-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="bigger-font">\U000026A1 Powered by the OpenAI GPT models, this app serves as your personal assistant in the recruitment or job applicaiton process! </p>', unsafe_allow_html=True)

st.markdown('<p class="big-font"> <a href="https://resumegpt.streamlit.app/I_am_a_Recruiter"><strong>If you are a recruiter</strong></a>, copy and paste the job description into the text field, then upload the applicants\' resumes in pdf format, and press submit! You\'ll receive a report \U0001F4C1, along with an illustrative plot \U0001F4CA that compares the applicants\' strengths in the required skills. This way, you can quickly identify the top talent for the job!</p>', unsafe_allow_html=True)

st.markdown('<p class="big-font"> <a href="https://resumegpt.streamlit.app/I_am_an_Applicant"><strong>If you are an applicant</strong></a>, copy and paste the job description into the text field, then upload your resume in pdf format, and press submit! You\'ll receive some suggestion on how to modify your resume to match better with the job description.</p>', unsafe_allow_html=True)

st.markdown('<p class="big-font"> \U0001F4B8 Every time you use this app, I am being charged by the OpenAI API. Please consider donating <a href="https://donate.stripe.com/cN2dUMe379mNcLu9AA">here</a> \U0001F4B0 \U0001F64F</p>', unsafe_allow_html=True)




st.markdown("<small>Created by Amirarsalan Rajabi :pencil2:</small>", unsafe_allow_html=True)
st.markdown("[GitHub](https://github.com/amirarsalan90)")
st.markdown("[Website](https://amirarsalan90.github.io)")