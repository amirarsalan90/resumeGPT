# resumeGPT

This is a simple streamlit app that uses OpenAI API:

1) If you are a recruiter, copy and paste the job description into the text field, then upload the applicants' resumes in pdf format, and press submit! You'll receive a report 📁, along with an illustrative plot 📊 that compares the applicants' strengths in the required skills. This way, you can quickly identify the top talent for the job!

2) If you are an applicant, copy and paste the job description into the text field, then upload your resume in pdf format, and press submit! You'll receive some suggestion on how to modify your resume to match better with the job description.

First, run the following: ```chmod +x run.sh```
Then, export your OpenAI API key to OPENAI_API_KEY evironment variable.
Finally, ```./run.sh``` and your app will be up and running!

A live demo is available here:
[https://resumegpt.streamlit.app](https://resumegpt.streamlit.app)

