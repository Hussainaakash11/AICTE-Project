import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- FUTURISTIC STYLING --------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    
    body {
        background-color: #0a0a0a;
        color: #ffffff;
        font-family: 'Orbitron', sans-serif;
    }
    
    .stTextArea, .stTextInput, .stFileUploader {
        background-color: #1a1a1a !important;
        color: #00ffcc !important;
        border-radius: 10px;
    }
    
    .stDataFrame {
        border: 2px solid #00ffcc;
        border-radius: 10px;
        box-shadow: 0px 0px 10px #00ffcc;
    }
    
    .css-1d391kg {  
        background-color: #0a0a0a;  
    }
    
    .css-1v0mbdj a {
        color: #00ffcc !important;
    }

    .stButton>button {
        border-radius: 10px;
        background-color: #00ffcc !important;
        color: #000000 !important;
        font-weight: bold;
    }

    </style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.image("https://th.bing.com/th/id/OIP.TgcrBy9oxkfFz1ATKy7C3wHaHN?rs=1&pid=ImgDetMain", width=200)  # Futuristic AI logo
st.title("AI Resume Screening & Candidate Ranking System")

# -------------------- FUNCTION TO EXTRACT TEXT --------------------
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
    return text

# -------------------- FUNCTION TO RANK RESUMES --------------------
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]

    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

# -------------------- JOB DESCRIPTION INPUT --------------------
st.header("💼 Job Description")
job_description = st.text_area("Enter the job description", height=150)

# -------------------- UPLOAD RESUMES --------------------
st.header("📂 Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

# -------------------- PROCESS FILES --------------------
if uploaded_files and job_description:
    st.header("⚡ Ranking Resumes")
    
    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    # Loading Animation
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        progress_bar.progress(percent_complete + 1)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Display results
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)

    st.success("✅ Ranking Complete!")
    st.dataframe(results)

# -------------------- FOOTER --------------------
st.markdown("<br><br><center>🚀 Made With ❤️ by Aakash Hussain | Designed for the Future</center>", unsafe_allow_html=True)
