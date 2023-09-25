!pip install langchain
# !pip install openai
!pip install PyPDF2
# !pip install faiss-cpu
# !pip install tiktoken

from PyPDF2 import PdfReader
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS

pdfreader = PdfReader('/content/Maleria docs.pdf')

from typing_extensions import Concatenate
# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

raw_text

# # We need to split the text using Character Text Split such that it sshould not increse token size
# text_splitter = CharacterTextSplitter(
#     separator = "\n",
#     chunk_size = 800,
#     chunk_overlap  = 200,
#     length_function = len,
# )
# texts = text_splitter.split_text(raw_text)

# len(texts)
# print(texts)

# !pip install InstructorEmbedding
# !pip install -U sentence-transformers

# from sentence_transformers import SentenceTransformer
# sentences = texts

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddings = model.encode(sentences)
# print(embeddings)

# print(texts)

from transformers import pipeline

# Initialize the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# User questions
questions = [
    "What is malaria?",
    "How is malaria transmitted?",
]
# context = " ".join(texts)

# Process questions and get answers
for question in questions:
    answers = qa_pipeline(question=question, context=raw_text)  
    print(f"Question: {question}")
    print(f"Answer: {answers['answer']}")
    print(f"Confidence: {answers['score']}")



    # Please try in Google collab