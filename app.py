import os
import logging
import sys
import psycopg2
import json
import dotenv
import firebase_admin
from langchain_openai import ChatOpenAI # type: ignore
from typing import List, Dict
from datetime import datetime, timedelta
from langchain_core.documents import Document
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta, timezone
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings # type: ignore
from langchain_community.vectorstores.pgvecto_rs import PGVecto_rs # type: ignore
from flask import Flask, request, jsonify

import traceback
logging.basicConfig(level=logging.INFO)


dotenv.load_dotenv()
# Initialize Firebase Admin
cred = credentials.Certificate('totoys.json')
firebase_admin.initialize_app(cred)

# Firestore client
db = firestore.client()

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize the embedding model
model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
)


def get_question_history(student_id: int) -> List[str]:
    """
    Retrieve the question history for a student from the last hour in Firebase Firestore.

    Args:
        student_id: The ID of the student

    Returns:
        List of questions asked by the student in the last hour
    """
    # Get the current time and calculate the time one hour ago
    now = datetime.now(timezone.utc)
    one_hour_ago = now - timedelta(hours=1)

    questions_asked = []

    try:
        # Fetch the document corresponding to the student
        doc_ref = db.collection('user_questions').document(str(student_id))
        doc = doc_ref.get()

        if doc.exists:
            data = doc.to_dict()
            ask_questions = data.get("ask_questions", [])

            # Iterate over the list of questions
            for question_data in ask_questions:
                question_time = question_data.get("time")
                if question_time:
                    # Convert the `time` field to a timezone-aware datetime object
                    question_time = datetime.fromisoformat(question_time).replace(tzinfo=timezone.utc)

                    # Check if the question was asked within the last hour
                    if question_time <= one_hour_ago:
                        question = question_data.get("question")
                        questions_asked.append(question)
        print(questions_asked)                        
        return questions_asked

    except Exception as e:
        print(f"Error retrieving question history: {str(e)}")
        return []
 

def get_context(db: str, query: str) -> List[Document]:
    """
    Retrieve relevant context from the vector database.
    
    Args:
        db: The name of the database to query
        query: The search query
        
    Returns:
        List of relevant documents
    """
    PORT = os.getenv("DB_PORT")
    HOST = os.getenv("DB_HOST")  
    USER = os.getenv("DB_USER")
    PASS = os.getenv("DB_PASS")
    DB_NAME = db

    # Define the connection URL
    URL = f"postgresql+psycopg2://{USER}:{PASS}@{HOST}:{PORT}/{DB_NAME}"
    
    try:
        db_instance = PGVecto_rs.from_collection_name(
            embedding=embeddings,
            db_url=URL,
            collection_name=db,
        )
        data = db_instance.similarity_search(query, k=2)
        return data
    except Exception as e:
        logging.error(f"Error in PGVecto_rs: {str(e)}")
        return []


# Initialize the language model
llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    api_key=os.getenv("open_api_key")
)

# Define the prompt template for the AI teacher
template = """
You are a virtual voice-based AI teacher developed by TotoysAI, designed to provide clear and detailed answers for students across all grade levels. Your responses should be tailored to the student's level and the subject matter.

Guidelines:
1. **Adaptive Response Depth**:
   - For "What is..." questions: Provide a concise 3-5 line answer with a simple example
   - For "Explain" or "Tell me about..." questions: Give a detailed explanation with:
     * Key concepts broken down into simple terms
     * Real-world examples that students can relate to
     * Important points to remember
   
2. **Context Awareness**:
   - Use the provided context to ensure accurate, curriculum-aligned answers
   - Reference previous questions to maintain continuity in the learning journey
   - If a concept builds on previous topics, briefly mention the connection

3. **Student-Friendly Language**:
   - Use clear, age-appropriate vocabulary
   - Break down complex terms when they're necessary
   - Include helpful analogies when possible

4. **Response Structure**:
   - Start with a direct answer to the question
   - Follow with supporting explanations
   - End with a brief summary or key takeaway

Context from Learning Materials:
{context}

Previous Questions Asked:
{history}

Current Question: {question}

Remember to keep your answer clear, engaging, and appropriate for a student's understanding level.

Answer:
"""

# Create a prompt template object
custom_rag_prompt = ChatPromptTemplate.from_template(template)


def get_answer(question: str, context: List[Document], history: List[str]) -> str:
    """
    Generate an answer based on the context and question history.
    
    Args:
        question: The question to answer
        context: List of relevant documents
        history: List of previous questions
        
    Returns:
        The generated answer
    """
    prompt_text = custom_rag_prompt.format(
        history=history,
        context="\n".join([doc.page_content for doc in context]),
        question=question
    )

    messages = [
        {"role": "system", "content": prompt_text}
    ]

    try:
        response = llm.invoke(messages)
        if response and hasattr(response, 'content'):
            return response.content
        else:
            return "Sorry, no valid answer generated."
    except Exception as e:
        logging.error(f"Error generating answer: {str(e)}")
        return "Sorry, I couldn't process your request at this moment!!."




def check_question_and_reframe_it(question: str, history: List[str]) -> Dict:
    """
    Evaluate and potentially reframe a question based on its clarity and context.
    
    Args:
        question: The question to evaluate
        history: List of previous questions
        
    Returns:
        Dictionary containing evaluation results
    """
    logging.info(f"Evaluating question: '{question}' with history: {history}")
    check_prompt_template = """
    You are an intelligent assistant tasked with evaluating the clarity and completeness of questions.
    
    1. If the provided question is clear, specific, and does not need additional context:
       - Return: {{"is_complete": "yes", "corrected_question": "{question}"}}
    
    2. If the question is vague but there is sufficient history to reframe it:
       - Use the history to improve the question and return it:
         {{"is_complete": "yes", "corrected_question": ?}}
    
    3. If the question is vague and there is no sufficient history to reframe it:
       - Return: {{"is_complete": "no", "corrected_question": null}}

    Question: "{question}"
    History of questions: "{history}"
    """

    prompt = check_prompt_template.format(question=question, history=history or "None")
    messages = [{"role": "system", "content": prompt}]
    
    try:
        response = llm.invoke(messages)
        return json.loads(response.content)
    except Exception as e:
        print(f"Error evaluating question: {str(e)}")
        return {"is_complete": "no", "corrected_question": None}


#-------------------------- ROUTES  ----------------------
@app.route('/qa', methods=['POST'])
def qa_page():
    try:
        # Parse JSON data from the request
        data = request.get_json()
        
        if not data:
            return jsonify({'message': 'No data received'}), 400
        
        db = data.get('db')
        query = data.get('query')
        student_id = data.get('student_id')
        
        if not all([db, query, student_id]):
            return jsonify({'message': 'Invalid input parameters'}), 400

        # Fetch question history
        question_history = get_question_history(student_id)
        
        # Evaluate and potentially reframe the question
        evaluation_result = check_question_and_reframe_it(query, question_history)
        
        if evaluation_result["is_complete"] == "no":
            return jsonify({"message": "The question is too vague to answer. Please provide more context."}), 400
        
        # Use the corrected question, if available
        corrected_query = evaluation_result.get("corrected_question", query)
        
        # Retrieve relevant context from the vector database
        context_documents = get_context(db, corrected_query)
        
        # Generate the answer
        answer = get_answer(corrected_query, context_documents, question_history)
        
        return jsonify({"answer": answer}), 200

    except Exception as e:
        logging.error(f"Error in /qa route: {traceback.format_exc()}")
        return jsonify({"message": "An error occurred while processing your request."}), 500

@app.route("/")
def home():
    return "Welcome to Totoys Backend!!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)