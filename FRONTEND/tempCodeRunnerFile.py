from flask import Flask,render_template,redirect,request,url_for, send_file, session
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, SimpleRNN
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import joblib
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from docx import Document
from PyPDF2 import PdfReader
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from transformers import AutoTokenizer, BartForConditionalGeneration
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from PyPDF2 import PdfReader

app = Flask(__name__)
app.secret_key = 'your_secret_key'


mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='AIteacher'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')











import pdfplumber
from docx import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the BART model and tokenizer
model_path = "summarization_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model_1 = AutoModelForSeq2SeqLM.from_pretrained(model_path)


@app.route('/algorithm_selection', methods=["GET", "POST"])
def algorithm_selection():
    error = None  # Initialize the error variable
    summary = None  # Initialize the summary variable

    if request.method == "POST":
        # Get form data
        input_text = request.form.get("input_text", "").strip()
        file = request.files.get("file")
        summary_ratio = request.form.get("summary_ratio", "30")

        # Validate inputs
        if not input_text and not file:
            error = "Please provide text or upload a file."
            return render_template('algorithm_selection.html', error=error)

        # If a file is uploaded, process it
        if file:
            filename = file.filename
            if filename.endswith('.pdf'):
                try:
                    with pdfplumber.open(file) as pdf:
                        input_text = ''.join(page.extract_text() for page in pdf.pages if page.extract_text())
                except Exception:
                    error = "Failed to process the PDF file. Please upload a valid PDF."
                    return render_template('algorithm_selection.html', error=error)
            elif filename.endswith('.docx'):
                try:
                    doc = Document(file)
                    input_text = '\n'.join(para.text for para in doc.paragraphs)
                except Exception:
                    error = "Failed to process the Word document. Please upload a valid .docx file."
                    return render_template('algorithm_selection.html', error=error)
            else:
                error = "Unsupported file format. Please upload a PDF or Word document."
                return render_template('algorithm_selection.html', error=error)

        # If input text is provided, use it
        if not input_text:
            error = "No text could be extracted from the file or provided input."
            return render_template('algorithm_selection.html', error=error)

        try:
            # Tokenize the input text
            inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

            # Calculate lengths for summarization
            input_length = len(tokenizer.tokenize(input_text))
            summary_ratio = int(summary_ratio) / 100
            max_length = max(30, int(input_length * summary_ratio))  # Minimum max_length is 30 tokens
            min_length = max(30, int(max_length * 0.5))  # Ensure min_length is at least 30 tokens

            # Generate the summary
            summary_ids = model_1.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

            # Decode the summary
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            error = f"An error occurred during summarization: {str(e)}"
            return render_template('algorithm_selection.html', error=error)

        return render_template('algorithm_selection.html', summary=summary, error=None)

    return render_template('algorithm_selection.html', error=error)




app.secret_key = 'your_secret_key'

# Set up Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Assuming genai is configured properly
import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-pro')

def generate_mcqs(text, number, tone):
    prompt = f"""
    Text: {text}
    
    You are an expert MCQ maker. Given the above text, create a quiz of {number} multiple choice questions in {tone} tone.
    Format like the example:
    
    1: Question?
    a) Option 1
    b) Option 2
    c) Option 3
    d) Option 4
    Correct answer: x
    """
    
    for attempt in range(3):  # Retry up to 3 times
        try:
            response = model.generate_content(prompt)
            quiz = response.text
            
            # Parse generated quiz
            mcqs = quiz.split('\n\n')
            quiz_data = []
            answer_key = {}
            for mcq in mcqs:
                lines = mcq.split('\n')
                if len(lines) < 5:
                    continue
                question = lines[0].strip()
                options = {chr(ord('a') + i): line[3:].strip() for i, line in enumerate(lines[1:5])}
                correct = lines[5][14:].strip()
                quiz_data.append({"Question": question, "Options": options})
                answer_key[question] = correct
                
            return quiz_data, answer_key
        
        except Exception as e:
            if "500" in str(e):
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None, None
    
    return None, None

def get_webpage_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        text = None
        file = request.files["file"]
        webpage_url = request.form["webpage_url"]
        txt = request.form["txt"]
        number_of_questions = int(request.form["number_of_questions"])
        tone = request.form["tone"]

        
        # File or URL or Text Area content retrieval
        if file:
            text = file.read().decode("utf-8")
        elif webpage_url:
            text = get_webpage_content(webpage_url)
        elif txt:
            text = txt

        # Generate MCQs
        quiz_data, answer_key = generate_mcqs(text, number_of_questions, tone)
        
        if quiz_data:
            session['quiz_data'] = quiz_data
            session['answer_key'] = answer_key
            return render_template("prediction.html", quiz_data=quiz_data)
        else:
            flash("Failed to generate MCQs.")
            return redirect(url_for("index"))

    return render_template("prediction.html")




@app.route("/download/<format>")
def download(format):
    quiz_data = session.get("quiz_data", [])
    answer_key = session.get("answer_key", {})

    if format == "pdf":
        pdf_file = SimpleDocTemplate('mcqs.pdf', pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        for mcq in quiz_data:
            elements.append(Paragraph(mcq['Question'], styles['Heading1']))
            for option, text in mcq['Options'].items():
                elements.append(Paragraph(f"{option}) {text}", styles['Normal']))
            elements.append(Paragraph("", styles['Normal']))

        pdf_file.build(elements)
        return send_file("mcqs.pdf", as_attachment=True)

    elif format == "word":
        document = Document()
        for mcq in quiz_data:
            document.add_heading(mcq['Question'], level=1)
            for option, text in mcq['Options'].items():
                document.add_paragraph(f"{option}) {text}")
            document.add_paragraph()

        document.save("mcqs.docx")
        return send_file("mcqs.docx", as_attachment=True)

    elif format == "answer_key":
        answer_key_doc = Document()
        answer_key_doc.add_heading("Answer Key", level=1)
        for question, answer in answer_key.items():
            answer_key_doc.add_paragraph(f"{question} - Correct answer: {answer}")

        answer_key_doc.save("answer_key.docx")
        return send_file("answer_key.docx", as_attachment=True)

    return redirect(url_for("results"))



if __name__ == '__main__':
    app.run(debug = True)