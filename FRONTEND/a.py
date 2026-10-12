import os
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from docx import Document
import PyPDF2

app = Flask(__name__)

# Load the BART model and tokenizer
model_path = "summarization_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Utility functions to extract text from files
def extract_text_from_docx(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@app.route('/summarize', methods=["GET", "POST"])
def summarizer():
    if request.method == "POST":
        text = request.form.get("text", "")
        file = request.files.get("file")
        summary_ratio = request.form.get("percentage", "50")  # Default to 50%

        try:
            summary_ratio = float(summary_ratio) / 100  # Convert to a decimal
            if summary_ratio <= 0 or summary_ratio > 1:
                raise ValueError("Summary ratio must be between 0 and 100.")
        except ValueError as e:
            return render_template('a.html', error="Invalid percentage value.", summary="")

        # If a file is uploaded, extract text
        if file:
            filename = file.filename
            if filename.endswith(".docx"):
                text = extract_text_from_docx(file)
            elif filename.endswith(".pdf"):
                text = extract_text_from_pdf(file)
            elif filename.endswith(".txt"):
                text = file.read().decode("utf-8")
            else:
                return render_template('a.html', error="Unsupported file type.", summary="")
        
        if not text:
            return render_template('a.html', error="No text or file content provided.", summary="")

        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

        # Calculate lengths based on input text
        input_length = len(tokenizer.tokenize(text))
        max_length = int(input_length * summary_ratio)  # Maximum length based on the desired ratio
        min_length = max(30, int(max_length * 0.5))  # Set a minimum length that is at least 30 tokens

        # Generate the summary
        summary_ids = model.generate(
            inputs["input_ids"], 
            max_length=max_length, 
            min_length=min_length, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )

        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Pass the summary to the template
        return render_template('a.html', summary=summary, error="")

    return render_template('a.html', summary="", error="")

if __name__ == "__main__":
    app.run(debug=True)
