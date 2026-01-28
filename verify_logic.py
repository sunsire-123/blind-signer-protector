import pdfplumber
from transformers import pipeline
import sys

def test_logic():
    print("Loading models...")
    try:
        # Trying the requested model
        classifier = pipeline("zero-shot-classification", model="nlpaueb/legal-bert-base-uncased")
        print("Model nlpaueb/legal-bert-base-uncased loaded.")
    except Exception as e:
        print(f"Requested model failed: {e}")
        # Build fallback
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        print("Fallback model loaded.")

    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        print("Summarizer loaded.")
    except Exception as e:
        print(f"Summarizer failed: {e}")
        return

    print("Extracting text...")
    text = ""
    with pdfplumber.open("test_contract.pdf") as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    
    print(f"Extracted text length: {len(text)}")
    
    clauses = [c.strip() for c in text.replace('\n', '.').split('.') if len(c.strip()) > 20]
    print(f"Found {len(clauses)} clauses.")
    
    candidate_labels = ["Adversarial/Risky Trap", "Unfair Arbitration", "Hidden Liability", "Standard Clause"]
    
    print("Running classification on sample clause...")
    # Test a known bad clause
    bad_clause = "The provider shall not be liable for any damages, incidental or consequential, arising from the use of the service."
    result = classifier(bad_clause, candidate_labels)
    print(f"Clause: {bad_clause}")
    with open("verification_result.txt", "w") as f:
        f.write(f"Result: {result['labels'][0]} ({result['scores'][0]:.2f})\n")
    print(f"Result: {result['labels'][0]} ({result['scores'][0]:.2f})")
    
    # Check if results are meaningful (not random)
    # If the model is not trained for NLI, it usually dumps everything into one bucket or random.
    
    print("Running summary...")
    summary = summarizer(text[:1000])
    print("Summary generated successfully.")

if __name__ == "__main__":
    test_logic()
