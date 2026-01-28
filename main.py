from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
from transformers import pipeline
import uvicorn
import io
import re
from fpdf import FPDF
from fastapi.responses import StreamingResponse

app = FastAPI(title="Legal Audit API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AUDIT LOGIC (Unchanged) ---
classifier = None
summarizer = None

def get_models():
    global classifier, summarizer
    if classifier is None:
        print("Loading Zero-Shot Classifier...")
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    if summarizer is None:
        print("Loading Summarizer...")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    return classifier, summarizer

@app.on_event("startup")
async def startup_event():
    get_models()

def extract_text(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            extract = page.extract_text()
            if extract:
                text += extract + "\n"
    return text

def get_tip(label):
    tips = {
        "Adversarial/Risky Trap": "Request specific definition or removal of ambiguous terms.",
        "Unfair Arbitration": "Propose mutual arbitration or a neutral venue (e.g., AAA rules).",
        "Hidden Liability": "Explicitly request mutual liability or a cap on damages."
    }
    return tips.get(label, "Review carefully.")

@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    contents = await file.read()
    text = extract_text(contents)
    
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text")

    clauses = [c.strip() for c in text.replace('\n', '.').split('.') if len(c.strip()) > 20]
    clauses = clauses[:30] 
    
    classifier_pipe, summarizer_pipe = get_models()
    
    labels = ["Adversarial/Risky Trap", "Unfair Arbitration", "Hidden Liability", "Standard Clause", "Beneficial Clause"]
    risk_labels = ["Adversarial/Risky Trap", "Unfair Arbitration", "Hidden Liability"]
    
    risks = []
    safe_points = []
    
    for clause in clauses:
        res = classifier_pipe(clause, labels)
        top = res['labels'][0]
        score = res['scores'][0]
        
        if top in risk_labels and score > 0.4:
            risks.append({
                "clause": clause,
                "label": top,
                "score": score,
                "tip": get_tip(top)
            })
        elif score > 0.5:
             safe_points.append(clause)
             
    summary_text = ""
    try:
        trunc = text[:3000]
        summ = summarizer_pipe(trunc, max_length=130, min_length=30, do_sample=False)
        summary_text = summ[0]['summary_text']
    except Exception as e:
        print(f"Summary failed: {e}")
        summary_text = "Analysis complete, but summary generation timed out."

    return {
        "risk_density": len(risks) / len(clauses) if clauses else 0,
        "risks": risks,
        "safe_points": safe_points[:10],
        "summary": summary_text
    }

# --- IMPROVED CONTRACT GENERATION FEATURE ---
class ContractParser:
    @staticmethod
    def extract_details(prompt):
        # Default values ensure the document is never blank
        details = {
            "BUYER": "[PARTY A NAME]",
            "SELLER": "[PARTY B NAME]",
            "BUYER_REP": "Authorized Signatory",
            "SELLER_REP": "Authorized Signatory",
            "AMOUNT": "[AGREED AMOUNT]",
            "JURISDICTION": "the laws of India",
            "ARBITRATION_CITY": "New Delhi",
            "TRANCHE_1": "50%",
            "TRANCHE_2": "25%",
            "TRANCHE_3": "25%",
            "CLOSING_DATE": "the Effective Date",
            "NON_COMPETE_YEARS": "2",
            "EMPLOYEE_RETENTION_YEARS": "2",
            "SCOPE": prompt # We inject the raw prompt as the 'Scope'
        }
        
        # 1. Advanced Regex Extraction
        patterns = {
            "BUYER": [r"Buyer:\s*([^,\n]+)", r"between\s+([^,\n]+)\s+and"],
            "SELLER": [r"Seller:\s*([^,\n]+)", r"and\s+([^,\n]+)"],
            "AMOUNT": [r"valued at\s*([^,\n]+)", r"price of\s*([^,\n]+)", r"rent of\s*([^,\n]+)"],
            "JURISDICTION": [r"laws of\s*([^,\n\.]+)", r"governed by\s*([^,\n\.]+)"],
            "ARBITRATION_CITY": [r"arbitration in\s*([^,\n\.]+)"],
        }

        # Try to find matches
        for key, regex_list in patterns.items():
            for pattern in regex_list:
                match = re.search(pattern, prompt, re.IGNORECASE)
                if match:
                    details[key] = match.group(1).strip()
                    break # Stop looking for this key if found
        
        return details

class ContractGenerator:
    @staticmethod
    def create_pdf(title, body):
        pdf = FPDF()
        pdf.add_page()
        
        # Title Page Logic
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 15, title.upper(), ln=True, align='C')
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 10, "Generated via Neural Legal Audit System", ln=True, align='C')
        pdf.line(10, 35, 200, 35)
        pdf.ln(10)
        
        # Body Logic
        pdf.set_font("Arial", size=10)
        
        # Clean Text
        clean_body = body.replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("₹", "INR ")
        try:
            clean_body = clean_body.encode('latin-1', 'replace').decode('latin-1')
        except:
            pass
            
        pdf.multi_cell(0, 6, clean_body)
        
        # Separate Signature Page
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "EXECUTION PAGE", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 5, "IN WITNESS WHEREOF, the Parties have executed this Agreement as of the date first written above.")
        pdf.ln(20)
        
        # Signature Blocks
        y = pdf.get_y()
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(90, 5, "SIGNED for PARTY A:", ln=0)
        pdf.cell(90, 5, "SIGNED for PARTY B:", ln=1)
        pdf.ln(15)
        pdf.cell(90, 5, "_______________________", ln=0)
        pdf.cell(90, 5, "_______________________", ln=1)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(90, 5, "Authorized Signatory", ln=0)
        pdf.cell(90, 5, "Authorized Signatory", ln=1)
        
        return pdf

    # --- THE MASTER LAWYER TEMPLATES ---
    # These templates ensure "Many Pages" regardless of simple input
    
    @staticmethod
    def get_spa_template(d):
        return f"""1. PREAMBLE
This SHARE PURCHASE AGREEMENT (the "Agreement") is made and entered into at {d['ARBITRATION_CITY']}, India.

BY AND BETWEEN:
{d['BUYER']} (hereinafter "Buyer"), AND {d['SELLER']} (hereinafter "Seller").

2. RECITALS
WHEREAS, the Seller owns 100% of the share capital of the entity; and
WHEREAS, the Buyer desires to acquire said shares for a total consideration of {d['AMOUNT']}.

3. CONSIDERATION
The total consideration shall be paid as follows:
3.1 Tranche 1: {d['TRANCHE_1']} upon Closing.
3.2 Tranche 2: {d['TRANCHE_2']} upon IP Transfer.
3.3 Tranche 3: {d['TRANCHE_3']} held in Escrow for 12 months.

4. COVENANTS
4.1 Non-Compete: Seller shall not compete for a period of {d['NON_COMPETE_YEARS']} years.
4.2 Employee Retention: Key employees to be retained for {d['EMPLOYEE_RETENTION_YEARS']} years.

5. REPRESENTATIONS AND WARRANTIES
The Seller represents that they have full legal right, power, and authority to enter into this Agreement.

6. INDEMNIFICATION
The Seller agrees to indemnify the Buyer against any losses arising from breach of warranties.

7. GOVERNING LAW
This Agreement is governed by {d['JURISDICTION']}."""

    @staticmethod
    def get_generic_master_template(d, doc_type="SERVICE AGREEMENT"):
        # This is the "Master Structure" that guarantees length
        return f"""1. PARTIES AND EFFECTIVE DATE
This {doc_type} (the "Agreement") is entered into on this day by and between:

PARTY A: {d['BUYER']}
Address: [ Registered Address of Party A ]

AND

PARTY B: {d['SELLER']}
Address: [ Registered Address of Party B ]

2. RECITALS
WHEREAS, Party A requires certain services/goods as described herein; and
WHEREAS, Party B represents that it has the requisite skills and capacity to provide such services/goods.

3. SCOPE OF AGREEMENT
The Parties agree to the following scope based on the User Request:
"{d['SCOPE']}"

Party B shall perform these duties with the highest standard of professional care and skill.

4. FINANCIAL TERMS
4.1 Fees: The total value of this agreement is {d['AMOUNT']}.
4.2 Invoicing: Party B shall invoice Party A on a monthly basis.
4.3 Taxes: All payments are exclusive of GST, which shall be charged additionally if applicable.

5. TERM AND TERMINATION
5.1 Term: This Agreement shall commence on the Effective Date and continue for 12 months unless terminated earlier.
5.2 Termination for Cause: Either Party may terminate this Agreement immediately upon written notice if the other Party commits a material breach.
5.3 Termination for Convenience: Party A may terminate this Agreement by providing 30 days' written notice.

6. CONFIDENTIALITY
6.1 Definition: "Confidential Information" means all non-public information disclosed by one Party to the other.
6.2 Obligations: The Receiving Party agrees to hold all Confidential Information in strict confidence and not to disclose it to third parties without prior written consent.
6.3 Duration: These obligations shall survive the termination of this Agreement for a period of 5 years.

7. INDEMNIFICATION
Party B agrees to indemnify, defend, and hold harmless Party A from and against any and all claims, damages, liabilities, costs, and expenses arising out of Party B's negligence or misconduct.

8. LIMITATION OF LIABILITY
Neither Party shall be liable to the other for any indirect, special, or consequential damages arising out of this Agreement. The total liability of either Party shall not exceed the total fees paid under this Agreement.

9. GOVERNING LAW AND DISPUTE RESOLUTION
9.1 Governing Law: This Agreement shall be governed by {d['JURISDICTION']}.
9.2 Arbitration: Any dispute arising under this Agreement shall be resolved by arbitration in {d['ARBITRATION_CITY']}.

10. GENERAL PROVISIONS
10.1 Severability: If any provision is found to be invalid, the remaining provisions shall remain in full force.
10.2 Entire Agreement: This Agreement constitutes the entire understanding between the Parties.
10.3 Force Majeure: Neither Party shall be liable for delays caused by events beyond their reasonable control."""

    @staticmethod
    def get_template(prompt):
        prompt_lower = prompt.lower()
        details = ContractParser.extract_details(prompt)
        
        # Route to the correct "Master Template"
        if "share purchase" in prompt_lower or "acquisition" in prompt_lower:
            return "SHARE PURCHASE AGREEMENT", ContractGenerator.get_spa_template(details)
        elif "nda" in prompt_lower or "non-disclosure" in prompt_lower:
            return "NON-DISCLOSURE AGREEMENT", ContractGenerator.get_generic_master_template(details, "NON-DISCLOSURE AGREEMENT")
        elif "lease" in prompt_lower or "rent" in prompt_lower:
            return "RESIDENTIAL LEASE AGREEMENT", ContractGenerator.get_generic_master_template(details, "LEASE AGREEMENT")
        elif "employment" in prompt_lower:
            return "EMPLOYMENT CONTRACT", ContractGenerator.get_generic_master_template(details, "EMPLOYMENT CONTRACT")
        else:
            return "GENERAL SERVICE AGREEMENT", ContractGenerator.get_generic_master_template(details, "SERVICE AGREEMENT")

@app.post("/generate")
async def generate_contract(payload: dict):
    prompt = payload.get("prompt", "")
    title, body = ContractGenerator.get_template(prompt)
    pdf = ContractGenerator.create_pdf(title, body)
    
    buf = io.BytesIO()
    pdf_content = pdf.output(dest='S').encode('latin-1') 
    buf.write(pdf_content)
    buf.seek(0)
    
    return StreamingResponse(
        buf, 
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={title.replace(' ', '_')}.pdf"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)