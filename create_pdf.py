from reportlab.pdfgen import canvas

def create_contract(filename):
    c = canvas.Canvas(filename)
    c.drawString(100, 800, "SERVICE AGREEMENT")
    
    text = [
        "1. This is a standard service agreement between Party A and Party B.",
        "2. The provider shall not be liable for any damages, incidental or consequential, arising from the use of the service.",
        "3. All disputes shall be settled by binding arbitration in a location solely chosen by the provider.",
        "4. The user agrees to indemnify the provider against all claims.",
        "5. Payment is due within 30 days of invoice.",
        "6. This agreement is governed by the laws of the State of California."
    ]
    
    y = 750
    for line in text:
        c.drawString(50, y, line)
        y -= 20
        
    c.save()

if __name__ == "__main__":
    create_contract("test_contract.pdf")
