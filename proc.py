from fpdf import FPDF

# Create PDF object
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Set font for Title
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 20, "PROCURAÇÃO PARTICULAR", ln=True, align='C')

# Set font for Body
pdf.set_font("Arial", size=12)

# Body Text
# Note: Using Latin-1 encoding logic for Portuguese characters in standard FPDF fonts
text_body = (
    "Eu, LEONARDO ANTONIO LUGARINI, portador da Cédula de Identidade (RG) nº 123502388, "
    "nomeio e constituo como minha bastante procuradora a Sra. FLAVIELI GARZARO LUGARINI, "
    "portadora da Cédula de Identidade (RG) nº 60265860, com poderes específicos "
    "para retirar meu passaporte italiano junto ao Consulado Geral da Itália em Curitiba.\n\n"
    "Declaro expressamente, conforme exigência consular, que o Consulado Geral da Itália em "
    "Curitiba está isento de qualquer responsabilidade quanto ao transporte ou eventual "
    "extravio do passaporte entregue à procuradora acima qualificada."
)

# Fix encoding for Portuguese characters (latin-1 is standard for FPDF core fonts)
text_body = text_body.encode('latin-1', 'replace').decode('latin-1')

pdf.multi_cell(0, 10, text_body)

# Date and Location
pdf.ln(20)
date_text = "Curitiba - PR, 19 de novembro de 2025."
pdf.cell(0, 10, date_text, ln=True, align='R')

# Signature
pdf.ln(30)
pdf.cell(0, 10, "_" * 50, ln=True, align='C')
pdf.cell(0, 10, "LEONARDO ANTONIO LUGARINI", ln=True, align='C')
pdf.set_font("Arial", size=10)
pdf.cell(0, 5, "Outorgante", ln=True, align='C')

# Output the PDF
file_path = "Procuracao_Consulado_Curitiba.pdf"
pdf.output(file_path)

