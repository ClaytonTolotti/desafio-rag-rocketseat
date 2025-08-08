PDF_PATH = '../os-sertoes.pdf'
EMBEDDING_001_GEMINI = 'models/embedding-001'
DB_VECTOR_OS_SERTOES = 'os_sertoes'
MODEL_GEMINI_1_5_FLASH = 'gemini-1.5-flash'
TEMPLATE_PROMPT = """
    Você é um bibliotecário. Responda as perguntas baseadas no contexto fornecido.
    
    Context: {context}
    
    Pergunta: {input}
"""