from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pdf_rag import initialize_processor, PDFProcessor

app = FastAPI()

# Configuration CORS pour permettre les requêtes du frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À remplacer par l'URL de votre frontend en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialiser le processeur au démarrage
processor: PDFProcessor = None

@app.on_event("startup")
async def startup_event():
    global processor
    processor = initialize_processor()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat", response_model=List[str])
async def chat(request: QueryRequest):
    """
    Endpoint pour le chat qui reçoit une question et retourne une liste de réponses.
    La réponse est divisée en plusieurs parties pour un affichage progressif.
    """
    if not processor:
        raise HTTPException(status_code=500, detail="Le processeur n'est pas initialisé")
    
    try:
        responses = processor.generate_response(request.query)
        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Endpoint de vérification de santé de l'API"""
    return {"status": "healthy", "processor_initialized": processor is not None}
