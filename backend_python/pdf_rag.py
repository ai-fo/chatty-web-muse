import os
import base64
import requests
import numpy as np
import faiss
from typing import List, Dict, Tuple
from pathlib import Path
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torch
from dataclasses import dataclass
import uuid

@dataclass
class TextChunk:
    text: str  # Contenu du chunk
    source: str  # Nom du fichier source
    page: int  # Numéro de page
    images: List[Image.Image]  # Images associées au chunk

import random
import re

@dataclass
class Message:
    role: str  # 'user' ou 'assistant'
    content: str  # Contenu du message

class ResponseSplitter:
    @staticmethod
    def split_into_chunks(text: str, min_chunks: int = 2, max_chunks: int = 3) -> List[str]:
        """Décompose une réponse en plusieurs parties cohérentes."""
        # Séparer d'abord par les sauts de ligne doubles
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        if not paragraphs:
            return [text]
        
        # Si on a moins de paragraphes que min_chunks
        if len(paragraphs) < min_chunks:
            # Essayer de diviser les paragraphes longs par les points
            new_paragraphs = []
            for p in paragraphs:
                sentences = [s.strip() + '.' for s in p.split('.') if s.strip()]
                new_paragraphs.extend(sentences)
            paragraphs = new_paragraphs
        
        # Déterminer le nombre final de chunks
        n_chunks = random.randint(min_chunks, min(max_chunks, len(paragraphs)))
        
        # Si on a trop de paragraphes, les regrouper
        if len(paragraphs) > n_chunks:
            # Calculer la taille moyenne des chunks
            chunk_size = len(paragraphs) // n_chunks
            chunks = []
            current_chunk = []
            current_size = 0
            
            for p in paragraphs:
                current_chunk.append(p)
                current_size += 1
                
                if current_size >= chunk_size and len(chunks) < n_chunks - 1:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
            
            # Ajouter le dernier chunk
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            
            return chunks
        
        # Si on a le bon nombre ou pas assez de paragraphes
        return paragraphs

class Conversation:
    def __init__(self):
        self.messages: List[Message] = []
        self.max_context_length = 10  # Nombre maximum de messages à conserver

    def add_message(self, role: str, content: str):
        """Ajoute un message à l'historique de la conversation."""
        self.messages.append(Message(role=role, content=content))
        # Garder seulement les derniers messages si on dépasse max_context_length
        if len(self.messages) > self.max_context_length:
            self.messages = self.messages[-self.max_context_length:]

    def get_context(self) -> List[Dict[str, str]]:
        """Retourne l'historique de la conversation au format attendu par l'API."""
        return [{'role': msg.role, 'content': msg.content} for msg in self.messages]

@dataclass
class PDFContent:
    source: str
    pages: List[dict]  # Liste des pages avec leur texte et images
    total_text: str  # Texte complet du PDF
    chunks: List[TextChunk]  # Chunks de texte pour la recherche

from config import (PIXTRAL_URL, MISTRAL_URL, MODEL_PARAMS, SEARCH_PARAMS, 
                  TEMP_DIR, EMBEDDING_MODEL, PDF_FOLDER, PIXTRAL_PATH,
                  MISTRAL_PATH)

class PDFProcessor:
    def __init__(self, pdf_folder: str):
        self.conversation = Conversation()
        self.pdf_folder = Path(pdf_folder)
        self.temp_dir = Path(TEMP_DIR)
        self.temp_dir.mkdir(exist_ok=True)
        # Charger le tokenizer et le modèle
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL)
        self.index = None
        self.pdfs: List[PDFContent] = []
        
    def extract_from_pdf(self, pdf_path: str) -> PDFContent:
        """Extrait le texte et les images du PDF complet."""
        reader = PdfReader(pdf_path)
        pdf_images = convert_from_path(pdf_path)
        filename = Path(pdf_path).name
        
        pages = []
        full_text = []
        
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()
            full_text.append(text)
            
            pages.append({
                'number': page_num + 1,
                'text': text,
                'images': [pdf_images[page_num]]
            })
        
        return PDFContent(
            source=filename,
            pages=pages,
            total_text='\n'.join(full_text),
            chunks=[]  # Les chunks seront créés plus tard dans process_pdf_directory
        )

    def encode_image_base64(self, image: Image.Image) -> dict:
        """Encode une image en base64 pour l'API VLLM."""
        # Sauvegarder temporairement l'image
        temp_path = self.temp_dir / f"temp_{uuid.uuid4()}.jpg"
        image.save(temp_path, "JPEG")
        
        # Lire et encoder en base64
        with open(temp_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        
        # Nettoyer le fichier temporaire
        temp_path.unlink()
        
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded}"
            }
        }

    def analyze_image(self, image: Image.Image) -> dict:
        """Prépare une image pour l'API VLLM."""
        # Encoder directement l'image en base64 avec le bon format
        return self.encode_image_base64(image)

    def create_chunks(self, text: str, source: str, page: int, images: List[Image.Image]) -> List[TextChunk]:
        """Découpe un texte en chunks avec chevauchement."""
        chunks = []
        words = text.split()
        chunk_size = SEARCH_PARAMS['chunk_size']
        overlap = SEARCH_PARAMS['chunk_overlap']
        min_length = SEARCH_PARAMS['min_chunk_length']
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < min_length:  # Ignorer les chunks trop courts
                continue
            
            chunk = TextChunk(
                text=' '.join(chunk_words),
                source=source,
                page=page,
                images=images
            )
            chunks.append(chunk)
        
        return chunks
    
    def process_pdf_directory(self) -> None:
        """Traite tous les PDF dans le dossier et crée l'index de recherche."""
        self.pdfs = []
        all_chunks = []
        
        for filename in os.listdir(self.pdf_folder):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_folder, filename)
                
                # Extraire le contenu du PDF
                pdf_content = self.extract_from_pdf(pdf_path)
                
                # Créer les chunks pour chaque page
                pdf_chunks = []
                for page in pdf_content.pages:
                    page_chunks = self.create_chunks(
                        text=page['text'],
                        source=pdf_content.source,
                        page=page['number'],
                        images=page['images']
                    )
                    pdf_chunks.extend(page_chunks)
                
                # Stocker le contenu et les chunks
                pdf_content.chunks = pdf_chunks
                self.pdfs.append(pdf_content)
                all_chunks.extend(pdf_chunks)
        
        # Créer les embeddings avec le modèle
        embeddings = []
        batch_size = 32
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            texts = [f"passage: {chunk.text}" for chunk in batch]  # Format spécial pour E5
            
            inputs = self.tokenizer(texts, padding=True, truncation=True, 
                                   max_length=SEARCH_PARAMS['chunk_size'], 
                                   return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Utiliser le token [CLS] pour E5
                batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        # Convertir en array numpy
        embeddings = np.vstack(embeddings)
        
        # Normaliser les embeddings pour la similarité cosinus
        faiss.normalize_L2(embeddings)
        
        # Créer l'index FAISS
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Produit scalaire pour cosinus
        self.index.add(embeddings.astype(np.float32))
        
        # Sauvegarder les chunks pour la recherche
        self.chunks = all_chunks

    def search(self, query: str, k: int = SEARCH_PARAMS["max_results"]) -> List[Tuple[PDFContent, float]]:
        """Recherche les chunks les plus pertinents et retourne les PDFs correspondants."""
        # Encoder la requête avec le format E5
        inputs = self.tokenizer(f"query: {query}", padding=True, truncation=True, 
                              max_length=SEARCH_PARAMS['chunk_size'], 
                              return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            query_embedding = outputs.last_hidden_state[:, 0].cpu().numpy()  # Token [CLS]
        
        # Normaliser l'embedding de la requête
        faiss.normalize_L2(query_embedding)
        
        # Rechercher les chunks les plus pertinents
        D, I = self.index.search(query_embedding.astype(np.float32), k*3)
        
        # Collecter les PDFs uniques avec leurs meilleurs scores
        pdf_scores = {}
        for idx, score in zip(I[0], D[0]):
            if idx >= len(self.chunks):
                continue
                
            chunk = self.chunks[idx]
            current_score = pdf_scores.get(chunk.source, -float('inf'))
            if score > current_score:
                pdf_scores[chunk.source] = score
        
        # Récupérer les PDFs correspondants
        final_results = []
        for pdf in self.pdfs:
            if pdf.source in pdf_scores:
                final_results.append((pdf, pdf_scores[pdf.source]))
        
        # Trier par score (similarité cosinus)
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:k]

    def analyze_images_with_pixtral(self, query: str, images: list) -> str:
        """Analyse un lot d'images avec Pixtral."""
        user_content = [
            {
                "type": "text",
                "text": f"Voici des images extraites d'un document. Décris ce que tu vois dans ces images en lien avec la question : {query}"
            }
        ]
        
        # Ajouter les images au message
        for img in images:
            try:
                image_content = self.analyze_image(img)
                user_content.append(image_content)
            except Exception:
                # En cas d'erreur sur une image, on continue avec les autres
                continue
        
        # Appeler Pixtral
        response = requests.post(
            PIXTRAL_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": PIXTRAL_PATH,
                "messages": [
                    {
                        "role": "system",
                        "content": "Tu es un assistant expert en analyse d'images. Ta tâche est de décrire précisément le contenu des images en lien avec la question posée. Décris les éléments visuels importants, le texte visible, et tout ce qui pourrait aider à répondre à la question."
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                **MODEL_PARAMS
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return ""  # En cas d'erreur, on retourne une chaîne vide
    
    def check_question_relevance(self, query: str) -> bool:
        """Vérifie si la question nécessite une recherche dans les documents."""
        response = requests.post(
            MISTRAL_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": MISTRAL_PATH,
                "messages": [
                    {
                        "role": "system",
                        "content": "Tu es un assistant qui détermine si une question est pertinente pour le service. Réponds uniquement par 'OUI' ou 'NON'. Réponds 'OUI' si la question concerne : des procédures techniques, des informations sur les documents/services, des contacts/support, ou des horaires. Réponds 'NON' si la question est personnelle (ex: 'Comment vas-tu ?'), hors sujet, ou sans lien avec le service."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                **MODEL_PARAMS
            }
        )
        
        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"].strip().upper()
            return answer == "OUI"
        return True  # En cas d'erreur, on préfère chercher dans les documents

    def generate_response(self, query: str, k: int = 3) -> List[str]:
        """Génère une réponse à la question en utilisant les documents pertinents et le contexte de la conversation."""
        # Vérifier d'abord si la question nécessite une recherche dans les documents
        if not self.check_question_relevance(query):
            response = ["Désolé, je ne peux pas répondre aux questions personnelles ou hors sujet. "
                       "Je suis là pour vous aider avec les questions concernant nos services, procédures et support."]
            for part in response:
                self.conversation.add_message('assistant', part)
            return response

        # Récupérer les PDFs pertinents
        relevant_pdfs = self.search(query, k=k)
        
        # Préparer le contexte textuel
        context = ""
        all_images = []
        
        for pdf_content, score in relevant_pdfs:
            context += f"\nDocument: {pdf_content.source}\n"
            context += f"Contenu:\n{pdf_content.total_text}\n"
            
            # Collecter toutes les images
            for page in pdf_content.pages:
                images = page['images'][:SEARCH_PARAMS['max_images_per_page']]
                all_images.extend(images)
        
        # Analyser les images par lots de 4
        image_descriptions = []
        for i in range(0, len(all_images), SEARCH_PARAMS['max_total_images']):
            batch = all_images[i:i + SEARCH_PARAMS['max_total_images']]
            description = self.analyze_images_with_pixtral(query, batch)
            if description:
                image_descriptions.append(description)
        
        # Combiner toutes les informations
        full_context = f"Contexte textuel:\n{context}\n\nAnalyse des images:\n"
        full_context += "\n".join(image_descriptions)
        
        # Ajouter la question actuelle à l'historique
        self.conversation.add_message('user', query)
        
        # Préparer les messages pour l'API
        messages = [
            {
                "role": "system",
                "content": "Tu es un assistant expert qui répond aux questions en utilisant uniquement les informations fournies et le contexte de la conversation. Si tu ne trouves pas l'information dans le contexte, dis-le clairement."
            }
        ]
        
        # Ajouter l'historique de la conversation
        messages.extend(self.conversation.get_context())
        
        # Ajouter les informations du contexte documentaire
        messages.append({
            "role": "system",
            "content": f"Informations disponibles :\n{full_context}"
        })
        
        # Générer la réponse finale avec Mistral
        response = requests.post(
            MISTRAL_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": MISTRAL_PATH,
                "messages": messages,
                **MODEL_PARAMS
            }
        )
        
        if response.status_code == 200:
            response_content = response.json()["choices"][0]["message"]["content"]
            
            # Décomposer la réponse en plusieurs parties
            response_parts = ResponseSplitter.split_into_chunks(response_content)
            
            # Ajouter chaque partie de la réponse à l'historique
            for part in response_parts:
                self.conversation.add_message('assistant', part)
            
            return response_parts
        else:
            error_message = "Une erreur est survenue lors de la génération de la réponse."
            self.conversation.add_message('assistant', error_message)
            return [error_message]

def initialize_processor():
    """Initialise et retourne une instance du processeur PDF."""
    processor = PDFProcessor(pdf_folder=PDF_FOLDER)
    processor.process_pdf_directory()
    return processor

def process_query(processor: PDFProcessor, query: str) -> List[str]:
    """Traite une question unique et retourne une liste de réponses en maintenant le contexte de la conversation."""
    return processor.generate_response(query)

def main():
    """Initialise le processeur PDF."""
    return initialize_processor()

if __name__ == "__main__":
    main()
