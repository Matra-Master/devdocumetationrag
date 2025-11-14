"""
API REST para el sistema RAG con FastAPI.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
import sys

# Añadir el directorio padre al path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from vector_database import VectorDatabase
from gemini_client import GeminiClient
#from document_processor import DocumentProcessor
from config.settings import Config


# Modelos Pydantic para requests/responses

class QueryRequest(BaseModel):
    question: str
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    follow_up_questions: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    database_status: Dict[str, Any]
    gemini_status: Dict[str, Any]

class DocumentInfo(BaseModel):
    collection_name: str
    total_documents: int
    embedding_model: str


# Inicializar FastAPI
app = FastAPI(
    title="RAG Documentation System",
    description="Sistema de documentación basado en RAG con Gemini y ChromaDB",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales para los servicios
vector_db = None
gemini_client = None
document_processor = None


@app.on_event("startup")
async def startup_event():
    """Inicializa los servicios al arrancar la aplicación."""
    global vector_db, gemini_client, document_processor
    
    try:
        print("Inicializando servicios...")
        
        # Inicializar servicios
        #vector_db = VectorDatabase()
        gemini_client = GeminiClient()
        #document_processor = DocumentProcessor()
        
        print("✅ Servicios inicializados correctamente")
        
    except Exception as e:
        print(f"❌ Error al inicializar servicios: {e}")
        raise


@app.get("/", response_class=HTMLResponse)
async def root():
    """Página de inicio con información básica."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Documentation System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #fff; padding: 3px 8px; border-radius: 3px; font-weight: bold; }
            .get { background: #61affe; }
            .post { background: #49cc90; }
            code { background: #f8f8f8; padding: 2px 4px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 RAG Documentation System</h1>
            <p>Sistema de documentación basado en RAG (Retrieval-Augmented Generation) con Gemini y ChromaDB.</p>
            
            <h2>📚 Endpoints Disponibles</h2>
            
            <div class="endpoint">
                <span class="method get">GET</span> <code>/health</code>
                <p>Verifica el estado del sistema y sus componentes.</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span> <code>/info</code>
                <p>Información sobre la base de datos de documentos.</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span> <code>/query</code>
                <p>Realiza consultas sobre los documentos LLMs.</p>
                <p><strong>Body:</strong> <code>{"question": "tu pregunta aquí"}</code></p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span> <code>/reload-documents</code>
                <p>Recarga los documentos desde el archivo llms.txt.</p>
            </div>
            
            <h2>🚀 Uso Rápido</h2>
            <p>Ejemplo de consulta:</p>
            <pre><code>curl -X POST "http://localhost:8000/query" \\
     -H "Content-Type: application/json" \\
     -d '{"question": "¿Qué son los LLMs?"}'</code></pre>
            
            <h2>📖 Documentación</h2>
            <p>Visita <a href="/docs">/docs</a> para la documentación interactiva de Swagger.</p>
        </div>
    </body>
    </html>
    """
    return html_content


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verifica el estado de salud del sistema."""
    try:
        # Verificar base de datos vectorial
        db_info = {'name':'TEST', 'count': 0} #vector_db.get_collection_info()
        db_status = {
            "connected": True,
            "collection": db_info["name"],
            "documents": db_info["count"]
        }
        
        # Verificar Gemini
        gemini_status = gemini_client.test_connection()
        
        overall_status = "healthy" if db_status["connected"] and gemini_status["success"] else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            message="Sistema funcionando correctamente" if overall_status == "healthy" else "Algunos servicios tienen problemas",
            database_status=db_status,
            gemini_status=gemini_status
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en verificación de salud: {str(e)}")


@app.get("/info", response_model=DocumentInfo)
async def get_info():
    """Obtiene información sobre la base de datos de documentos."""
    try:
        info = {'name':'TEST', 'count': 0, 'embedding_model': 'all-MiniLM-L6-v2----test'} #vector_db.get_collection_info()
        return DocumentInfo(
            collection_name=info["name"],
            total_documents=info["count"],
            embedding_model=info["embedding_model"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener información: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Realiza una consulta sobre los documentos cargados."""
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")
        
        # Buscar documentos relevantes
        search_results = vector_db.search_similar_documents(
            query=question, 
            n_results=request.max_results
        )
        
        # Preparar documentos de contexto para Gemini
        context_documents = search_results.get("documents", [])
        
        # Generar respuesta con Gemini
        answer = gemini_client.generate_response(question, context_documents)
        
        # Preparar información de fuentes
        sources = []
        for i, (doc, metadata, distance) in enumerate(zip(
            search_results.get("documents", []),
            search_results.get("metadatas", []),
            search_results.get("distances", [])
        )):
            sources.append({
                "rank": i + 1,
                "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                "source": metadata.get("source", "unknown"),
                "chunk_id": metadata.get("chunk_id", 0),
                "similarity_score": float(1 - distance),  # Convertir distancia a similarity
                "tokens": metadata.get("tokens", 0)
            })
        
        # Generar preguntas de seguimiento
        follow_up_questions = gemini_client.suggest_follow_up_questions(question, context_documents)
        
        return QueryResponse(
            question=question,
            answer=answer,
            sources=sources,
            follow_up_questions=follow_up_questions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar consulta: {str(e)}")


@app.post("/reload-documents")
async def reload_documents():
    """Recarga los documentos desde el archivo llms.txt."""
    try:
        # Verificar que el archivo existe
        if not os.path.exists(Config.LLMS_FILE_PATH):
            raise HTTPException(
                status_code=404, 
                detail=f"Archivo {Config.LLMS_FILE_PATH} no encontrado"
            )
        
        # Limpiar colección actual
        vector_db.clear_collection()
        
        # Procesar documentos
        chunks = document_processor.process_llms_file()
        
        # Añadir a la base de datos vectorial
        success = vector_db.add_documents(chunks)
        
        if not success:
            raise HTTPException(status_code=500, detail="Error al cargar documentos en la base de datos")
        
        # Obtener información actualizada
        info = vector_db.get_collection_info()
        
        return {
            "message": "Documentos recargados exitosamente",
            "total_chunks": len(chunks),
            "collection_documents": info["count"],
            "file_path": Config.LLMS_FILE_PATH
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al recargar documentos: {str(e)}")


@app.get("/search")
async def search_documents(
    q: str = Query(..., description="Consulta de búsqueda"),
    limit: int = Query(5, description="Número máximo de resultados", ge=1, le=20)
):
    """Búsqueda simple en los documentos sin generar respuesta."""
    try:
        if not q.strip():
            raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")
        
        # Buscar documentos
        results = vector_db.search_similar_documents(query=q, n_results=limit)
        
        # Formatear resultados
        formatted_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results.get("documents", []),
            results.get("metadatas", []),
            results.get("distances", [])
        )):
            formatted_results.append({
                "rank": i + 1,
                "content": doc,
                "metadata": metadata,
                "similarity_score": float(1 - distance)
            })
        
        return {
            "query": q,
            "total_results": len(formatted_results),
            "results": formatted_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")


if __name__ == "__main__":
    print("🚀 Iniciando servidor RAG Documentation System...")
    print(f"📍 URL: http://{Config.HOST}:{Config.PORT}")
    print(f"📚 Documentación: http://{Config.HOST}:{Config.PORT}/docs")
    
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG
    )