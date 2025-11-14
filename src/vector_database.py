"""
BASE DE DATOS VECTORIAL 
========================

Este archivo maneja todo lo relacionado con ChromaDB. Básicamente es donde 
guardas los chunks de texto convertidos a números (embeddings) para después 
poder buscar cuáles son más similares a una pregunta del usuario.

QUÉ NECESITAS HACER:

1. CONECTAR A CHROMADB
   - Crear una conexión persistente (que guarde los datos en disco)
   - Configurar la carpeta donde se guardan los datos
   - Manejar si la base de datos ya existe o hay que crearla nueva

2. MODELO DE EMBEDDINGS
   - Usar sentence-transformers para convertir texto a números
   - El modelo recomendado es 'all-MiniLM-L6-v2' (funciona bien y es rápido) este se obtiene desde .env
   - Cada texto se convierte en un array de números que representa su "significado"

3. OPERACIONES BÁSICAS
   - AGREGAR: Meter chunks nuevos a la base de datos
   - BUSCAR: Encontrar chunks similares a una pregunta
   - ACTUALIZAR: Cambiar un chunk existente
   - BORRAR: Eliminar chunks o limpiar toda la colección

4. LA BÚSQUEDA (LO MÁS IMPORTANTE)
   - El usuario hace una pregunta
   - Convertir la pregunta a embedding
   - Buscar en la base de datos qué chunks tienen embeddings más similares
   - Devolver esos chunks ordenados por relevancia

5. GESTIÓN DE METADATOS
   - Junto con cada chunk guardar info extra: archivo, chunk_id, etc.
   - Esto sirve para después saber de dónde vino cada resultado

IMPORTANTE:

- CHROMA_DB_PATH: dónde guardar los datos (ej: "./chroma_db")
- CHROMA_COLLECTION_NAME: nombre de tu colección (ej: "llms_docs")  
- EMBEDDING_MODEL: qué modelo usar (ej: "all-MiniLM-L6-v2")

"""

import os
import sys
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Añadir el directorio padre al path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Config


class VectorDatabase:
    """
    Maneja la base de datos vectorial ChromaDB para almacenamiento y búsqueda semántica.
    """
    
    def __init__(self, db_path: str = "", collection_name: str = "", embedding_model: str = ""):
        """
        Inicializa la conexión a ChromaDB y el modelo de embeddings.
        
        Args:
            db_path: Ruta donde guardar la base de datos (default: Config.CHROMA_DB_PATH)
            collection_name: Nombre de la colección (default: Config.CHROMA_COLLECTION_NAME)
            embedding_model: Modelo de embeddings a usar (default: Config.EMBEDDING_MODEL)
        """
        self.db_path = db_path or Config.CHROMA_DB_PATH
        self.collection_name = collection_name or Config.CHROMA_COLLECTION_NAME
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL
        
        # Inicializar modelo de embeddings
        try:
            print(f"Cargando modelo de embeddings: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print(f"Modelo cargado correctamente")
        except Exception as e:
            raise Exception(f"Error al cargar modelo de embeddings: {str(e)}")
        
        # Inicializar ChromaDB
        try:
            print(f"Conectando a ChromaDB en: {self.db_path}")
            
            # Crear directorio si no existe
            os.makedirs(self.db_path, exist_ok=True)
            
            # Configurar ChromaDB con persistencia
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Obtener o crear colección
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"embedding_model": self.embedding_model_name}
            )
            
            print(f"Coleccion '{self.collection_name}' lista (documentos: {self.collection.count()})")
            
        except Exception as e:
            raise Exception(f"Error al conectar con ChromaDB: {str(e)}")
    
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos.
        
        Args:
            texts: Lista de textos a convertir
            
        Returns:
            Lista de embeddings (arrays de números)
        """
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            raise Exception(f"Error al generar embeddings: {str(e)}")
    
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Añade chunks de documentos a la base de datos vectorial.
        
        Args:
            chunks: Lista de diccionarios con 'content' y 'metadata'
            
        Returns:
            True si se añadieron exitosamente
        """
        try:
            if not chunks:
                print("No hay chunks para añadir")
                return False
            
            print(f"Procesando {len(chunks)} chunks...")
            
            # Extraer contenido y metadata
            documents = [chunk["content"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            ids = [chunk["metadata"]["id"] for chunk in chunks]
            
            # Generar embeddings
            print("Generando embeddings...")
            embeddings = self.generate_embeddings(documents)
            
            # Añadir a ChromaDB
            print("Añadiendo a ChromaDB...")
            self.collection.add(
                documents=documents,
                embeddings=embeddings,  # type: ignore
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Documentos añadidos exitosamente (total: {self.collection.count()})")
            return True
            
        except Exception as e:
            print(f"Error al añadir documentos: {e}")
            return False
    
    
    def search_similar_documents(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Busca documentos similares a una consulta.
        
        Args:
            query: Texto de búsqueda
            n_results: Número de resultados a devolver
            
        Returns:
            Diccionario con 'documents', 'metadatas', 'distances'
        """
        try:
            if not query.strip():
                return {"documents": [], "metadatas": [], "distances": []}
            
            # Generar embedding de la consulta
            query_embedding = self.generate_embeddings([query])[0]
            
            # Buscar en ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count())
            )
            
            # ChromaDB devuelve listas de listas, aplanar para facilitar uso
            return {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else []
            }
            
        except Exception as e:
            print(f"Error en búsqueda: {e}")
            return {"documents": [], "metadatas": [], "distances": []}
    
    
    def update_document(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Actualiza un documento existente.
        
        Args:
            doc_id: ID del documento a actualizar
            content: Nuevo contenido
            metadata: Nueva metadata
            
        Returns:
            True si se actualizó exitosamente
        """
        try:
            embedding = self.generate_embeddings([content])[0]
            
            self.collection.update(
                ids=[doc_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata]
            )
            
            print(f"Documento {doc_id} actualizado")
            return True
            
        except Exception as e:
            print(f"Error al actualizar documento: {e}")
            return False
    
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Elimina un documento por su ID.
        
        Args:
            doc_id: ID del documento a eliminar
            
        Returns:
            True si se eliminó exitosamente
        """
        try:
            self.collection.delete(ids=[doc_id])
            print(f"Documento {doc_id} eliminado")
            return True
            
        except Exception as e:
            print(f"Error al eliminar documento: {e}")
            return False
    
    
    def clear_collection(self) -> bool:
        """
        Elimina todos los documentos de la colección.
        
        Returns:
            True si se limpió exitosamente
        """
        try:
            # Eliminar colección
            self.client.delete_collection(name=self.collection_name)
            
            # Recrear colección vacía
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"embedding_model": self.embedding_model_name}
            )
            
            print(f"Coleccion '{self.collection_name}' limpiada")
            return True
            
        except Exception as e:
            print(f"Error al limpiar colección: {e}")
            return False
    
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre la colección.
        
        Returns:
            Diccionario con información de la colección
        """
        try:
            return {
                "name": self.collection_name,
                "count": self.collection.count(),
                "embedding_model": self.embedding_model_name,
                "db_path": self.db_path
            }
        except Exception as e:
            print(f"Error al obtener info: {e}")
            return {"name": self.collection_name, "count": 0, "embedding_model": self.embedding_model_name}
    
    
    def get_all_documents(self, limit: Optional[int] = None) -> Any:
        """
        Obtiene todos los documentos de la colección.
        
        Args:
            limit: Número máximo de documentos a devolver (None = todos)
            
        Returns:
            Diccionario con documentos y metadata
        """
        try:
            count = self.collection.count()
            if count == 0:
                return {"documents": [], "metadatas": [], "ids": []}
            
            n_results = limit if limit and limit < count else count
            
            results = self.collection.get(
                limit=n_results,
                include=["documents", "metadatas"]
            )
            
            return dict(results)  # type: ignore
            
        except Exception as e:
            print(f"Error al obtener documentos: {e}")
            return {"documents": [], "metadatas": [], "ids": []}


# Función de prueba
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Probar VectorDatabase")
    parser.add_argument("--test", choices=["search", "info", "add"], 
                       default="info", help="Tipo de prueba a ejecutar")
    parser.add_argument("--query", type=str, default="¿Qué son los LLMs?",
                       help="Consulta para búsqueda (solo con --test search)")
    args = parser.parse_args()
    
    print("Probando VectorDatabase...")
    print("=" * 60)
    
    try:
        # Inicializar base de datos
        vector_db = VectorDatabase()
        
        if args.test == "info":
            # Mostrar información
            info = vector_db.get_collection_info()
            print("\nInformacion de la coleccion:")
            print("-" * 60)
            for key, value in info.items():
                print(f"{key}: {value}")
            
            if info["count"] > 0:
                print("\nPrimeros 3 documentos:")
                docs = vector_db.get_all_documents(limit=3)
                for i, (doc, meta) in enumerate(zip(docs["documents"], docs["metadatas"])):
                    print(f"\n[{i+1}] ID: {docs['ids'][i]}")
                    print(f"    Fuente: {meta.get('source', 'unknown')}")
                    print(f"    Tokens: {meta.get('tokens', 0)}")
                    print(f"    Preview: {doc[:100]}...")
        
        elif args.test == "search":
            # Realizar búsqueda
            print(f"\nBuscando: '{args.query}'")
            print("-" * 60)
            
            results = vector_db.search_similar_documents(args.query, n_results=3)
            
            if results["documents"]:
                print(f"\nEncontrados {len(results['documents'])} resultados:\n")
                for i, (doc, meta, dist) in enumerate(zip(
                    results["documents"], 
                    results["metadatas"], 
                    results["distances"]
                )):
                    similarity = 1 - dist
                    print(f"[{i+1}] Similitud: {similarity:.3f}")
                    print(f"    Fuente: {meta.get('source', 'unknown')} (chunk {meta.get('chunk_id', 0)})")
                    print(f"    Tokens: {meta.get('tokens', 0)}")
                    print(f"    Contenido: {doc[:200]}...")
                    print()
            else:
                print("No se encontraron resultados (la base de datos puede estar vacía)")
        
        elif args.test == "add":
            # Añadir documentos de prueba
            print("\nAñadiendo documentos de prueba...")
            print("-" * 60)
            
            test_chunks = [
                {
                    "content": "Los LLMs (Large Language Models) son modelos de inteligencia artificial entrenados con grandes cantidades de texto.",
                    "metadata": {
                        "id": "test-1",
                        "chunk_id": 0,
                        "source": "test.txt",
                        "source_path": "test.txt",
                        "tokens": 20,
                        "characters": 100
                    }
                },
                {
                    "content": "Python es un lenguaje de programación de alto nivel, interpretado y de propósito general.",
                    "metadata": {
                        "id": "test-2",
                        "chunk_id": 1,
                        "source": "test.txt",
                        "source_path": "test.txt",
                        "tokens": 15,
                        "characters": 90
                    }
                }
            ]
            
            success = vector_db.add_documents(test_chunks)
            if success:
                print("\nDocumentos de prueba añadidos correctamente")
                info = vector_db.get_collection_info()
                print(f"Total de documentos: {info['count']}")
        
        print("\n" + "=" * 60)
        print("Prueba completada exitosamente")
        
    except Exception as e:
        print(f"\nError en la prueba: {e}")
        sys.exit(1)
