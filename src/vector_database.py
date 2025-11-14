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
