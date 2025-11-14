# RAG for developer documentation

Sistema de documentación basado en RAG (Retrieval-Augmented Generation) que utiliza **Google Gemini** para responder preguntas sobre contenido técnico almacenado en una base de datos vectorial **ChromaDB**.

## Características

- **Procesamiento inteligente de documentos**: Divide automáticamente el contenido en chunks optimizados
- **Base de datos vectorial**: Utiliza ChromaDB para almacenamiento y búsqueda semántica eficiente
- **IA conversacional**: Integración con Google Gemini para respuestas naturales y contextualmente relevantes
- **API REST completa**: Endpoints para consultas, administración y monitoreo
- **Búsqueda semántica**: Encuentra contenido relevante usando embeddings de alta calidad
- **Interfaz web**: Página de inicio con documentación y ejemplos

## Arquitectura

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documentos    │    │   ChromaDB       │    │   Google        │
│   (llms.txt)    │───▶│   (Vectorial)    │───▶│   Gemini API    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Server                              │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────────────────┐   │
│  │ Document    │ │ Vector       │ │ Gemini Client           │   │
│  │ Processor   │ │ Database     │ │ (RAG + Response Gen)    │   │
│  └─────────────┘ └──────────────┘ └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   API Endpoints     │
                    │ /query, /health,    │
                    │ /info, /search      │
                    └─────────────────────┘
```

## Instalación y Configuración

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd devdocumetationrag
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# venv\Scripts\activate   # En Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
```

Edita el archivo `.env` y añade tu API key de Google:

```env
GOOGLE_API_KEY=tu_api_key_aqui
```

#### Obtener API Key de Google Gemini

1. Ve a [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Crea una nueva API key
3. Copia la clave y pégala en tu archivo `.env`

### 5. Ejecutar configuración inicial

```bash
python setup.py
```

Este script:
- Verifica todos los requisitos
- Inicializa la base de datos vectorial ChromaDB
- Procesa el archivo `llms.txt` y lo divide en chunks
- Genera embeddings y los almacena
- Prueba la conexión con Gemini
- Ejecuta una consulta de ejemplo

### 6. Iniciar el servidor

```bash
python src/main.py
```

El servidor estará disponible en: http://localhost:8000


## API Endpoints

### Página Principal
- **GET** `/` - Página de inicio con documentación

### Estado del Sistema
- **GET** `/health` - Verifica el estado de todos los componentes
- **GET** `/info` - Información sobre la base de datos

### Consultas
- **POST** `/query` - Realiza consultas con RAG
- **GET** `/search` - Búsqueda directa en documentos

### Administración
- **POST** `/reload-documents` - Recarga documentos desde archivo

### Documentación
- **GET** `/docs` - Documentación interactiva (Swagger)

## Ejemplos de Uso

### Consulta básica

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "¿Qué son los LLMs?"}'
```

### Búsqueda de documentos

```bash
curl "http://localhost:8000/search?q=transformers&limit=3"
```

### Verificar estado

```bash
curl "http://localhost:8000/health"
```

### Respuesta de ejemplo

```json
{
  "question": "¿Qué son los LLMs?",
  "answer": "Los Modelos de Lenguaje Grande (LLMs) son sistemas de inteligencia artificial que han revolucionado el procesamiento de lenguaje natural...",
  "sources": [
    {
      "rank": 1,
      "content_preview": "Los Modelos de Lenguaje Grande (Large Language Models o LLMs) son sistemas...",
      "source": "llms.txt",
      "chunk_id": 0,
      "similarity_score": 0.89,
      "tokens": 245
    }
  ],
  "follow_up_questions": [
    "¿Cuáles son las principales arquitecturas de LLMs?",
    "¿Cómo se entrenan los modelos de lenguaje grande?",
    "¿Qué aplicaciones tienen los LLMs en la actualidad?"
  ]
}
```

## Configuración Avanzada

### Variables de Entorno

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `GOOGLE_API_KEY` | API Key de Google Gemini | **Requerido** |
| `CHROMA_DB_PATH` | Ruta de la base de datos ChromaDB | `./chroma_db` |
| `CHROMA_COLLECTION_NAME` | Nombre de la colección | `llms_docs` |
| `HOST` | Host del servidor | `0.0.0.0` |
| `PORT` | Puerto del servidor | `8000` |
| `EMBEDDING_MODEL` | Modelo de embeddings | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Tamaño de chunks en tokens | `1000` |
| `CHUNK_OVERLAP` | Superposición entre chunks | `200` |
| `GEMINI_MODEL` | Modelo de Gemini a usar | `gemini-1.5-flash` |
| `MAX_TOKENS` | Tokens máximos en respuesta | `1024` |
| `TEMPERATURE` | Temperatura para generación | `0.7` |

### Personalización de Documentos

Para usar tus propios documentos:

1. Reemplaza el contenido de `data/llms.txt` con tu texto
2. Ejecuta `python setup.py` para recargar
3. O usa el endpoint `/reload-documents`

### Modelos de Embeddings

Puedes cambiar el modelo de embeddings editando `EMBEDDING_MODEL` en `.env`:

- `all-MiniLM-L6-v2` (por defecto) - Rápido y eficiente
- `all-mpnet-base-v2` - Mayor calidad, más lento
- `multi-qa-MiniLM-L6-cos-v1` - Optimizado para Q&A


## Solución de Problemas

### Error: "No se ha podido resolver la importación"

Asegúrate de tener todas las dependencias instaladas:

```bash
pip install -r requirements.txt
```

### Error: "GOOGLE_API_KEY no configurado"

1. Verifica que el archivo `.env` existe
2. Asegúrate de que contiene `GOOGLE_API_KEY=tu_clave_aqui`
3. Reinicia el servidor

### Error: "Archivo llms.txt no encontrado"

Verifica que el archivo `data/llms.txt` existe y contiene contenido.

### ChromaDB no funciona

Elimina la carpeta `chroma_db` y ejecuta `python setup.py` nuevamente.

## Desarrollo

### Estructura de Módulos

- **`document_processor.py`**: Maneja la lectura y división de documentos en chunks
- **`vector_database.py`**: Interfaz con ChromaDB para almacenamiento vectorial
- **`gemini_client.py`**: Cliente para Google Gemini API
- **`main.py`**: Servidor FastAPI con todos los endpoints

### Añadir Nuevas Características

1. **Nuevo endpoint**: Añade funciones en `main.py`
2. **Nuevo tipo de documento**: Extiende `DocumentProcessor`
3. **Nuevo modelo de embeddings**: Modifica `VectorDatabase`
4. **Nuevas opciones de Gemini**: Extiende `GeminiClient`

### Optimizaciones Implementadas

- **Chunks inteligentes**: División basada en tokens con superposición
- **Embeddings eficientes**: Modelo optimizado `all-MiniLM-L6-v2`
- **Caché de ChromaDB**: Almacenamiento persistente en disco
- **API asíncrona**: FastAPI para manejo concurrente
