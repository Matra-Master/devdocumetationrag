#!/usr/bin/env python3
"""
Script de inicialización para cargar documentos y configurar el sistema RAG.
"""
import os
import sys

# Añadir el directorio src al path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.insert(0, src_dir)

from src.document_processor import DocumentProcessor
from src.vector_database import VectorDatabase
from src.gemini_client import GeminiClient
from config.settings import Config


def check_requirements():
    """Verifica que todos los requisitos estén disponibles."""
    print("🔍 Verificando requisitos...")
    
    # Verificar archivo llms.txt
    if not os.path.exists(Config.LLMS_FILE_PATH):
        print(f"❌ Error: Archivo {Config.LLMS_FILE_PATH} no encontrado")
        print("   Por favor, asegúrate de que el archivo llms.txt esté en la carpeta data/")
        return False
    
    # Verificar API key de Google
    if not Config.GOOGLE_API_KEY:
        print("❌ Error: GOOGLE_API_KEY no configurado")
        print("   Por favor, configura tu API key en el archivo .env")
        print("   Puedes copiar .env.example a .env y añadir tu API key")
        return False
    
    print("✅ Todos los requisitos están disponibles")
    return True


def initialize_database():
    """Inicializa y carga la base de datos vectorial."""
    print("\n📚 Inicializando base de datos vectorial...")
    
    try:
        # Inicializar procesador de documentos
        print("🔄 Inicializando procesador de documentos...")
        processor = DocumentProcessor()
        
        # Inicializar base de datos vectorial
        print("🔄 Inicializando ChromaDB...")
        vector_db = VectorDatabase()
        
        # Limpiar colección existente (si existe)
        print("🧹 Limpiando colección existente...")
        vector_db.clear_collection()
        
        # Procesar archivo llms.txt
        print("📖 Procesando archivo llms.txt...")
        chunks = processor.process_llms_file()
        print(f"✅ Se generaron {len(chunks)} chunks del documento")
        
        # Cargar chunks en la base de datos vectorial
        print("💾 Cargando chunks en ChromaDB...")
        success = vector_db.add_documents(chunks)
        
        if success:
            print("✅ Base de datos vectorial inicializada correctamente")
            
            # Mostrar información de la colección
            info = vector_db.get_collection_info()
            print(f"📊 Información de la colección:")
            print(f"   - Nombre: {info['name']}")
            print(f"   - Documentos: {info['count']}")
            print(f"   - Modelo de embeddings: {info['embedding_model']}")
            
            return True
        else:
            print("❌ Error al cargar documentos en la base de datos")
            return False
            
    except Exception as e:
        print(f"❌ Error al inicializar base de datos: {e}")
        return False


def test_gemini_connection():
    """Prueba la conexión con Gemini API."""
    print("\n🤖 Probando conexión con Gemini...")
    
    try:
        client = GeminiClient()
        result = client.test_connection()
        
        if result["success"]:
            print("✅ Conexión con Gemini exitosa")
            print(f"📱 Modelo: {result['model']}")
            return True
        else:
            print(f"❌ Error de conexión con Gemini: {result['message']}")
            return False
            
    except Exception as e:
        print(f"❌ Error al conectar con Gemini: {e}")
        return False


def run_sample_query():
    """Ejecuta una consulta de ejemplo para probar el sistema completo."""
    print("\n🧪 Ejecutando consulta de ejemplo...")
    
    try:
        # Inicializar servicios
        vector_db = VectorDatabase()
        gemini_client = GeminiClient()
        
        # Consulta de ejemplo
        sample_query = "¿Qué son los LLMs y cuáles son sus principales características?"
        print(f"📝 Pregunta: {sample_query}")
        
        # Buscar documentos relevantes
        results = vector_db.search_similar_documents(sample_query, n_results=3)
        context_docs = results.get("documents", [])
        
        if not context_docs:
            print("⚠️  No se encontraron documentos relevantes")
            return False
        
        # Generar respuesta
        response = gemini_client.generate_response(sample_query, context_docs)
        
        print(f"\n💬 Respuesta:")
        print(f"{response}")
        
        # Mostrar fuentes
        print(f"\n📚 Fuentes utilizadas:")
        for i, doc in enumerate(context_docs[:2], 1):
            preview = doc[:150] + "..." if len(doc) > 150 else doc
            print(f"   {i}. {preview}")
        
        print("\n✅ Consulta de ejemplo completada exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en consulta de ejemplo: {e}")
        return False


def main():
    """Función principal del script de inicialización."""
    print("🚀 Iniciando configuración del sistema RAG...")
    print("=" * 50)
    
    # 1. Verificar requisitos
    if not check_requirements():
        print("\n❌ Configuración cancelada debido a requisitos faltantes")
        sys.exit(1)
    
    # 2. Inicializar base de datos
    if not initialize_database():
        print("\n❌ Configuración cancelada debido a error en base de datos")
        sys.exit(1)
    
    # 3. Probar conexión con Gemini
    if not test_gemini_connection():
        print("\n❌ Configuración cancelada debido a error en Gemini")
        sys.exit(1)
    
    # 4. Ejecutar consulta de ejemplo
    if not run_sample_query():
        print("\n⚠️  Configuración completada pero con errores en la consulta de ejemplo")
    
    print("\n" + "=" * 50)
    print("🎉 ¡Sistema RAG configurado exitosamente!")
    print("\n📋 Próximos pasos:")
    print("1. Ejecutar el servidor: python src/main.py")
    print("2. Abrir el navegador en: http://localhost:8000")
    print("3. Probar la API en: http://localhost:8000/docs")
    print("\n💡 Comandos útiles:")
    print("   - Probar conexión: curl http://localhost:8000/health")
    print("   - Hacer consulta: curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{\"question\":\"¿Qué son los LLMs?\"}'")


if __name__ == "__main__":
    main()