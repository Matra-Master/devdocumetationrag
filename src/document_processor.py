"""
PROCESADOR DE DOCUMENTOS 
=========================

Básicamente, este archivo tiene que agarrar un documento de texto (como el llms.txt) 
y cortarlo en pedacitos inteligentes para que después el sistema RAG pueda buscar 
información relevante.

QUÉ NECESITAS HACER:

1. LEER ARCHIVOS
   - Una función que lea archivos .txt
   - Que no se rompa si el archivo no existe
   - Que maneje UTF-8 correctamente

2. CONTAR TOKENS  
   - Usar tiktoken para contar tokens (no palabras!)
   - Esto es importante para que no se pase del límite del modelo de IA
   - Configurar el encoding correcto

3. CORTAR EL TEXTO EN CHUNKS
   - Dividir el texto en pedazos de tamaño configurable
   - Prioridad: primero dividir por párrafos, después por oraciones
   - No cortar palabras por la mitad
   - Cada chunk debe tener sentido por sí solo

4. OVERLAP ENTRE CHUNKS
   - Que los chunks se superpongan un poco para no perder contexto
   - Si un párrafo habla de "transformers" y el siguiente también, 
     que ambos chunks tengan esa info para mantener coherencia

5. METADATA
   - A cada chunk ponerle un ID único
   - Guardar de qué archivo viene, qué número de chunk es, cuántos tokens tiene
   - Esto sirve para después saber de dónde salió cada resultado


TIPS PARA LA IMPLEMENTACIÓN:
- Usa text.split('\\n\\n') para dividir por párrafos
- Usa text.split('. ') para dividir por oraciones  
- El overlap puede ser algo como las últimas 50 palabras del chunk anterior
- Si un párrafo es muy largo, divídelo por oraciones
- Siempre manejar errores (try/except)

EL OBJETIVO FINAL:
Que cuando le pases el archivo llms.txt, te devuelva una lista de chunks 
listos para meter en la base de datos vectorial, con toda su metadata bien puesta.
"""

import os
import sys
import tiktoken
from typing import List, Dict, Any
import uuid

# Añadir el directorio padre al path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Config


class DocumentProcessor:
    """
    Procesa documentos de texto y los divide en chunks inteligentes para RAG.
    """
    def __init__(self, chunk_size: int = 0, chunk_overlap: int = 0):
        """
        Inicializa el procesador de documentos.
        Args:
            chunk_size: Tamaño máximo de cada chunk en tokens (default: Config.CHUNK_SIZE)
            chunk_overlap: Cantidad de tokens de superposición entre chunks (default: Config.CHUNK_OVERLAP)
        """
        self.chunk_size = chunk_size if chunk_size > 0 else Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap > 0 else Config.CHUNK_OVERLAP
        # Inicializar tokenizador de tiktoken (compatible con modelos de OpenAI/GPT)
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"⚠️  Error al cargar encoding de tiktoken: {e}")
            print("   Usando encoding por defecto...")
            self.encoding = tiktoken.get_encoding("gpt2")
    def read_file(self, file_path: str) -> str:
        """
        Lee un archivo de texto de forma segura.
        Args:
            file_path: Ruta al archivo a leer
        Returns:
            Contenido del archivo como string
        Raises:
            FileNotFoundError: Si el archivo no existe
            Exception: Si hay un error al leer el archivo
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"El archivo {file_path} no existe")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if not content.strip():
                raise ValueError(f"El archivo {file_path} está vacío")
            print(f"✅ Archivo leído exitosamente: {file_path} ({len(content)} caracteres)")
            return content
        except FileNotFoundError:
            raise
        except Exception as e:
            raise Exception(f"Error al leer archivo {file_path}: {str(e)}")
    def count_tokens(self, text: str) -> int:
        """
        Cuenta la cantidad de tokens en un texto usando tiktoken.
        Args:
            text: Texto a contar
        Returns:
            Número de tokens
        """
        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            print(f"⚠️  Error al contar tokens: {e}")
            # Fallback: aproximación usando palabras
            return len(text.split()) * 2
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Divide un texto en oraciones de forma inteligente.
        Args:
            text: Texto a dividir
        Returns:
            Lista de oraciones
        """
        # Dividir por puntos, pero mantener los puntos
        sentences = []
        current = ""
        for char in text:
            current += char
            if char in ['.', '!', '?', '\n']:
                stripped = current.strip()
                if stripped:
                    sentences.append(stripped)
                current = ""
        # Agregar cualquier texto restante
        if current.strip():
            sentences.append(current.strip())
        return sentences
    def create_chunks(self, text: str) -> List[str]:
        """
        Divide el texto en chunks inteligentes con overlap.
        Prioridad:
        1. Primero dividir por párrafos (\\n\\n)
        2. Si un párrafo es muy largo, dividir por oraciones
        3. Mantener overlap entre chunks
        Args:
            text: Texto completo a dividir
        Returns:
            Lista de chunks de texto
        """
        chunks = []
        # Dividir por párrafos primero
        paragraphs = text.split('\n\n')
        current_chunk = ""
        current_tokens = 0
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            paragraph_tokens = self.count_tokens(paragraph)
            # Si el párrafo solo es muy grande, dividir por oraciones
            if paragraph_tokens > self.chunk_size:
                # Guardar chunk actual si existe
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0
                # Dividir párrafo largo en oraciones
                sentences = self.split_into_sentences(paragraph)
                for sentence in sentences:
                    sentence_tokens = self.count_tokens(sentence)
                    if current_tokens + sentence_tokens > self.chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            # Crear overlap: tomar las últimas palabras del chunk anterior
                            words = current_chunk.split()
                            overlap_words = min(len(words), 50)  # Máximo 50 palabras de overlap
                            overlap_text = ' '.join(words[-overlap_words:])
                            current_chunk = overlap_text + ' ' + sentence
                            current_tokens = self.count_tokens(current_chunk)
                        else:
                            current_chunk = sentence
                            current_tokens = sentence_tokens
                    else:
                        current_chunk += ' ' + sentence if current_chunk else sentence
                        current_tokens += sentence_tokens
            # Si el párrafo cabe en el chunk actual
            elif current_tokens + paragraph_tokens <= self.chunk_size:
                current_chunk += '\n\n' + paragraph if current_chunk else paragraph
                current_tokens += paragraph_tokens
            # Si no cabe, guardar chunk actual y empezar uno nuevo con overlap
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Crear overlap
                    words = current_chunk.split()
                    overlap_words = min(len(words), 50)
                    overlap_text = ' '.join(words[-overlap_words:])
                    current_chunk = overlap_text + '\n\n' + paragraph
                    current_tokens = self.count_tokens(current_chunk)
                else:
                    current_chunk = paragraph
                    current_tokens = paragraph_tokens
        # Agregar el último chunk si existe
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    def create_chunk_metadata(self, chunk: str, chunk_id: int, source_file: str) -> Dict[str, Any]:
        """
        Crea metadata para un chunk.
        Args:
            chunk: Texto del chunk
            chunk_id: ID numérico del chunk
            source_file: Archivo de origen
        Returns:
            Diccionario con metadata del chunk
        """
        return {
            "id": str(uuid.uuid4()),  # ID único universal
            "chunk_id": chunk_id,      # ID numérico secuencial
            "source": os.path.basename(source_file),
            "source_path": source_file,
            "tokens": self.count_tokens(chunk),
            "characters": len(chunk)
        }
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Procesa un archivo completo y devuelve chunks con metadata.
        Args:
            file_path: Ruta al archivo a procesar
        Returns:
            Lista de diccionarios con 'content' y 'metadata'
        """
        try:
            # Leer archivo
            print(f"📖 Procesando archivo: {file_path}")
            content = self.read_file(file_path)
            # Crear chunks
            print(f"✂️  Dividiendo en chunks (tamaño: {self.chunk_size} tokens, overlap: {self.chunk_overlap} tokens)...")
            chunks = self.create_chunks(content)
            # Crear chunks con metadata
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "content": chunk,
                    "metadata": self.create_chunk_metadata(chunk, i, file_path)
                }
                processed_chunks.append(chunk_data)
            print(f"✅ Procesamiento completo: {len(processed_chunks)} chunks creados")
            # Mostrar estadísticas
            total_tokens = sum(c["metadata"]["tokens"] for c in processed_chunks)
            avg_tokens = total_tokens / len(processed_chunks) if processed_chunks else 0
            print(f"📊 Estadísticas:")
            print(f"   - Total de tokens: {total_tokens}")
            print(f"   - Promedio de tokens por chunk: {avg_tokens:.1f}")
            print(f"   - Chunk más pequeño: {min(c['metadata']['tokens'] for c in processed_chunks)} tokens")
            print(f"   - Chunk más grande: {max(c['metadata']['tokens'] for c in processed_chunks)} tokens")
            return processed_chunks
        except Exception as e:
            print(f"❌ Error al procesar archivo: {e}")
            raise
    def process_llms_file(self) -> List[Dict[str, Any]]:
        """
        Procesa específicamente el archivo llms.txt desde la configuración.
        Returns:
            Lista de chunks con metadata listos para la base de datos vectorial
        """
        try:
            llms_path = Config.LLMS_FILE_PATH
            if not os.path.exists(llms_path):
                raise FileNotFoundError(
                    f"Archivo llms.txt no encontrado en: {llms_path}\n"
                    f"Asegúrate de que el archivo existe en el directorio {Config.DATA_DIR}"
                )
            return self.process_file(llms_path)
        except Exception as e:
            print(f"❌ Error al procesar llms.txt: {e}")
            raise


# Función de prueba
if __name__ == "__main__":
    print("🧪 Probando DocumentProcessor...")
    print("=" * 60)
    try:
        processor = DocumentProcessor()
        chunks = processor.process_llms_file()
        print("\n" + "=" * 60)
        print("🎉 Prueba exitosa!")
        print(f"Se crearon {len(chunks)} chunks")
        if chunks:
            print("\n📝 Ejemplo del primer chunk:")
            print("-" * 60)
            print(f"Contenido (primeros 200 caracteres):")
            print(chunks[0]["content"][:200] + "...")
            print(f"\nMetadata:")
            for key, value in chunks[0]["metadata"].items():
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"\n❌ Error en la prueba: {e}")
        sys.exit(1)
