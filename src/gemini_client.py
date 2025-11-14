"""
Módulo para integración con Google Gemini API.
"""
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from config.settings import Config


class GeminiClient:
    """Cliente para interactuar con Google Gemini API."""
    
    def __init__(self):
        """Inicializa el cliente de Gemini."""
        if not Config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY no está configurado. Por favor, configura tu API key en el archivo .env")
        
        # Configurar la API key
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        
        # Inicializar el modelo
        try:
            # Intentar con los modelos más recientes disponibles
            self.model = genai.GenerativeModel('models/gemini-2.5-flash-preview-05-20')
        except Exception:
            try:
                self.model = genai.GenerativeModel('models/gemini-2.5-pro-preview-03-25')
            except Exception:
                # Listar y usar el primer modelo generativo disponible
                models = list(genai.list_models())
                generative_models = [m for m in models if 'generateContent' in m.supported_generation_methods]
                if generative_models:
                    self.model = genai.GenerativeModel(generative_models[0].name)
                else:
                    raise ValueError("No se encontraron modelos generativos disponibles")
        
        # Configuración de generación
        self.generation_config = genai.types.GenerationConfig(
            max_output_tokens=Config.MAX_TOKENS,
            temperature=Config.TEMPERATURE,
        )
    
    def generate_response(self, prompt: str, context_documents: List[str] = None) -> str:
        """
        Genera una respuesta usando Gemini basada en el prompt y contexto.
        
        Args:
            prompt (str): Pregunta o prompt del usuario
            context_documents (List[str], optional): Documentos de contexto relevantes
            
        Returns:
            str: Respuesta generada por Gemini
        """
        try:
            # Construir el prompt completo
            full_prompt = self._build_rag_prompt(prompt, context_documents)
            
            # Generar respuesta
            response = self.model.generate_content(
                full_prompt,
                generation_config=self.generation_config
            )
            
            return response.text
            
        except Exception as e:
            return f"Error al generar respuesta: {str(e)}"
    
    def _build_rag_prompt(self, user_query: str, context_documents: List[str] = None) -> str:
        """
        Construye un prompt optimizado para RAG.
        
        Args:
            user_query (str): Pregunta del usuario
            context_documents (List[str], optional): Documentos de contexto
            
        Returns:
            str: Prompt completo para Gemini
        """
        base_prompt = """Eres un asistente especializado en Modelos de Lenguaje Grande (LLMs) y tecnologías de inteligencia artificial. 
Tu tarea es responder preguntas basándote ÚNICAMENTE en la información proporcionada en los documentos de contexto. 
INSTRUCCIONES IMPORTANTES:
1. Utiliza SOLO la información de los documentos de contexto proporcionados
2. Si la información no está en el contexto, indica claramente que no tienes esa información
3. Sé preciso y cita información específica cuando sea relevante
4. Si la pregunta no está relacionada con el contenido proporcionado, indícalo educadamente
5. Responde siempre en español
"""
        
        if context_documents and len(context_documents) > 0:
            # Añadir documentos de contexto
            context_section = "DOCUMENTOS DE CONTEXTO:\n"
            for i, doc in enumerate(context_documents, 1):
                context_section += f"\n--- Documento {i} ---\n{doc}\n"
            
            context_section += f"\n\nPREGUNTA DEL USUARIO: {user_query}\n\nRESPUESTA:"
            
            return base_prompt + context_section
        else:
            # Sin contexto, respuesta directa
            no_context_prompt = base_prompt + f"""
No se encontraron documentos relevantes en la base de conocimiento para responder tu pregunta.
PREGUNTA DEL USUARIO: {user_query}
RESPUESTA: Lo siento, no encontré información relevante en mi base de conocimientos para responder tu pregunta sobre: "{user_query}". 
Mi conocimiento está limitado a la información específica cargada en el sistema. 
¿Podrías reformular tu pregunta o hacer una consulta más específica sobre LLMs?
"""
            return no_context_prompt
    
    def generate_summary(self, text: str) -> str:
        """
        Genera un resumen del texto proporcionado.
        
        Args:
            text (str): Texto a resumir
            
        Returns:
            str: Resumen generado
        """
        summary_prompt = f"""
Por favor, genera un resumen conciso y informativo del siguiente texto sobre LLMs:
TEXTO:
{text}
RESUMEN:
"""  
        try:
            response = self.model.generate_content(
                summary_prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception as e:
            return f"Error al generar resumen: {str(e)}"
    
    def suggest_follow_up_questions(self, user_query: str, context_documents: List[str]) -> List[str]:
        """
        Sugiere preguntas de seguimiento basadas en la consulta y contexto.
        
        Args:
            user_query (str): Pregunta original del usuario
            context_documents (List[str]): Documentos de contexto
            
        Returns:
            List[str]: Lista de preguntas sugeridas
        """
        if not context_documents:
            return []
        
        context_preview = " ".join(context_documents)[:1000]  # Limitar contexto
        
        follow_up_prompt = f"""
Basándote en la pregunta del usuario y el contexto proporcionado, sugiere 3 preguntas de seguimiento relevantes que el usuario podría estar interesado en hacer.
PREGUNTA ORIGINAL: {user_query}
CONTEXTO: {context_preview}...
Proporciona exactamente 3 preguntas de seguimiento, una por línea, numeradas del 1 al 3:
"""
        
        try:
            response = self.model.generate_content(
                follow_up_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=200,
                    temperature=0.7
                )
            )
            
            # Parsear respuesta para extraer preguntas
            lines = response.text.strip().split('\n')
            questions = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Limpiar numeración
                    clean_question = line.split('.', 1)[-1].strip() if '.' in line else line
                    clean_question = clean_question.lstrip('- ').strip()
                    if clean_question:
                        questions.append(clean_question)
            
            return questions[:3]  # Limitar a 3 preguntas
            
        except Exception as e:
            print(f"Error al generar preguntas de seguimiento: {e}")
            return []
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Prueba la conexión con Gemini API.
        
        Returns:
            Dict[str, Any]: Resultado de la prueba
        """
        try:
            # Primero listar modelos disponibles
            models = list(genai.list_models())
            available_models = [m.name for m in models]
            print(f"Modelos disponibles: {available_models[:3]}...")  # Mostrar primeros 3
            
            test_prompt = "Di 'Conexión exitosa' si puedes leer este mensaje."
            response = self.model.generate_content(test_prompt)
            
            return {
                "success": True,
                "message": "Conexión con Gemini API exitosa",
                "response": response.text,
                "model": self.model.model_name if hasattr(self.model, 'model_name') else "gemini",
                "available_models": len(available_models)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error de conexión: {str(e)}",
                "model": "unknown"
            }


def main():
    """Función principal para testing del módulo."""
    try:
        # Inicializar cliente
        client = GeminiClient()
        
        # Probar conexión
        print("Probando conexión con Gemini...")
        test_result = client.test_connection()
        
        if test_result["success"]:
            print("✅ Conexión exitosa!")
            print(f"Modelo: {test_result['model']}")
            print(f"Respuesta de prueba: {test_result['response']}")
            
            # Ejemplo de generación con contexto
            print("\n--- Ejemplo de RAG ---")
            context = ["Los LLMs son modelos de lenguaje grande que utilizan transformers."]
            query = "¿Qué son los LLMs?"
            
            response = client.generate_response(query, context)
            print(f"Pregunta: {query}")
            print(f"Respuesta: {response}")
            
        else:
            print("❌ Error de conexión:")
            print(test_result["message"])
            
    except Exception as e:
        print(f"Error al inicializar cliente Gemini: {e}")
        print("Asegúrate de que tu API key esté configurada correctamente en el archivo .env")


if __name__ == "__main__":
    main()