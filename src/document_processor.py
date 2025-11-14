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