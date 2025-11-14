# 🚀 Guía de Inicio Rápido

## Pasos para ejecutar el sistema RAG

### 1. Instalación
```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configuración
```bash
# Copiar archivo de configuración
cp .env.example .env

# Editar .env y añadir tu API key de Google:
# GOOGLE_API_KEY=tu_api_key_aqui
```

### 3. Inicialización
```bash
# Ejecutar script de configuración
python setup.py
```

### 4. Ejecutar servidor
```bash
# Iniciar API
python src/main.py

# El servidor estará disponible en:
# http://localhost:8000
```

### 5. Probar sistema
```bash
# Verificar estado
curl http://localhost:8000/health

# Hacer consulta
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "¿Qué son los LLMs?"}'
```

### 6. Desactivar o borrar venv
```bash
# Desactivat el venv
deactivate                # Linux/Mac
venv\Scripts\deactivate    # Windows

# o podemos eliminar la carpeta
rm -rf venv
```

## 📚 Endpoints principales

- **GET** `/` - Página de inicio
- **POST** `/query` - Hacer consultas
- **GET** `/health` - Estado del sistema
- **GET** `/docs` - Documentación Swagger

## 🔧 Personalización

Para usar tus propios documentos:
1. Reemplaza el contenido de `data/llms.txt`
2. Ejecuta `python setup.py` para recargar
3. ¡Listo! El sistema usará tu contenido