---
author: Franco Carabajal y Germán Unzue
date: 2025-11-18
paging: Slide %d / %d
---

# API de consulta de documentación
## Bazada en RAG y alimentada con llms.txt


 **RAG** = Retrieval-Augmented Generation

---

# El problema

**Los desarrolladores que conozco no leen suficiente documentación...**

**...pero le preguntan a la IA al respecto.**

Así que hicimos una IA que *realmente sepa* documentación.


---

# Demo: Health endpoint

Fran: `Ctrl-e` para correr ejemplo
```bash
curl -s http://localhost:8000/health | jq
```

---

# Demo: Preguntar a la documentación

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How can I link between pages?", "max_results": 3}' \
  | jq --indent 5 '.answer, .sources[0].similarity_score'
```

---

# Stack

- **FastAPI** - API framework (Python)
- **ChromaDB** - DB Vectorial
- **Google Gemini** - Modelo de inferencia
- **Sentence Transformers** - Embeddings
- **llms.txt** - Formato de documentación
- **Swagger** - Documentación de la documentación

---

# Thank You!

Las aplicaciones posibles de esto son:

- Servir un API para que otros se hagan un frontend
- Sumarlo a un IDE para que se le pueda preguntar sobre documentación.
- La implementación es lo suficientemente general como para usar otra cosa en vez de `llms.txt`

