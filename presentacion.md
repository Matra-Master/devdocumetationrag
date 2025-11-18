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

# Live Demo: Ask a Question

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are LLMs?", "max_results": 3}' \
  | jq --indent 5 '.answer, .sources[0].similarity_score'
```

---

# Live Demo: Semantic Search

```bash
curl -s "http://localhost:8000/search?q=transformer%20architecture&limit=2" \
  | jq '.results[] | {rank, similarity_score, preview: .content[:100]}'
```

---

# Tech Stack

- **FastAPI** - REST API framework
- **ChromaDB** - Vector database
- **Google Gemini** - LLM inference
- **Sentence Transformers** - Embeddings
- **llms.txt** - Documentation format

---

# Thank You!

**Questions?**

Repository: github.com/yourusername/devdocumetationrag
Docs: http://localhost:8000/docs
