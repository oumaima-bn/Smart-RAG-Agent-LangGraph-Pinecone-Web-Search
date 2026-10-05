# Corrections apportées

## 🔴 Bugs bloquants

1. **Fichiers en conflit de noms** — `config.py`, `main.py` et `requirements.txt`
   existaient en deux versions (une pour `frontend/`, une pour `backend/`) mais
   étaient tous à plat dans le même dossier. En les uploadant tel quel, les
   versions frontend étaient silencieusement écrasées par les versions backend.
   **Fix** : séparation en `backend/` et `frontend/`, comme documenté dans le README.

2. **`frontend/app.py`** — `requests` et `json` étaient utilisés (dans les blocs
   `except`) mais jamais importés → `NameError` dès la première erreur réseau.
   **Fix** : ajout de `import requests` et `import json`.

3. **`backend/main.py`** — `rag_agent.stream(...)` était appelé sans préciser
   `stream_mode="updates"`. Le mode par défaut de LangGraph ("values") ne renvoie
   pas un dict `{nom_du_noeud: état}`, ce qui cassait toute la logique de trace
   et la récupération de la réponse finale.
   **Fix** : `stream_mode="updates"` ajouté explicitement, + garde contre un flux vide.

4. **`requirements.txt`** — il manquait `python-multipart`, pourtant requis par
   FastAPI pour gérer l'upload de fichiers (`UploadFile`). Sans lui, l'endpoint
   `/upload-document/` plante à l'exécution.
   **Fix** : ajouté, + `uvicorn[standard]` (nécessaire pour `--reload`).

## 🟠 Bugs silencieux / incohérences

5. **`backend/vectorstore.py`** — le nom d'index Pinecone était codé en dur
   (`"langgraph-rag-index"`), différent du nom `"rag-index"` attendu par le
   README et défini dans `config.py`. Résultat : l'app créait/utilisait un
   index Pinecone différent de celui que l'utilisateur crée manuellement.
   **Fix** : utilise désormais `PINECONE_INDEX_NAME` depuis `config.py`.

6. **`backend/agent.py`** — `retriever_instance.invoke(query, k=5)` : le
   paramètre `k` n'est pas accepté par `.invoke()` sur un retriever LangChain ;
   il doit être fixé à la création via `search_kwargs`.
   **Fix** : `get_retriever(k=5)` configure maintenant `search_kwargs={"k": k}`.

7. **`requirements.txt`** — `uuid` était listé comme dépendance pip, alors que
   c'est un module de la bibliothèque standard Python. L'installer via pip
   peut provoquer des conflits.
   **Fix** : supprimé.

## 🟡 Nettoyage / robustesse

8. **`backend/agent.py`** — nom de modèle Groq (`llama3-70b-8192`) sorti en
   dur dans le code et remplacé par une variable `GROQ_MODEL` dans `config.py`
   (défaut : `llama-3.3-70b-versatile`). ⚠️ Les modèles disponibles sur Groq
   changent régulièrement — vérifie la liste à jour sur
   https://console.groq.com/docs/models avant de lancer l'app.
9. **`requirements.txt`** — suppression de `docx2txt` et `unstructured`
   (non utilisés dans le code actuel, qui ne gère que le PDF ; `unstructured`
   est une dépendance lourde qui peut échouer à l'installation).
10. **`backend/main.py`** — suppression d'un `MemorySaver()` créé mais jamais
    utilisé (le vrai checkpointer est celui créé dans `agent.py`), et ajout
    d'un `except HTTPException: raise` pour ne pas re-envelopper les erreurs
    HTTP volontaires dans un message 500 générique.

## Non testé (nécessite tes propres clés API)

Je ne peux pas exécuter ce projet de bout en bout sans clés Groq/Pinecone/Tavily
réelles, et mon bac à sable n'a pas accès à ces domaines. J'ai vérifié :
- la syntaxe de tous les fichiers (`py_compile`) ✅
- l'import réel du frontend avec ses dépendances ✅
- l'import réel de `backend/config.py` et la résolution correcte de
  `GROQ_MODEL` / `PINECONE_INDEX_NAME` ✅

À toi de valider le comportement runtime une fois tes clés API en place.
