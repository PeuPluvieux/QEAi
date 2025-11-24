# Project Context

## Purpose
MiTAC QE Local RAG Assistant is a Streamlit application that lets quality engineering teams upload MiTAC documentation (PDF, DOCX, PPT, XLS/XLSX, TXT), index it locally, and ask grounded questions. It keeps all source material on disk, summarizes relevant passages with OpenAI models, and cites the originating document so QE engineers can trace every answer.

## Tech Stack
- Python 3.x with Streamlit for the interactive UI and control flow
- OpenAI Python SDK (`OpenAI`) for embeddings (`text-embedding-3-small`) and responses (`gpt-4.1-mini`)
- PyPDF2, python-docx, python-pptx, and pandas for document parsing across QE file formats
- NumPy for vector math and storage of embeddings in `index/embeddings.npy`
- Local JSONL store (`index/chunks.jsonl`) for chunk metadata plus files staged under `data/`

## Project Conventions

### Code Style
- Follow Python's PEP8 defaults (snake_case functions, 4-space indents, module-level constants for config).
- Keep Streamlit widgets grouped by tab and prefer descriptive labels/help text so non-engineers understand options.
- Avoid silent failures when ingesting by returning human-readable status strings for UI display.

### Architecture Patterns
- Single-page Streamlit app split into logical sections: ingestion/indexing helpers, retrieval/answering, and UI tabs.
- Lightweight RAG design: chunk text locally, persist embeddings/metadata, and use cosine similarity + OpenAI answers.
- Two-stage retrieval fallback (strict retrieval, query rewrite, ungrounded response) to balance precision and recall.
- File-system boundaries: raw docs under `data/<project>/<folder>/`, derived indexes inside `index/`.

### Testing Strategy
- No automated test suite yet; validation happens manually by uploading seed documents and exercising chat/upload/browse tabs in Streamlit.
- When touching retrieval logic, sanity-check similarity scores and verify cited documents match the quotes shown.
- For ingestion changes, re-run the "Scan data/ folder and (re)build index" flow to ensure re-indexing succeeds on representative files.

### Git Workflow
- Work from feature branches named after the change (e.g., `add-project-context-doc`) and open PRs back to `main`.
- Keep commits scoped to a single concern (ingestion, retrieval, UI) so QA can bisect quickly.
- Run Streamlit locally before pushing to confirm no regressions in the interactive flows.

## Domain Context
- Target users are MiTAC Quality Engineering teams compiling lab/installation procedures, test specs, and compliance docs.
- Questions tend to involve cable handling rules, fixture requirements, QE workflows, and similar manufacturing/QE content.
- Responses must be citeable; engineers rely on references to match supplier/regulatory documentation.

## Important Constraints
- Data privacy: documents stay on the local workstation; only embeddings and prompts are sent to OpenAI.
- Only OpenAI API credentials stored in `.txt` are used—no vendor-specific APIs described yet.
- Embedding dimensions fixed at 1536 (per `text-embedding-3-small`); altering the model requires migrating stored vectors.
- Streamlit app assumes a writable filesystem for `data/` and `index/`; deployments must preserve these directories.

## External Dependencies
- OpenAI API (embeddings + responses) – requires `OPENAI_API_KEY`.
- Streamlit runtime for the UI.
- Document parsing libraries: PyPDF2, python-docx, python-pptx, pandas (with Excel backends) for ingestion.
- NumPy for linear algebra and storing vectors on disk.
