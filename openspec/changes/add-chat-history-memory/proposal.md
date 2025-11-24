# Change: Add interactive chat history with short-term memory

## Why
QA engineers currently submit one-off prompts—the UI clears the context after every answer, and the model can't reference prior exchanges. This makes follow-up questions tedious and prevents the assistant from tailoring responses to an evolving conversation.

## What Changes
- Persist recent user/assistant turns within the Streamlit session so the chatbot can reference the latest exchanges when composing answers.
- Render an interactive chat history pane that shows each turn, allows scrolling, and lets users resend/edit a previous question to continue the thread.
- Update retrieval + prompt construction to include a sliding memory window (e.g., last 5 turns) before invoking OpenAI so short-term context is preserved.

## Impact
- Affected specs: `chat-experience`
- Affected code: `app.py` (Streamlit chat tab, session state mgmt, prompt construction)
