## 1. Add session-backed history storage
- [ ] Introduce `st.session_state` structures for user/assistant turns and initialize them when the app loads.
- [ ] Create helper functions to append new turns, truncate to the configured memory window, and allow resending/editing a turn.

## 2. Update UI for interactive chat
- [ ] Replace the current single text area with a chat-style layout that lists prior turns, supports scrolling, and exposes controls (e.g., “Resend” or “Edit”) per entry.
- [ ] Ensure submitting a message updates the history view immediately and focuses the input for fast follow-ups.

## 3. Enhance retrieval + prompting with memory
- [ ] When composing the RAG prompt, include the last N turns (user + assistant) ahead of the current question so OpenAI sees short-term context.
- [ ] Cap the tokens/turns included (e.g., last 5 exchanges or configurable limit) to avoid prompt bloat.

## 4. Validation
- [ ] Manually test by asking multi-step questions, resending previous prompts, and confirming that references to earlier answers work without retyping.
- [ ] Verify history clears when the session resets/reloads to avoid leaking conversations between users.
