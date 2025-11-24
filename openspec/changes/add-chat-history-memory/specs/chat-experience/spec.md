## ADDED Requirements

### Requirement: Show interactive chat history
The chatbot UI SHALL persist and render the most recent user/assistant messages for the active Streamlit session so users can review, scroll, and re-send any prior turn without re-entering text.

#### Scenario: Display existing conversation
- **GIVEN** a user has submitted at least one question and received an answer
- **WHEN** the chat tab remains open in the same session
- **THEN** all prior user/assistant turns appear in chronological order with timestamps or message badges
- **AND** the user can click an affordance (e.g., “Resend”/“Edit”) on a past user turn to populate the input box
- **AND** re-submitting uses the selected text without losing other history entries.

### Requirement: Maintain short-term conversational memory
The chatbot SHALL include a sliding memory window (minimum last 5 exchanges) when retrieving documents and building prompts so follow-up questions can reference earlier answers.

#### Scenario: Follow-up question references prior answer
- **GIVEN** a user asked “What is the cable bend requirement?” and the assistant answered using document sources
- **AND** the user immediately asks “How does that change for fiber jumpers?”
- **WHEN** the assistant constructs the retrieval prompt
- **THEN** the previous Q/A pair is included in context (subject to the memory window limit)
- **AND** the generated answer correctly interprets “that” as the prior cable bend requirement without the user restating it.

#### Scenario: Memory truncated after limit
- **GIVEN** more than the configured number of exchanges have occurred in one session
- **WHEN** a new message is sent
- **THEN** the oldest exchanges beyond the window are dropped from the prompt context
- **AND** the UI still shows the full visible history so users can resend any turn even if it falls outside the model memory window.
