# Change: Enable Netlify + Git versioned deployments

## Why
Teams want to run the Streamlit RAG assistant as a deployable app through Netlify with Git-driven releases so each update becomes a traceable version without manual setup.

## What Changes
- Add Netlify deployment scaffolding (configuration, start script, environment expectations) so the Streamlit server can run behind a Netlify site.
- Stamp builds with Git metadata and expose the running version in the UI and release artifacts.
- Document prerequisites (OPENAI_API_KEY, data/index storage) and preview/production deploy flows for Netlify-connected repos.

## Impact
- Affected specs: `deployment`
- Affected code: `netlify.toml`, `app.py` (version display and runtime config), deployment scripts/Dockerfile, CI or Netlify build settings for releases
