## 1. Netlify deployment scaffolding
- [ ] Add `netlify.toml` with build command to install Python deps, start Streamlit on `$PORT`, and configure data/index locations.
- [ ] Provide a start script or container entrypoint that binds to `0.0.0.0`, disables CORS as needed, and surfaces health/log output for Netlify.

## 2. Versioning and metadata
- [ ] Surface Git commit/tag and Netlify deploy ID in the UI via environment variables or a small build-time metadata file.
- [ ] Add a build step to write version metadata into the deployment artifact for debugging and support.

## 3. Git-based release packaging
- [ ] Create a release build target (e.g., Makefile or script) that produces a zip/tarball with app code, `netlify.toml`, and setup instructions.
- [ ] Configure CI (Netlify build hook or VCS pipeline) to produce tagged release artifacts alongside preview/production deploys.

## 4. Validation
- [ ] Run a Netlify-style build locally (e.g., `netlify dev` or container with `$PORT` and `OPENAI_API_KEY`) to confirm the app boots and serves responses.
- [ ] Verify preview deploys are isolated from production data/index files and that version info updates per commit/tag.
