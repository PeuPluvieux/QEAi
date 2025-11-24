## ADDED Requirements
### Requirement: Netlify Git Deployments
The system SHALL provide configuration to deploy the Streamlit app to Netlify directly from the Git repository with environment-driven secrets and declared data/index paths.

#### Scenario: Deploy from connected repository
- **WHEN** the repository is connected to Netlify with `OPENAI_API_KEY` configured and a build is triggered
- **THEN** the Netlify build uses the repo configuration to install dependencies, start the Streamlit server on `$PORT`, and serve it at the assigned Netlify URL.

#### Scenario: Preview deploy per commit
- **WHEN** a pull request build runs on Netlify
- **THEN** the app is deployed to an isolated preview URL tied to the PR commit so changes can be reviewed without affecting production.

### Requirement: Versioned Releases
The system SHALL stamp each deploy with Git metadata so users can identify the running version.

#### Scenario: Surface build info in UI
- **WHEN** the app renders
- **THEN** it displays the Git tag/commit and Netlify deploy ID pulled from environment variables or generated build metadata.

#### Scenario: Versioned download artifact
- **WHEN** a Git tag is pushed
- **THEN** CI produces a downloadable release archive labeled with the tag and includes deployment instructions.

### Requirement: Local and offline install path
The system SHALL provide an install script that mirrors the Netlify build locally for users who prefer running outside Netlify.

#### Scenario: Local install from release
- **WHEN** a user downloads a release archive
- **THEN** running the provided setup command installs dependencies, sets required environment variables, and starts the app using the same port/bind settings as the Netlify deployment.
