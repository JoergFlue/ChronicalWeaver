# Chronicle Weaver Implementation Plan

This plan breaks down the Chronicle Weaver project into modular, testable steps. Each section begins with acceptance criteria to guide development and testing.

---

## Phase 0: Planning & Foundation

### Acceptance Tests
- The repository is initialized with a clear directory structure.
- A README and initial documentation exist.
- The development environment is reproducible (pipenv/poetry).
- Basic CI pipeline runs lint and test jobs.

### Implementation Steps
1. Initialize Git repository and set up feature branching.
2. Create project directories for UI, agents, memory, LLM, images, tests, and docs.
3. Add README.md and PROJECT.md.
4. Set up Python 3.11+ environment with pipenv/poetry.
5. Add initial requirements (PyQt6, CrewAI, LangChain, LiteLLM, etc.).
6. Configure CI for linting and running tests.
7. Add basic documentation structure.

---

## Phase 1: Core LLM & Main Agent System

### Acceptance Tests
- The LiteLLM wrapper can connect to at least one LLM backend and return responses.
- The MainAgent class can send/receive messages and route tasks.
- Minimal PyQt6 UI allows user input and displays agent responses.
- Unit tests cover LLM wrapper and MainAgent logic.

### Implementation Steps
1. Implement LiteLLM wrapper for OpenAI (mock for others).
2. Create MainAgent class with input/output and routing logic.
3. Build minimal PyQt6 UI: text input, send button, conversation display.
4. Integrate MainAgent with UI.
5. Write unit tests for LLM wrapper and MainAgent.

---

## Phase 2: Memory & Data Persistence

### Acceptance Tests
- Short-term memory (ConversationBufferWindowMemory) stores session context.
- Long-term memory (SQLite) persists summaries, agent configs, props, etc.
- CRUD operations for agents and props work via UI and backend.
- Unit/integration tests cover memory logic and persistence.

### Implementation Steps
1. Integrate LangChain's short-term memory with MainAgent.
2. Set up SQLite schema for long-term memory (agents, props, summaries).
3. Implement CRUD for agents and props (backend and UI).
4. Connect MainAgent to both memory layers.
5. Write tests for memory and persistence.

---

## Phase 3: Agent Management & Core Sub-Agents

### Acceptance Tests
- "Agents" tab UI lists, enables/disables, and configures agents.
- Sub-agents (Prompt Tracking, Continuity Check) operate as described.
- MainAgent delegates tasks to sub-agents.
- Tests verify agent configuration, delegation, and sub-agent logic.

### Implementation Steps
1. Build "Agents" tab UI: list, enable/disable, config form.
2. Implement Prompt Tracking and Continuity Check agents.
3. Add delegation logic to MainAgent.
4. Connect agent config UI to backend.
5. Write tests for agent management and sub-agent behaviors.

---

## Phase 4: Image Generation & Advanced Features

### Acceptance Tests
- Image generation APIs (DALL-E 3, Stability AI, local) are abstracted and callable.
- "Generate Image" button in UI triggers image creation.
- Images display inline or in viewer; metadata is stored.
- "Library" tab UI manages props/clothing with images.
- Tests cover image generation, display, and library CRUD.

### Implementation Steps
1. Implement image generation abstraction for APIs.
2. Add "Generate Image" button and display logic to UI.
3. Store images and metadata in SQLite.
4. Build "Library" tab UI for item management.
5. Write tests for image and library features.

---

## Phase 5: Polish, Testing & Deployment

### Acceptance Tests
- UI/UX is refined and consistent.
- Comprehensive unit, integration, and E2E tests pass.
- Documentation is complete and up-to-date.
- PyInstaller produces a working Windows 11 executable.
- CI pipeline runs all tests and builds.

### Implementation Steps
1. Refine UI/UX and fix bugs.
2. Add E2E tests (Playwright/Selenium for PyQt UI).
3. Complete user and developer documentation.
4. Package app with PyInstaller.
5. Ensure CI runs all tests and builds.

---

## Cross-Cutting Engineering Practices
- Modular design: Each component in its own file/directory.
- Version control: Regular commits, feature branches.
- Testing: Unit, integration, E2E, CI.
- Code review: Peer review recommended.
- Documentation: Inline and external.
- Error handling: Robust try/except, clear messages.
- Logging: Python logging module.
- Dependency management: pipenv/poetry.
- Retrospectives: After major phases.
