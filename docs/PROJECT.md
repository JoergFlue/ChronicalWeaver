AIStudio: Chronicle Weaver - Detailed Briefing Document
1. Executive Summary
The "Chronicle Weaver" project aims to develop a user-friendly, AI-driven roleplaying assistant for Windows 11. It is designed to provide rich, continuous roleplay experiences through a modular agent system, flexible Large Language Model (LLM) backends, robust memory management, and integrated image generation. The project prioritizes modularity, scalability, user-centric design, robust testing (including Test-Driven Development), comprehensive documentation, and iterative development. The application, tentatively named "Chronicle Weaver," will feature a tabbed UI for core functionalities: "Roleplay," "Agents," "Library," and "Settings."

2. Core Project Goals & Principles
The overarching goal is to create a robust, maintainable, and user-friendly AI roleplaying application. This is guided by several core principles:

Modularity: Breaking down the system into independent, reusable components with clearly defined interfaces. "Each component (LLM Manager, Memory Manager, Agent classes, UI components) should be in its own file/directory with clearly defined interfaces."
Scalability: Designing the system to easily integrate new LLMs, agents, and features in the future.
User-Centric Design: Focusing on an intuitive UI for agent management, content library, and immersive roleplay, as detailed in the UI wireframe.
Robust Testing: Implementing Test-Driven Development (TDD) principles, including unit, integration, and end-to-end tests, to ensure reliability and prevent regressions. "The 'Comprehensive Development Plan' places a much stronger emphasis on robust testing, listing it as a core principle."
Documentation: Maintaining clear internal (code comments, docstrings) and external (user guide, developer guide) documentation.
Iterative Development: Embracing feedback and continuous improvement through scheduled retrospectives after each major phase.
3. Technology Stack
The project leverages a refined and specific technology stack for optimal performance and maintainability:

Frontend UI: PyQt6 for a robust, native-looking, and Pythonic desktop application. While PySide6 was also considered, PyQt6 is the definitive choice.
Backend Framework/Orchestration: CrewAI for agentic workflow and inter-agent communication. LangChain is also utilized for LLM wrappers and memory components.
LLM Integration Abstraction: LiteLLM serves as a unified API to seamlessly connect with various LLM backends including OpenAI, Google Gemini, and local models like Ollama and LM Studio. "The 'Comprehensive Development Plan' explicitly proposes LiteLLM as a unified abstraction layer for various LLM backends."
Memory/Database:SQLite for structured data such as prop items, clothing, agent configurations, and summarized interactions. It's chosen for its simplicity and file-based nature.
LangChain's ConversationSummaryBufferMemory or ConversationBufferWindowMemory for conversational memory, backed by SQLite.
Image Generation:API-based: OpenAI DALL-E 3 API and Stability AI API (requiring API keys).
Local: Integration with Automatic1111 API or ComfyUI API, if the user has a local setup.
Development Language: Python 3.11+.
Packaging: PyInstaller for creating a standalone Windows 11 executable.
Version Control: Git.
4. Agent System & Configuration
A core feature of Chronicle Weaver is its sophisticated agent system, allowing for specialized AI behaviors and roles.

4.1. Agent Types and Functionality
Main Agent: The primary point of contact for the user's direct input and handles core roleplay interactions. It "handles user input/output," "Routes tasks to sub-agents," and "Maintains session context."
Sub-Agents: Specialized agents that the Main Agent can delegate tasks to. Examples include:
Prompt Tracking Agent: Logs user and assistant prompts and responses.
Continuity Check Agent: Periodically reviews recent conversation against long-term memory for inconsistencies and suggests corrections.
Search Agent: Integrates with web search APIs (e.g., SerpApi) to retrieve factual information relevant to the roleplay or user queries.
Alternate Sub-Ego Agent: Allows defining multiple distinct personalities/roles for the assistant, enabling dynamic personality shifts based on context or user command.
Prop Agent / Access Prop/Clothing Library: Manages and allows agents to "equip" items, influencing descriptions or image generation prompts.
4.2. Agent Configuration Interface ("Agents" Tab)
The "AIStudio Agent Configuration Interface" and "Copilot Agent Configuration" templates outline comprehensive settings for each agent, accessible via the "Agents" tab in the UI:

Basic Information:Name: Unique identifier (e.g., "Lore Keeper," "Creative Muse").
Role: Concise description of primary function (e.g., "Manages long-term narrative continuity").
Description / Backstory: Defines the agent's personality, historical context, and foundational knowledge. This can dynamically generate the LLM's system prompt. Example: "A diligent archivist with a vast memory for historical events and character details. Prioritizes consistency and factual accuracy within the narrative."
Avatar: Visual representation in the UI.
Core AI Configuration:System Prompt / Persona Definition: Foundational instruction set passed to the LLM, defining personality, tone, rules, and directives. Example: "You are a helpful and creative roleplaying assistant. Your goal is to guide the user through an imaginative story, maintaining consistency and responding in character."
Preferred LLM Model: Allows selecting a specific LLM model (e.g., GPT-4o, Llama3) for the agent or using the "Global Default."
Temperature (Creativity): Controls randomness of output (0.0 to 1.0, default 0.7). Higher values are more creative but less coherent.
Max Output Tokens: Limits the length of the generated response (default 500).
Agentic Capabilities & Tools:Enabled Tools: Checkbox list for external functions like "Web Search," "Image Generation," "Access Prop/Clothing Library," "Read/Write Long-Term Memory," and "Calculate."
Sub-Agent Delegation: Defines which other configured sub-agents this agent can delegate tasks to.
Memory Access Level: Controls the agent's interaction with memory systems (Full Read/Write, Read Only, Short-Term Conversation Only, No Memory Access).
Status & Control:Enabled: Toggle to activate or deactivate the agent.
Is Main Interface Agent: Designates the primary point of contact (only one).
Default Activity Level: How frequently or proactively a sub-agent intervenes (High, Medium, Low, Passive).
Advanced Settings (Optional): Logging Level, JSON/YAML Export/Import for configuration sharing.
5. Memory System
The application employs a dual-layered memory system to ensure narrative continuity:

Short-Term Conversation Memory: Utilizes LangChain's ConversationBufferWindowMemory to store the immediate conversation context. This maintains the flow of the current session.
Long-Term Continuity Memory: Stored persistently in an SQLite database. This includes:
Past interaction summaries (using ConversationSummaryBufferMemory to condense long conversations).
Character states (traits, clothing, relationships).
Plot points and world events.
Agent configurations and prop/clothing library details.
The MainAgent will access and update short-term memory during a conversation and consult long-term memory for continuity cues.
6. Image Generation
Integrated image generation enhances the roleplaying experience:

Integration: Abstracted interface for DALL-E 3, Stability AI APIs, and local Automatic1111/ComfyUI APIs.
Functionality: Users can trigger image generation via a dedicated "Generate Image" button in the "Roleplay" tab. Images can be generated based on current roleplay context or a specific user-provided prompt.
Display: Generated images will appear inline within the conversation history or in a dedicated viewer.
Image Library: Generated images will be stored with metadata, potentially linking to prop/clothing entries.
7. User Interface (UI) Design (Chronicle Weaver)
The UI is designed to be intuitive and organized, featuring a main window with a tabbed interface:

7.1. "Roleplay" Tab (Main View)
Main Conversation Display Area: Scrollable multi-line text area showing conversation history, with clear differentiation between user input and assistant responses. "Image Integration: Placeholder for generated images to appear inline within the conversation history, or a clickable link to view in a popup."
Bottom Input & Action Area: Includes a user text input field, a "Send" button, and a "Generate Image" button.
Status/Feedback Bar: Small text area for brief updates like "AI is thinking..." or "Image generated!".
7.2. "Agents" Tab
Left Panel (Agent List): Displays active and inactive agents with enable/disable toggles. Buttons for "New Agent" and "Delete Selected Agent."
Right Panel (Agent Configuration): Form for configuring the selected agent, including Name, Role/Description, Base System Prompt, and Associated LLM (if specific).
7.3. "Library" Tab
Left Panel (Item Categories/Filters): Optional list for categories like "All Items," "Clothing," "Props," "Locations."
Middle Panel (Item List): List/table of available items with Name, Category, and a small preview image. Buttons for "Add New Item" and "Delete Selected Item."
Right Panel (Item Details): Displays details for the selected item, including Name, Description, Category, Image File Path, and an Image Preview.
7.4. "Settings" Tab
LLM Backend Configuration: Dropdown to select the active LLM backend (OpenAI, Gemini, Ollama, LM Studio). Dynamic input fields appear based on selection (e.g., API Key, Local Host URL, Default Model Name). Includes sliders/inputs for "Temperature (Creativity)" and "Max Tokens (Response Length)."
Image Generation Settings: Dropdown for image API provider (DALL-E 3, Stability AI, Automatic1111) and relevant API key input.
General Application Settings: Options for theme (Light/Dark) and data storage location.
8. Development Plan Phases
The project will follow a detailed, iterative, and phased development approach:

Phase 0: Planning & Foundation (2-4 Days): Finalize requirements, set up development environment, establish Git repository, define architecture, and initialize documentation.
Phase 1: Core LLM & Main Agent System (2-3 Weeks): Implement LiteLLM wrapper for various LLMs, basic CrewAI MainAgent, and a minimal text-based PyQt6 UI for roleplay.
Phase 2: Memory & Data Persistence (2-3 Weeks): Implement short-term (LangChain) and long-term (SQLite) memory systems, with CRUD operations for agent configurations and prop/clothing items. Integrate memory with the Main Agent.
Phase 3: Agent Management & Core Sub-Agents (3-4 Weeks): Develop the comprehensive "Agents" tab UI for agent configuration, refine the Main Agent to dynamically utilize sub-agents, and implement basic "Prompt Tracking" and "Continuity Check" agents.
Phase 4: Image Generation & Advanced Features (3-4 Weeks): Integrate various image generation APIs (DALL-E 3, Stability AI, Automatic1111) with UI integration. Implement "Search Agent" and "Alternate Sub-Ego Agent." Integrate the "Library" tab UI for prop/clothing management.
Phase 5: Polish, Comprehensive Testing & Deployment (2-3 Weeks): Refine UI/UX, conduct exhaustive unit, integration, and End-to-End (E2E) testing (potentially using Playwright/Selenium for PyQt UI), prepare user and developer documentation, and package the application using PyInstaller for Windows 11 distribution. Performance profiling and bug fixing will be ongoing.
9. Cross-Cutting Engineering Practices
Throughout all phases, strict engineering practices will be adhered to:

Modular Design: Ensuring independent components with clear interfaces.
Version Control (Git): Regular commits, feature branching.
Testing: Comprehensive unit, integration, and E2E tests, with Continuous Integration (CI) setup. "Testing tasks are integrated into almost every phase."
Code Review: (Recommended) Peer review for quality and bug identification.
Documentation: Inline comments, docstrings, and external user/developer guides.
Error Handling: Robust try-except blocks and clear messages.
Logging: Utilizing Python's logging module for tracking and debugging.
Dependency Management: Using pipenv or poetry for reproducible environments.
Retrospectives: Scheduled meetings after major phases for reflection and adaptation.
This detailed plan provides a robust framework for developing Chronicle Weaver, ensuring a high-quality, feature-rich, and maintainable AI roleplaying assistant.
