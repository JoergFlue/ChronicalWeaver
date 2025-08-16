# Phase 1: Core LLM & Main Agent System - Detailed Plan

## Implementation Confidence: 8/10

### Potential Problems
- LiteLLM wrapper integration with multiple LLMs may require custom adapters.
- CrewAI orchestration may have undocumented edge cases.
- PyQt6 UI event handling and threading (for async LLM calls).

### Acceptance Tests
- LiteLLM wrapper connects to OpenAI and returns valid responses.
- MainAgent class sends/receives messages, routes tasks, and maintains session context.
- Minimal PyQt6 UI allows user input, displays agent responses, and handles errors.
- Unit tests cover LLM wrapper, MainAgent logic, and UI event flow.

### Step-by-Step Implementation
1. Implement LiteLLM wrapper for OpenAI (mock for Gemini, Ollama, LM Studio).
2. Create MainAgent class: input/output, routing, session context.
3. Build minimal PyQt6 UI: text input, send button, conversation display, error feedback.
4. Integrate MainAgent with UI, handle async LLM calls (threading or asyncio).
5. Write unit tests for LiteLLM wrapper, MainAgent, and UI event flow.
6. Document LLM integration and agent architecture.
7. Verify all acceptance tests pass.

---

# Phase 1 Risks & Mitigations
- **Risk:** LLM API changes or rate limits. **Mitigation:** Use mock responses and document API keys/config.
- **Risk:** UI freezes on long LLM calls. **Mitigation:** Use threading/async for LLM requests.
- **Risk:** CrewAI orchestration complexity. **Mitigation:** Start with simple agent logic, expand iteratively.
