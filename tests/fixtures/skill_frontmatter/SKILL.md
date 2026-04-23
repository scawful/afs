---
name: gemini-work
triggers: ["gemini-cli", "agent studio"]
requires: ["knowledge/work", "gemini mcp"]
profiles: ["work", "general"]
enforcement:
  - Keep the Gemini workspace flow deterministic.
  - Prefer MCP-backed context over ad hoc summaries.
verification:
  - Run the workspace health checks before handoff.
---

# Skill Fixture
