"""LLM prompt templates for ASM analysis and CoT generation."""

ASM_COT_SYSTEM_PROMPT = """You are an expert 65816 SNES assembly analyst specializing in Zelda: A Link to the Past.
Your task is to analyze assembly code samples and generate detailed, step-by-step reasoning about what the code does and why.

Focus on:
1. The purpose of the code based on the instruction/question
2. Memory addressing modes and SNES hardware interactions
3. Register usage patterns (A, X, Y, Direct Page, Stack)
4. Cycle efficiency considerations
5. Bank handling and long addressing ($00-$FF banks)
6. DMA, VRAM, OAM, and PPU interactions where relevant
7. Game-specific patterns (sprites, Link state, dungeon logic)

Provide reasoning that would help a learner understand the code deeply.
Be concise but thorough. Focus on teaching understanding, not just describing."""


ASM_ANALYSIS_TEMPLATE = """Analyze this 65816 SNES assembly sample:

## Task/Instruction
{instruction}

## Additional Context
{input}

## Assembly Code
```asm
{output}
```

---

Provide step-by-step reasoning about:
1. What is being asked/accomplished?
2. How does the code achieve this?
3. What SNES-specific patterns are used?
4. What are the key memory/register operations?
5. Are there optimization considerations?

Format your response as clear, structured reasoning steps."""


ASM_CODE_ONLY_TEMPLATE = """Analyze this 65816 SNES assembly routine:

```asm
{code}
```

Provide detailed reasoning about:
1. Purpose and overall function
2. Key instructions and their effects
3. Memory and register flow
4. SNES hardware interactions (if any)
5. Potential optimizations or concerns

Be concise but thorough."""


ASM_OPTIMIZATION_TEMPLATE = """Analyze the optimization approach in this assembly code:

## Original Task
{instruction}

## Optimized Code
```asm
{output}
```

Explain:
1. What optimization techniques are applied?
2. How many cycles are saved (estimate)?
3. What trade-offs were made?
4. Are there further optimization opportunities?"""


ASM_DEBUG_TEMPLATE = """Analyze this debugging scenario:

## Problem Description
{instruction}

## Context
{input}

## Code Under Analysis
```asm
{output}
```

Provide debugging reasoning:
1. What could cause the described issue?
2. What do the register/memory operations reveal?
3. What are the likely bug locations?
4. How would you fix this?"""


ASM_HOOK_TEMPLATE = """Analyze this ROM hook/patch:

## Hook Purpose
{instruction}

## Implementation
```asm
{output}
```

Explain:
1. Where does this hook intercept?
2. How does it preserve game state?
3. What does the custom code accomplish?
4. What are the safety considerations?"""


def get_prompt_template(domain: str) -> str:
    """Get the appropriate prompt template for a domain."""
    templates = {
        "asm": ASM_ANALYSIS_TEMPLATE,
        "asm_optimize": ASM_OPTIMIZATION_TEMPLATE,
        "asm_debug": ASM_DEBUG_TEMPLATE,
        "asm_hook": ASM_HOOK_TEMPLATE,
    }
    return templates.get(domain, ASM_ANALYSIS_TEMPLATE)


def build_analysis_prompt(
    instruction: str,
    output: str,
    input_text: str = "",
    domain: str = "asm",
) -> str:
    """Build a complete analysis prompt for a sample."""
    template = get_prompt_template(domain)

    if "{code}" in template:
        return template.format(code=output)

    return template.format(
        instruction=instruction,
        input=input_text or "(No additional context)",
        output=output,
    )
