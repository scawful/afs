# Triforce MoE - Expansion Research

_Research proposal for expanding the Triforce Mixture of Experts system_
_Last updated: 2025-01-02_

## Executive Summary

The Triforce MoE system currently consists of three specialized 65816 assembly experts named after Zelda's creation goddesses:

| Model | Intent | Specialty | Base |
|-------|--------|-----------|------|
| **Din** | Optimization | Code size/cycle reduction | Qwen 7B/14B + LoRA |
| **Nayru** | Generation | Assembly code writing | Qwen 7B + LoRA |
| **Farore** | Debugging | Bug finding and fixing | Qwen 7B (planned) |

This document explores expansion options including new expert models, larger parameter sizes, and architectural improvements to the MoE routing system.

---

## 1. Veran - ROM Analysis Expert

### Character Context

Veran is the Sorceress of Shadows from Oracle of Ages - a demon fairy with powers of **possession**, **time manipulation**, and **shapeshifting**. She infiltrates hosts to control them from within and reveals hidden truths.

### Proposed Specialization: Reverse Engineering & ROM Analysis

Veran's thematic abilities map naturally to ROM analysis tasks:

| Veran Power | ROM Analysis Analog |
|-------------|---------------------|
| Possession | Deep dive into binary, understanding code from "inside" |
| Time manipulation | Tracing execution history, understanding call stacks |
| Shapeshifting | Recognizing code patterns across different forms |
| Revealing hidden truth | Finding undocumented routines, Easter eggs, compression |

### Proposed Intents

```python
class QueryIntent(str, Enum):
    # ... existing ...
    ANALYSIS = "analysis"       # veran - ROM analysis, reverse engineering
```

### Pattern Keywords for Classifier

```python
ANALYSIS_PATTERNS = {
    # Reverse engineering
    r"\breverse\s+engineer": 1.0,
    r"\bdisassembl(e|y)": 1.0,
    r"\bdecompile\b": 0.9,
    r"\bwhat\s+does\s+this\s+(code|routine|function)\s+do\b": 1.0,

    # Pattern recognition
    r"\bidentify\b": 0.7,
    r"\brecognize\b": 0.7,
    r"\bpattern\b": 0.8,
    r"\bsimilar\s+to\b": 0.8,
    r"\blike\s+the\s+vanilla\b": 0.9,

    # ROM structure
    r"\bfind\s+(the|a)\s+(routine|function|address)\b": 0.9,
    r"\blocate\b": 0.7,
    r"\bwhere\s+is\b": 0.8,
    r"\bROM\s+(map|structure|layout)\b": 1.0,

    # Binary analysis
    r"\bhex\s+(dump|pattern)\b": 0.8,
    r"\bbyte\s+sequence\b": 0.8,
    r"\bcompression\b": 0.9,
    r"\bencryption\b": 0.8,
}
```

### Training Data Sources

1. **Disassembly annotations** - Documented ALTTP disassembly with comments
2. **Pattern libraries** - Common 65816 idioms, Nintendo patterns
3. **Cross-reference data** - How routines call each other
4. **ROM maps** - Memory layouts, bank structures

### Model Architecture

| Config | Details |
|--------|---------|
| Base model | Qwen 2.5 Coder 7B or 14B |
| Fine-tuning | LoRA r=64, alpha=128 |
| Training samples | ~5,000 analysis Q&A pairs |
| Focus | Understanding existing code, not generating new code |

---

## 2. Additional Zelda Character Experts

### 2.1 Ganon/Ganondorf - Vulnerability Analysis

**Thematic Fit:** The King of Evil seeks to exploit weaknesses and break systems.

**Proposed Specialty:** Security analysis, edge case detection, exploit finding

```python
VULNERABILITY_PATTERNS = {
    r"\bvulnerab(le|ility)\b": 1.0,
    r"\bexploit\b": 1.0,
    r"\bbuffer\s+overflow\b": 1.0,
    r"\bedge\s+case\b": 0.9,
    r"\brace\s+condition\b": 0.9,
    r"\bout\s+of\s+bounds\b": 1.0,
    r"\bstack\s+(corruption|smash)\b": 1.0,
    r"\buninitialized\b": 0.8,
    r"\bcrash\s+(the\s+)?(game|system)\b": 0.9,
}
```

**Use Cases:**
- Finding ACE (Arbitrary Code Execution) vectors
- Detecting unintended behavior in hooks
- Stress testing code paths
- ROM corruption analysis

### 2.2 Zelda - Architecture & Planning

**Thematic Fit:** The Princess of Wisdom guides and orchestrates from above.

**Proposed Specialty:** High-level design, code organization, project planning

```python
ARCHITECTURE_PATTERNS = {
    r"\barchitect(ure)?\b": 1.0,
    r"\bdesign\b": 0.8,
    r"\bstruct(ure)?\b": 0.7,
    r"\borganiz(e|ation)\b": 0.8,
    r"\bmodular\b": 0.9,
    r"\brefactor\b": 0.9,
    r"\bseparate\s+concerns\b": 1.0,
    r"\bhow\s+should\s+I\s+organiz(e)\b": 1.0,
    r"\bbest\s+practice\b": 0.8,
}
```

**Use Cases:**
- ROM hack project structure
- Bank allocation strategies
- Hook organization
- Memory layout planning

### 2.3 Link - Integration & Bridging

**Thematic Fit:** The Hero connects disparate elements, bridging worlds and completing quests.

**Proposed Specialty:** Integration testing, cross-module communication, API design

```python
INTEGRATION_PATTERNS = {
    r"\bintegrat(e|ion)\b": 1.0,
    r"\bconnect\b": 0.7,
    r"\bbridge\b": 0.8,
    r"\binterface\b": 0.8,
    r"\bcompatib(le|ility)\b": 0.9,
    r"\bhook\s+into\b": 0.9,
    r"\bcall\s+from\b": 0.7,
    r"\bwork\s+with\b": 0.6,
    r"\bcombine\b": 0.7,
}
```

**Use Cases:**
- Coordinating multiple patches
- Ensuring hooks don't conflict
- Testing cross-bank jumps
- Validating save file compatibility

### 2.4 Impa - Protection & Security

**Thematic Fit:** Zelda's guardian and Sheikah protector.

**Proposed Specialty:** Defensive coding, anti-crash protection, validation

```python
PROTECTION_PATTERNS = {
    r"\bprotect\b": 0.9,
    r"\bsafe(ty|guard)?\b": 0.8,
    r"\bvalidat(e|ion)\b": 0.9,
    r"\bsanity\s+check\b": 1.0,
    r"\bbounds\s+check\b": 1.0,
    r"\bnull\s+check\b": 0.9,
    r"\bdefensive\b": 0.9,
    r"\bprevent\s+(crash|corruption)\b": 1.0,
}
```

**Use Cases:**
- Adding safety checks to hooks
- Validating user input
- Preventing save corruption
- Safe default behaviors

### 2.5 Tingle - Mapping & Navigation

**Thematic Fit:** The eccentric mapmaker who charts unexplored territory.

**Proposed Specialty:** Code coverage, navigation, documentation

```python
MAPPING_PATTERNS = {
    r"\bmap\b": 0.7,
    r"\bchart\b": 0.7,
    r"\bcoverage\b": 0.9,
    r"\bdocument\b": 0.8,
    r"\bexplain\b": 0.7,
    r"\bwalkthrough\b": 0.8,
    r"\bflow\b": 0.7,
    r"\bcall\s+(graph|tree)\b": 1.0,
    r"\btrace\b": 0.8,
}
```

**Use Cases:**
- Generating call graphs
- Documenting ROM regions
- Creating execution traces
- Mapping memory usage

---

## 3. Parameter Size Scaling

### Current State

The Triforce models use 7B parameter bases with LoRA adapters:

| Model | Base | LoRA Rank | Trainable Params | VRAM (Q4) |
|-------|------|-----------|------------------|-----------|
| Din v3 | Qwen 2.5 Coder 7B | r=32 | ~26M | 4-5 GB |
| Din v4 | Custom fused | N/A | Full model | 4-5 GB |
| Nayru v5 | Qwen 2.5 Coder 7B | r=64 | ~52M | 4-5 GB |

### Scaling Options

Based on [2025 VRAM requirements research](https://localllm.in/blog/ollama-vram-requirements-for-local-llms):

| Model Size | VRAM (FP16) | VRAM (Q4_K_M) | Speed (tok/s) | Quality |
|------------|-------------|---------------|---------------|---------|
| 7B | 14-15 GB | 4-5 GB | 45-55 | Good |
| 14B | 28 GB | 10-12 GB | 35-40 | Better |
| 32B | 64 GB | 16-24 GB | 15-20 | Much Better |
| 70B | 140+ GB | 40-48 GB | 8-12 | Excellent |

### Recommendations by Use Case

**Development Machine (RTX 4090, 24GB):**
- Primary: 14B models at Q4_K_M for best quality/speed balance
- Fallback: 7B at Q8 for maximum speed
- Maximum: 32B at Q4_K_M with reduced context

**Inference Server (48GB+ A6000):**
- Optimal: 32B at Q4_K_M with full context
- Quality: 70B at Q4_K_M with careful memory management

**Consumer Hardware (16GB):**
- Stick with 7B models at Q4_K_M
- Multiple experts can share VRAM when not running concurrently

### Quality vs Speed Tradeoffs

```
Quality Score (relative to 70B baseline)
    |
1.0 |                                    * 70B
    |                              * 32B
0.9 |                        * 14B
    |                  * 7B
0.8 |
    |
0.7 |
    +-------------------------------------> Tokens/sec
         10    20    30    40    50
```

Research from [Databricks](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms) and [Modal](https://modal.com/blog/how-much-vram-need-fine-tuning) indicates that for domain-specific tasks like 65816 assembly:
- 7B with good LoRA can match 14B base on in-domain tasks
- 14B with LoRA approaches 32B for specialized knowledge
- Diminishing returns above 32B for narrow domains

### Proposed Scaling Strategy

**Phase 1 (Current):** 7B with LoRA
- Din, Nayru, Farore all on 7B bases
- Quick iteration, low resource usage
- Good for development and testing

**Phase 2:** Selective 14B upgrade
- Upgrade Veran (analysis) to 14B - benefits from more reasoning
- Upgrade Zelda (architecture) to 14B - needs broader context
- Keep optimization/generation at 7B - narrow tasks

**Phase 3:** Quality tier system
```python
@dataclass
class ExpertConfig:
    name: str
    model_id: str
    tier: Literal["fast", "balanced", "quality"]

TIER_CONFIGS = {
    "fast": {"model": "qwen2.5-coder:7b", "quantization": "Q4_K_M"},
    "balanced": {"model": "qwen2.5-coder:14b", "quantization": "Q4_K_M"},
    "quality": {"model": "qwen2.5-coder:32b", "quantization": "Q4_K_M"},
}
```

---

## 4. Specialized vs Generalist Tradeoffs

### Current MoE Architecture

The Triforce MoE uses a **sparse expert selection** model:

```
User Query
    |
    v
[Intent Classifier] -- keyword-based routing
    |
    +---> din (optimization)
    +---> nayru (generation)
    +---> farore (debugging)
    +---> fallback (general)
```

### MoE Routing Mechanisms (2025)

Based on [recent MoE research](https://huggingface.co/blog/moe):

**TopK Routing (Current approach):**
- Route to K experts (K=1 in Triforce)
- Simple, deterministic
- Risk: Expert imbalance, underutilized experts

**ReLU Routing:**
- Continuous activation, not discrete selection
- Variable number of experts per token
- More flexible compute allocation

**Load Balancing:**
> "If all tokens are sent to just a few popular experts, that will make training inefficient."

The Triforce classifier should track routing statistics:

```python
@dataclass
class RoutingStats:
    total_queries: int
    expert_distribution: dict[str, int]
    intent_confidence_avg: dict[str, float]
    fallback_rate: float
```

### Many Small Specialists vs Few Large Generalists

| Approach | Pros | Cons |
|----------|------|------|
| **Many small** (7B x 6) | Fast switching, targeted expertise, modular training | More routing complexity, potential gaps |
| **Few large** (32B x 2) | Broader knowledge, fewer handoffs | Slower, expensive, harder to update |
| **Hybrid** (14B x 3) | Balanced tradeoff | Moderate complexity |

### Recommended Approach

**Hierarchical MoE with shared backbone:**

```
                    [Shared Backbone - 14B]
                           |
        +------------------+------------------+
        |                  |                  |
   [Optimization]    [Generation]      [Analysis]
    LoRA (Din)       LoRA (Nayru)     LoRA (Veran)
```

Benefits:
- Single large model loaded in memory
- Hot-swap LoRA adapters per task (~100MB each)
- Combine knowledge from shared backbone
- Fast switching between experts

Implementation with Ollama:

```bash
# Load base model once
ollama run qwen2.5-coder:14b --keep-alive 1h

# Swap adapters
ollama run din-v4:14b --lora-path ~/.afs/adapters/din.gguf
ollama run nayru-v5:14b --lora-path ~/.afs/adapters/nayru.gguf
```

---

## 5. Training Considerations

### LoRA vs Full Fine-tuning

From [recent research (2024-2025)](https://arxiv.org/abs/2410.21228):

> "LoRA and full fine-tuning yield weight matrices whose singular value decompositions exhibit very different structure."

**Key findings:**
- LoRA matches full fine-tuning quality for small, focused datasets
- LoRA adapters develop "intruder dimensions" not present in full fine-tuning
- Best LoRA performance requires 10x higher learning rate than full fine-tuning
- LoRA is optimal when dataset size ~ trainable parameter count

### Implications for Triforce

Given our training data sizes:
- Din: ~3,000 optimization examples
- Nayru: ~5,000 generation examples
- Farore: ~1,500 debugging examples (planned)

LoRA with r=32-64 (~26-52M params) is well-matched to our data scale.

For Veran (analysis), we may need:
- Larger rank (r=128) for complex reasoning
- More training data (~10,000 examples)
- Consider 14B base for improved context understanding

### Training Pipeline Updates

```python
# Proposed configuration for Veran
veran_config = LoRAConfig(
    base_model="qwen2.5-coder:14b",
    r=128,
    alpha=256,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    learning_rate=1e-4,  # 10x typical
    num_epochs=3,
    batch_size=4,
)
```

---

## 6. Implementation Roadmap

### Phase 1: Veran Integration (Q1 2025)

- [ ] Add `QueryIntent.ANALYSIS` to classifier
- [ ] Create analysis pattern keywords
- [ ] Collect training data from ALTTP disassembly
- [ ] Train Veran LoRA adapter on 7B base
- [ ] Integrate into orchestrator

### Phase 2: Scaling Experiments (Q2 2025)

- [ ] Test 14B base for Veran
- [ ] Benchmark quality vs 7B
- [ ] Evaluate shared backbone approach
- [ ] Measure VRAM usage patterns

### Phase 3: Extended Expert Pool (Q3 2025)

- [ ] Evaluate Ganon (vulnerability) need
- [ ] Prototype Zelda (architecture) expert
- [ ] Load balancing improvements
- [ ] Multi-expert ensemble for complex queries

### Phase 4: Production Optimization (Q4 2025)

- [ ] Optimize routing for latency
- [ ] Implement adaptive model selection
- [ ] Cache frequently-used experts
- [ ] Benchmark on real ROM hacking tasks

---

## 7. Architecture Diagram

```
                           ┌─────────────────────────────────────────┐
                           │           Triforce MoE Router           │
                           │                                         │
                           │  ┌─────────────────────────────────┐   │
                           │  │    Intent Classifier (v2)       │   │
User Query ──────────────> │  │                                 │   │
                           │  │  • Keyword patterns             │   │
                           │  │  • Confidence scoring           │   │
                           │  │  • Multi-intent detection       │   │
                           │  └─────────────┬───────────────────┘   │
                           │                │                        │
                           │   ┌────────────┼────────────┐          │
                           │   │            │            │          │
                           │   ▼            ▼            ▼          │
                           │ ┌────┐      ┌────┐      ┌────┐        │
                           │ │Din │      │Nay-│      │Far-│        │
                           │ │7B  │      │ru  │      │ore │        │
                           │ │LoRA│      │7B  │      │7B  │        │
                           │ └────┘      │LoRA│      │LoRA│        │
                           │             └────┘      └────┘        │
                           │   ▼            ▼            ▼          │
                           │ Optim       Generate     Debug         │
                           └───┼────────────┼──────────┼───────────┘
                               │            │          │
                               │   ┌────────┴───────┐  │
                               │   │  Future Tier   │  │
                               │   ├────────────────┤  │
                               │   │ Veran   (14B)  │  │
                               │   │ Analysis       │  │
                               │   ├────────────────┤  │
                               │   │ Zelda   (14B)  │  │
                               │   │ Architecture   │  │
                               │   ├────────────────┤  │
                               │   │ Ganon   (7B)   │  │
                               │   │ Vulnerabilities│  │
                               │   └────────────────┘  │
                               │                       │
                               ▼                       ▼
                        ┌──────────────────────────────────┐
                        │         Response Synthesis       │
                        │                                  │
                        │  • Combine multi-expert results  │
                        │  • Validate assembly syntax      │
                        │  • Format with asar checks       │
                        └──────────────────────────────────┘
```

---

## 8. Open Questions

1. **Cross-expert queries:** How to handle "optimize this code, explain what you changed"?
   - Sequential routing (optimize -> analyze)?
   - Ensemble voting?
   - Meta-expert for coordination?

2. **Training data overlap:** Should Din and Nayru share common knowledge?
   - Risk of interference vs. benefit of coverage

3. **Dynamic model loading:** Can we predict next likely expert and pre-load?
   - Based on conversation context
   - User preference learning

4. **Evaluation metrics:** Beyond assembly correctness, how to measure quality?
   - Cycle counting automation
   - Size comparison benchmarks
   - User satisfaction tracking

---

## References

### MoE Architecture
- [Mixture of Experts Explained - HuggingFace](https://huggingface.co/blog/moe)
- [MoE Survey 2025 - arXiv](https://arxiv.org/html/2503.07137v1)
- [MoE Models Comparison 2025 - Friendli AI](https://friendli.ai/blog/moe-models-comparison)
- [ReMoE: Fully Differentiable MoE - ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/94dc604e115237a7f4a758b3146cd976-Paper-Conference.pdf)

### LLM Scaling & VRAM
- [VRAM Requirements Guide - LocalLLM](https://localllm.in/blog/ollama-vram-requirements-for-local-llms)
- [GPU Requirements for AI Models - BACloud](https://www.bacloud.com/en/blog/163/guide-to-gpu-requirements-for-running-ai-models.html)
- [Local LLM Hardware Guide 2025 - Introl](https://introl.com/blog/local-llm-hardware-pricing-guide-2025)

### LoRA & Fine-tuning
- [LoRA Original Paper - arXiv](https://arxiv.org/abs/2106.09685)
- [LoRA vs Full Fine-tuning - arXiv 2024](https://arxiv.org/abs/2410.21228)
- [LoRA Fine-tuning Guide - Databricks](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)
- [VRAM for Fine-tuning - Modal](https://modal.com/blog/how-much-vram-need-fine-tuning)

### Zelda Lore
- [Veran - Zelda Wiki](https://zelda.fandom.com/wiki/Veran)
