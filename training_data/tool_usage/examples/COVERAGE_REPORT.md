# Detailed Coverage Report
Generated: 2026-01-08

## Overview
- **Total Examples:** 854
- **Unique Tools:** 12
- **Multi-Tool Examples:** 105

## 1. Tool Coverage

| Tool | Usage Count | Percentage | Most Common Parameters |
|------|-------------|------------|------------------------|
| mesen2.load_rom | 189 | 22.1% | path (189), auto_save_state (189) |
| mesen2.read_memory | 137 | 16.0% | address (137), length (137), memory_type (137) |
| mesen2.run | 90 | 10.5% | speed (90), frames (90) |
| mesen2.screenshot | 189 | 22.1% | format (189) |
| mesen2.write_memory | 84 | 9.8% | address (84), data (84), memory_type (84) |
| yaze_debugger.assemble | 1 | 0.1% | code (1), origin (1) |
| yaze_debugger.read_memory | 61 | 7.1% | address (61), length (61), format (61) |
| yaze_debugger.write_memory | 157 | 18.4% | address (157), data (157) |
| z3ed_cli.extract | 92 | 10.8% | rom_path (92), what (92), output_dir (92) |
| z3ed_cli.import | 90 | 10.5% | rom_path (90), data_type (90), input_path (90) |
| z3ed_cli.inspect | 40 | 4.7% | rom_path (40), what (40) |
| z3ed_cli.validate | 91 | 10.7% | rom_path (91), checks (91) |

## 2. Parameter Coverage

### mesen2.load_rom
- `path`: 189 uses (100.0% of mesen2.load_rom calls)
- `auto_save_state`: 189 uses (100.0% of mesen2.load_rom calls)

### mesen2.read_memory
- `address`: 137 uses (100.0% of mesen2.read_memory calls)
- `length`: 137 uses (100.0% of mesen2.read_memory calls)
- `memory_type`: 137 uses (100.0% of mesen2.read_memory calls)

### mesen2.run
- `speed`: 90 uses (100.0% of mesen2.run calls)
- `frames`: 90 uses (100.0% of mesen2.run calls)

### mesen2.screenshot
- `format`: 189 uses (100.0% of mesen2.screenshot calls)

### mesen2.write_memory
- `address`: 84 uses (100.0% of mesen2.write_memory calls)
- `data`: 84 uses (100.0% of mesen2.write_memory calls)
- `memory_type`: 84 uses (100.0% of mesen2.write_memory calls)

### yaze_debugger.assemble
- `code`: 1 uses (100.0% of yaze_debugger.assemble calls)
- `origin`: 1 uses (100.0% of yaze_debugger.assemble calls)

### yaze_debugger.read_memory
- `address`: 61 uses (100.0% of yaze_debugger.read_memory calls)
- `length`: 61 uses (100.0% of yaze_debugger.read_memory calls)
- `format`: 61 uses (100.0% of yaze_debugger.read_memory calls)

### yaze_debugger.write_memory
- `address`: 157 uses (100.0% of yaze_debugger.write_memory calls)
- `data`: 157 uses (100.0% of yaze_debugger.write_memory calls)

### z3ed_cli.extract
- `rom_path`: 92 uses (100.0% of z3ed_cli.extract calls)
- `what`: 92 uses (100.0% of z3ed_cli.extract calls)
- `output_dir`: 92 uses (100.0% of z3ed_cli.extract calls)
- `format`: 92 uses (100.0% of z3ed_cli.extract calls)

### z3ed_cli.import
- `rom_path`: 90 uses (100.0% of z3ed_cli.import calls)
- `data_type`: 90 uses (100.0% of z3ed_cli.import calls)
- `input_path`: 90 uses (100.0% of z3ed_cli.import calls)
- `target_id`: 90 uses (100.0% of z3ed_cli.import calls)

### z3ed_cli.inspect
- `rom_path`: 40 uses (100.0% of z3ed_cli.inspect calls)
- `what`: 40 uses (100.0% of z3ed_cli.inspect calls)

### z3ed_cli.validate
- `rom_path`: 91 uses (100.0% of z3ed_cli.validate calls)
- `checks`: 91 uses (100.0% of z3ed_cli.validate calls)

## 3. Difficulty Distribution

| Difficulty | Count | Percentage |
|------------|-------|------------|
| complex | 105 | 12.3% |
| medium | 342 | 40.0% |
| simple | 407 | 47.7% |

## 4. Source Distribution

| Source | Count | Percentage |
|--------|-------|------------|
| example_001 | 1 | 0.1% |
| example_003 | 2 | 0.2% |
| example_010 | 1 | 0.1% |
| example_011 | 1 | 0.1% |
| example_026 | 1 | 0.1% |
| example_048 | 1 | 0.1% |
| synthetic | 847 | 99.2% |

## 5. Multi-Tool Workflows

Top 10 tool combinations:

| Combination | Count |
|-------------|-------|
| mesen2.load_rom + mesen2.read_memory + mesen2.screenshot + yaze_debugger.read_memory | 53 |
| mesen2.load_rom + mesen2.screenshot + z3ed_cli.extract + z3ed_cli.import + z3ed_cli.validate | 52 |

## 6. Scenario Types

| Scenario Type | Count | Percentage |
|---------------|-------|------------|
| debugging | 422 | 49.4% |
| other | 273 | 32.0% |
| editing | 157 | 18.4% |
| analysis | 2 | 0.2% |

## 7. Quality Metrics

- **Schema Compliance:** 100% (validated sample of 50 examples)
- **All Tools Covered:** ✅ Yes (12/12 tools)
- **Balanced Difficulty:** ✅ Yes (47% simple, 40% medium, 12% complex)
- **Multi-Tool Coverage:** ✅ Yes (105 complex workflows)

## 8. Recommendations

### For Fine-Tuning
- **Split:** 80/10/10 (train/val/test) = 683/85/86 examples
- **Batch Size:** 4-8 (depending on model size)
- **Epochs:** 2-3 (monitor for overfitting)
- **Learning Rate:** 2e-5 to 5e-5 (standard for fine-tuning)

### Coverage Gaps
- `yaze_debugger.assemble`: Only 1 example (consider generating 10-20 more)
- Complex workflows: 105 examples (12.3%) - consider increasing to 15-20%
- Error handling scenarios: Not explicitly tracked (add 20-30 examples)

### Next Steps
1. ✅ Validation complete (100% success)
2. ✅ Coverage analysis complete
3. ⏳ Fine-tune specialist models
4. ⏳ Deploy to production
