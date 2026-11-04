# Record engineering guidance provenance

Reviewed pinned primary sources on 2026-10-30. Local guidance is an original, selective adaptation; no upstream skill pack was installed or executed.

- Reviewed [Matt Pocock’s domain modeling](https://github.com/mattpocock/skills/blob/a496f5513d4a4cf889bc48e189e579f4b0bdf6aa/skills/engineering/domain-modeling/SKILL.md) at `a496f5513d4a4cf889bc48e189e579f4b0bdf6aa` under [MIT](https://github.com/mattpocock/skills/blob/a496f5513d4a4cf889bc48e189e579f4b0bdf6aa/LICENSE).
- Reviewed [Jesse Vincent’s Superpowers systematic debugging](https://github.com/obra/superpowers/blob/2ce8aa15d05bd57213e6f746f953aafb7ded7a8b/skills/systematic-debugging/SKILL.md) at `2ce8aa15d05bd57213e6f746f953aafb7ded7a8b` under [MIT](https://github.com/obra/superpowers/blob/2ce8aa15d05bd57213e6f746f953aafb7ded7a8b/LICENSE).
- Reviewed [DietrichGebert’s Ponytail](https://github.com/DietrichGebert/ponytail/blob/96608e078b2d5ea607c5c3b8b5a54d1c1fba6846/skills/ponytail/SKILL.md) at `96608e078b2d5ea607c5c3b8b5a54d1c1fba6846` under [MIT](https://github.com/DietrichGebert/ponytail/blob/96608e078b2d5ea607c5c3b8b5a54d1c1fba6846/LICENSE).

## Keep the useful decisions

Use code-backed language, relevant boundary evidence, and complete fixes with proportionate checks. Domain modeling contributes lazy terminology/decision capture; debugging contributes reproduction, bad-state tracing, and a useful hypothesis; Ponytail contributes reuse and complete scope.

Keep existing domain sources. Oracle already has `~/src/hobby/oracle-of-secrets/Docs/Technical/terminology_glossary.md` and `Docs/Planning/Status/decisions.org`; yaze has `~/src/hobby/yaze/docs/internal/architecture/` and `docs/public/developer/architecture.md`. Read only task-relevant entries and confirm them against code. Preserve canon approval and each repo's current checkout/testing rules.

## Leave out the framework

Omit mandatory interviews, automatic glossary writes after each resolved term, rigid phases, required test-first ceremony for every edit, fixed retry counts, cross-skill dependencies, persistent persona modes, and blanket caller inventories. Don't turn internal judgment into checklists, scorecards, compulsory documentation, automatic PRs, or repeated full-suite runs. A small fix gets the check that proves it; wider contracts justify wider checks. Existing mandatory repo checks still apply.

## Preserve upstream notices

The sources above use the MIT license. Retain their notices here for attribution; no ownership claim is made over upstream material.

Copyright (c) 2026 Matt Pocock

Copyright (c) 2025 Jesse Vincent

Copyright (c) 2026 DietrichGebert

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
