import * as assert from "node:assert";
import { describe, it } from "node:test";
import { buildAfsContextMessage } from "../../src/chat/contextFormatter";

describe("buildAfsContextMessage", () => {
  it("renders session, pack, query, and scratchpad context", () => {
    const text = buildAfsContextMessage({
      sessionPrompt: "Follow AFS grounding guidance.",
      status: {
        context_path: "/tmp/workspace/.context",
        profile: "dev",
        mount_counts: { scratchpad: 3, knowledge: 10 },
        total_files: 13,
        index: { enabled: true, has_entries: true, stale: false, total_entries: 24 },
        actions: ["Review context.diff before editing."],
      },
      freshness: {
        mount_scores: { scratchpad: 0.91, knowledge: 0.42 },
      },
      pack: {
        project: "afs",
        profile: "dev",
        model: "generic",
        pack_mode: "focused",
        estimated_tokens: 400,
        sections: [
          { title: "Scratchpad", body: "State goes here." },
          { title: "Index Hits", body: "Hit summary." },
        ],
      },
      queryEntries: [
        {
          mount_type: "scratchpad",
          relative_path: "notes.md",
          absolute_path: "/tmp/workspace/.context/scratchpad/notes.md",
          is_dir: false,
          size_bytes: 42,
          modified_at: null,
          content_excerpt: "Most relevant note.",
        },
      ],
      scratchpadState: "# State\nCurrent work",
      scratchpadDeferred: "# Deferred\nFollow up later",
    });

    assert.match(text, /Prepared AFS Session Prompt/);
    assert.match(text, /AFS Context Status/);
    assert.match(text, /Index: enabled=yes, has_entries=yes, stale=no, entries=24/);
    assert.match(text, /Freshness: scratchpad=91%, knowledge=42%/);
    assert.match(text, /Recommended actions:/);
    assert.match(text, /Review context\.diff before editing\./);
    assert.match(text, /AFS Session Pack/);
    assert.match(text, /Indexed Context Hits/);
    assert.match(text, /scratchpad\/notes\.md/);
    assert.match(text, /Most relevant note\./);
    assert.match(text, /Scratchpad Notes/);
  });
});
