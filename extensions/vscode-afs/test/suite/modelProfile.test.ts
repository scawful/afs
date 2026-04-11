import * as assert from "node:assert";
import { describe, it } from "node:test";
import { resolveAfsModelProfile } from "../../src/chat/modelProfile";

describe("resolveAfsModelProfile", () => {
  it("respects explicit configuration", () => {
    assert.strictEqual(
      resolveAfsModelProfile("claude", { id: "gemini-2.5-pro" }),
      "claude",
    );
  });

  it("infers gemini models automatically", () => {
    assert.strictEqual(
      resolveAfsModelProfile("auto", { id: "gemini-2.5-pro", vendor: "google" }),
      "gemini",
    );
  });

  it("falls back to generic for unknown models", () => {
    assert.strictEqual(
      resolveAfsModelProfile("auto", { id: "gpt-5-mini", vendor: "copilot" }),
      "generic",
    );
  });
});
