import * as assert from "node:assert";
import { describe, it } from "node:test";
import {
  contextPathFromFilePath,
  shouldIgnoreAutoRebuildPath,
} from "../../src/utils/contextPaths";

describe("contextPaths", () => {
  it("extracts the containing .context path from nested files", () => {
    assert.strictEqual(
      contextPathFromFilePath("/tmp/workspace/.context/scratchpad/state.md"),
      "/tmp/workspace/.context",
    );
  });

  it("ignores index artifact paths for auto-rebuild", () => {
    assert.strictEqual(
      shouldIgnoreAutoRebuildPath("/tmp/workspace/.context/global/context_index.sqlite3-wal"),
      true,
    );
    assert.strictEqual(
      shouldIgnoreAutoRebuildPath("/tmp/workspace/.context/scratchpad/state.md"),
      false,
    );
  });
});
