import * as assert from "node:assert";
import { describe, it } from "node:test";
import {
  buildMcpConfigCandidates,
  defaultWorkspaceMcpConfigPath,
  resolveExistingMcpConfigPath,
} from "../../src/mcp/configCandidates";

describe("MCP config candidate discovery", () => {
  it("includes workspace, host, and user-level candidates", () => {
    const candidates = buildMcpConfigCandidates({
      home: "/Users/test",
      workspaceFolders: ["/Users/test/workspace"],
      antigravityContextRoot: "/Users/test/.gemini/antigravity",
    });

    assert.ok(candidates.includes("/Users/test/workspace/.cursor/mcp.json"));
    assert.ok(candidates.includes("/Users/test/workspace/.vscode/mcp.json"));
    assert.ok(candidates.includes("/Users/test/.gemini/antigravity/mcp.json"));
    assert.ok(candidates.includes("/Users/test/.gemini/antigravity/mcp_config.json"));
    assert.ok(candidates.includes("/Users/test/.cursor/mcp.json"));
    assert.ok(candidates.includes("/Users/test/.vscode/mcp.json"));
    assert.ok(candidates.includes("/Users/test/Library/Application Support/Cursor/User/mcp.json"));
    assert.ok(candidates.includes("/Users/test/Library/Application Support/Antigravity/User/mcp.json"));
  });

  it("picks the first existing candidate", () => {
    const path = resolveExistingMcpConfigPath(
      ["/missing.json", "/exists.json", "/other.json"],
      (candidate) => candidate === "/exists.json",
    );

    assert.strictEqual(path, "/exists.json");
  });

  it("defaults to workspace cursor config when no existing file is found", () => {
    assert.strictEqual(
      defaultWorkspaceMcpConfigPath(["/Users/test/workspace"]),
      "/Users/test/workspace/.cursor/mcp.json",
    );
  });
});
