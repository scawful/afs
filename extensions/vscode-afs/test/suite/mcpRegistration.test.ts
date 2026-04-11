import * as assert from "node:assert";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, it } from "node:test";
import { __resetTestState, __setConfiguration } from "vscode";
import {
  buildServerEntry,
  checkRegistration,
  mergeMcpServerConfig,
} from "../../src/mcp/registration";

const tempDirs: string[] = [];

describe("MCP registration helpers", () => {
  beforeEach(() => {
    __resetTestState();
  });

  afterEach(() => {
    for (const tempDir of tempDirs.splice(0)) {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("builds an MCP server entry using configured args and env", () => {
    __setConfiguration("afs.server.args", ["--verbose"]);
    __setConfiguration("afs.server.env", { AFS_LOG_LEVEL: "debug" });

    const entry = buildServerEntry({
      command: "python3",
      args: ["-m", "afs"],
      env: { AFS_ROOT: "/tmp/workspace" },
    });

    assert.deepStrictEqual(entry, {
      command: "python3",
      args: ["-m", "afs", "mcp", "serve", "--verbose"],
      env: {
        AFS_ROOT: "/tmp/workspace",
        AFS_LOG_LEVEL: "debug",
      },
    });
  });

  it("merges the AFS entry while preserving other MCP servers", () => {
    const merged = mergeMcpServerConfig(
      {
        version: "1.0",
        mcpServers: {
          other: { command: "other-server", args: [] },
          afs: { command: "old-afs", args: ["old"] },
        },
      },
      { command: "new-afs", args: ["mcp", "serve"], env: { AFS_ROOT: "/tmp/workspace" } },
    );

    assert.strictEqual(merged.version, "1.0");
    assert.deepStrictEqual(merged.mcpServers?.other, {
      command: "other-server",
      args: [],
    });
    assert.deepStrictEqual(merged.mcpServers?.afs, {
      command: "new-afs",
      args: ["mcp", "serve"],
      env: { AFS_ROOT: "/tmp/workspace" },
    });
  });

  it("reports the registered AFS entry for an override config path", () => {
    const configPath = writeTempConfig({
      mcpServers: {
        afs: {
          command: "python3",
          args: ["-m", "afs", "mcp", "serve"],
          env: { AFS_ROOT: "/tmp/workspace" },
        },
      },
    });
    __setConfiguration("afs.mcp.configPath", configPath);

    const status = checkRegistration();

    assert.strictEqual(status.registered, true);
    assert.strictEqual(status.configPath, configPath);
    assert.deepStrictEqual(status.entry, {
      command: "python3",
      args: ["-m", "afs", "mcp", "serve"],
      env: { AFS_ROOT: "/tmp/workspace" },
    });
  });

  it("surfaces parse errors for invalid MCP config files", () => {
    const tempDir = mkdtempSync(path.join(os.tmpdir(), "afs-mcp-invalid-"));
    tempDirs.push(tempDir);
    const configPath = path.join(tempDir, "mcp.json");
    writeFileSync(configPath, "{not-json", "utf-8");
    __setConfiguration("afs.mcp.configPath", configPath);

    const status = checkRegistration();

    assert.strictEqual(status.registered, false);
    assert.strictEqual(status.configPath, configPath);
    assert.match(status.parseError ?? "", /Could not parse MCP config/);
  });
});

function writeTempConfig(contents: Record<string, unknown>): string {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), "afs-mcp-config-"));
  tempDirs.push(tempDir);
  const configPath = path.join(tempDir, "mcp.json");
  writeFileSync(configPath, JSON.stringify(contents, null, 2), "utf-8");
  return configPath;
}
