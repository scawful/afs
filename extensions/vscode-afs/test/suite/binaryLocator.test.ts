import * as assert from "node:assert";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, it } from "node:test";
import { __resetTestState, __setConfiguration, workspace } from "vscode";
import {
  locateAfsBinary,
  resolveAfsBinary,
} from "../../src/utils/binaryLocator";

const tempRoots: string[] = [];

describe("binaryLocator", () => {
  beforeEach(() => {
    __resetTestState();
  });

  afterEach(() => {
    for (const tempRoot of tempRoots.splice(0)) {
      fs.rmSync(tempRoot, { recursive: true, force: true });
    }
  });

  it("uses an explicit configured command", () => {
    __setConfiguration("afs.server.command", "custom-afs");
    const info = locateAfsBinary(makeLogger());
    assert.deepStrictEqual(info, { command: "custom-afs", args: [], env: {} });
  });

  it("prefers workspace venv python when present", () => {
    const root = makeWorkspace();
    fs.mkdirSync(path.join(root, ".venv", "bin"), { recursive: true });
    fs.writeFileSync(path.join(root, ".venv", "bin", "python"), "", "utf-8");
    workspace.workspaceFolders = [{ name: "demo", uri: { fsPath: root } }];

    const info = locateAfsBinary(makeLogger());
    assert.deepStrictEqual(info, {
      command: path.join(root, ".venv", "bin", "python"),
      args: ["-m", "afs"],
      env: {},
    });
  });

  it("skips deferred PATH probing for explicit or workspace-resolved binaries", async () => {
    const explicit = { command: "/tmp/afs", args: [], env: {} };
    const resolved = await resolveAfsBinary(explicit, makeLogger());
    assert.strictEqual(resolved, explicit);
  });
});

function makeWorkspace(): string {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "afs-vscode-binary-"));
  tempRoots.push(tempRoot);
  return tempRoot;
}

function makeLogger() {
  return {
    appendLine() {},
    dispose() {},
  };
}
