import * as assert from "node:assert";
import { describe, it, beforeEach } from "node:test";
import { __getLastStatusBarItem, __resetTestState } from "vscode";
import { AfsStatusBar } from "../../src/views/statusBar";

describe("AfsStatusBar", () => {
  beforeEach(() => {
    __resetTestState();
  });

  it("includes session hints in the connected tooltip", () => {
    const statusBar = new AfsStatusBar();
    statusBar.update(
      "connected",
      undefined,
      {
        sessionId: "sess-vscode",
        payloadFile: "/tmp/session_client_vscode.json",
        contextPath: "/tmp/workspace/.context",
        promptJson: "/tmp/session_system_prompt_vscode.json",
        promptText: "/tmp/session_system_prompt_vscode.txt",
        workspace: "/tmp/workspace",
        cliHints: {
          workspacePath: "/tmp/workspace",
          queryShortcut: "afs query <text> --path /tmp/workspace",
          queryCanonical: "afs context query <text> --path /tmp/workspace",
          indexRebuild: "afs index rebuild --path /tmp/workspace",
          notes: ["Indexed retrieval may be stale."],
        },
      },
    );

    const item = __getLastStatusBarItem();
    assert.ok(item);
    assert.strictEqual(item.text, "$(check) AFS: workspace");
    assert.match(item.tooltip, /AFS connected/);
    assert.match(item.tooltip, /Workspace: workspace/);
    assert.match(item.tooltip, /Query: afs query <text> --path \/tmp\/workspace/);
    assert.match(item.tooltip, /Rebuild: afs index rebuild --path \/tmp\/workspace/);
    assert.match(item.tooltip, /Note: Indexed retrieval may be stale\./);
  });

  it("stays hidden when disabled", () => {
    const statusBar = new AfsStatusBar(false);
    statusBar.update("connected");

    const item = __getLastStatusBarItem();
    assert.ok(item);
    assert.strictEqual(item.visible, false);

    statusBar.setEnabled(true);
    assert.strictEqual(item.visible, true);
  });
});
