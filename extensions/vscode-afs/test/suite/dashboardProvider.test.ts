import * as assert from "node:assert";
import { afterEach, beforeEach, describe, it } from "node:test";
import { __resetTestState, __setActiveTextEditor, workspace } from "vscode";
import { AfsDashboardProvider } from "../../src/views/dashboardProvider";
import { MockTransport } from "./mockTransport";

describe("AfsDashboardProvider", () => {
  beforeEach(() => {
    __resetTestState();
  });

  afterEach(() => {
    __resetTestState();
  });

  it("renders session hint commands and actions when transport session info is available", () => {
    const transport = new MockTransport();
    transport.sessionInfo = {
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
    };

    const provider = new AfsDashboardProvider(
      transport,
      {
        appendLine() {},
        dispose() {},
      } as never,
      { command: "/usr/bin/python3", args: ["-m", "afs"], env: { AFS_ROOT: "/tmp/workspace" } },
    );

    const html = (provider as any).buildHtml(
      true,
      transport.capabilities(),
      null,
      null,
      null,
      transport.getSessionInfo(),
      { registered: true, configPath: "/tmp/mcp.json" },
      {
        command: "/usr/bin/python3",
        args: ["-m", "afs", "mcp", "serve"],
        env: { AFS_ROOT: "/tmp/workspace" },
      },
    ) as string;

    assert.match(html, /Session Hints/);
    assert.match(html, /MCP/);
    assert.match(html, /Update MCP Registration/);
    assert.match(html, /Copy Config Path/);
    assert.match(html, /\/usr\/bin\/python3 -m afs mcp serve/);
    assert.match(html, /afs query &lt;text&gt; --path \/tmp\/workspace/);
    assert.match(html, /afs context query &lt;text&gt; --path \/tmp\/workspace/);
    assert.match(html, /afs index rebuild --path \/tmp\/workspace/);
    assert.match(html, /Query More/);
    assert.match(html, /Copy Query Command/);
    assert.match(html, /Copy Rebuild Command/);
    assert.match(html, /Indexed retrieval may be stale\./);
  });

  it("resolves dashboard tool args from the active session context path", () => {
    const transport = new MockTransport();
    transport.sessionInfo = {
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
        notes: [],
      },
    };

    const provider = new AfsDashboardProvider(
      transport,
      {
        appendLine() {},
        dispose() {},
      } as never,
      { command: "afs", args: [], env: {} },
    );

    const args = (provider as any).resolveContextArgs(transport.getSessionInfo()) as Record<string, string>;
    assert.deepStrictEqual(args, { context_path: "/tmp/workspace/.context" });
  });

  it("prefers the active editor workspace context in multi-root windows", () => {
    workspace.workspaceFolders = [
      { name: "one", uri: { fsPath: "/tmp/workspace-one" } },
      { name: "two", uri: { fsPath: "/tmp/workspace-two" } },
    ];
    __setActiveTextEditor("/tmp/workspace-two/src/file.ts");

    const transport = new MockTransport();
    transport.sessionInfo = {
      sessionId: "sess-vscode",
      payloadFile: "/tmp/session_client_vscode.json",
      contextPath: "/tmp/workspace-one/.context",
      promptJson: "/tmp/session_system_prompt_vscode.json",
      promptText: "/tmp/session_system_prompt_vscode.txt",
      workspace: "/tmp/workspace-one",
      cliHints: {
        workspacePath: "/tmp/workspace-one",
        queryShortcut: "afs query <text> --path /tmp/workspace-one",
        queryCanonical: "afs context query <text> --path /tmp/workspace-one",
        indexRebuild: "afs index rebuild --path /tmp/workspace-one",
        notes: [],
      },
    };

    const provider = new AfsDashboardProvider(
      transport,
      {
        appendLine() {},
        dispose() {},
      } as never,
      { command: "afs", args: [], env: {} },
    );

    const args = (provider as any).resolveContextArgs(transport.getSessionInfo()) as Record<string, string>;
    assert.deepStrictEqual(args, { context_path: "/tmp/workspace-two/.context" });
  });
});
