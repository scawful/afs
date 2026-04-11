import * as assert from "node:assert";
import * as path from "node:path";
import { describe, it, beforeEach } from "node:test";
import {
  __setActiveTextEditor,
  __setShowInformationMessage,
  __resetTestState,
  __setOpenTextDocument,
  __setShowErrorMessage,
  __setShowInputBox,
  __setShowQuickPick,
  __setShowTextDocument,
  __setShowWarningMessage,
  commands,
  workspace,
} from "vscode";
import { registerCommands } from "../../src/commands/index";
import { ContextService } from "../../src/services/contextService";
import { FileService } from "../../src/services/fileService";
import { IndexService } from "../../src/services/indexService";
import { MockTransport } from "./mockTransport";

describe("registerCommands", () => {
  beforeEach(() => {
    __resetTestState();
  });

  it("records a turn around index.query prompt submissions", async () => {
    const transport = new MockTransport();
    const workspaceRoot = "/tmp/afs-vscode-workspace";
    const selectedPath = path.join(workspaceRoot, ".context", "knowledge", "note.md");
    const shownDocs: unknown[] = [];

    transport.toolResponses["context.query"] = {
      entries: [
        {
          relative_path: "note.md",
          mount_type: "knowledge",
          size_bytes: 42,
          absolute_path: selectedPath,
        },
      ],
    };

    workspace.workspaceFolders = [{ name: "demo", uri: { fsPath: workspaceRoot } }];
    __setShowInputBox(async () => "sprite state");
    __setShowQuickPick(async (items) => (await Promise.resolve(items))[0]);
    __setOpenTextDocument(async (uri) => ({ uri }));
    __setShowTextDocument(async (doc) => {
      shownDocs.push(doc);
      return doc;
    });

    registerCommands(
      { subscriptions: [] } as never,
      {
        transport,
        contextService: new ContextService(transport),
        fileService: new FileService(transport),
        indexService: new IndexService(transport),
        treeProvider: { refresh() {} } as never,
        binaryInfo: { command: "afs", args: [], env: {} },
        logger: { appendLine() {}, dispose() {} } as never,
      },
    );

    await commands.executeCommand("afs.index.query");

    assert.deepStrictEqual(
      transport.turnEvents.map((event) => event.event),
      ["begin", "complete"],
    );
    assert.strictEqual(transport.turnEvents[0].prompt, "sprite state");
    assert.strictEqual(transport.turnEvents[0].summary, "Search AFS context index");
    assert.strictEqual(transport.turnEvents[1].summary, "Context query returned 1 result(s)");
    assert.strictEqual(shownDocs.length, 1);
  });

  it("records a failed turn when index.query errors", async () => {
    const transport = new MockTransport();
    const workspaceRoot = "/tmp/afs-vscode-workspace";
    const errorMessages: string[] = [];

    transport.toolErrors["context.query"] = new Error("query exploded");
    workspace.workspaceFolders = [{ name: "demo", uri: { fsPath: workspaceRoot } }];
    __setShowInputBox(async () => "broken query");
    __setShowErrorMessage(async (message) => {
      errorMessages.push(String(message));
      return undefined;
    });

    registerCommands(
      { subscriptions: [] } as never,
      {
        transport,
        contextService: new ContextService(transport),
        fileService: new FileService(transport),
        indexService: new IndexService(transport),
        treeProvider: { refresh() {} } as never,
        binaryInfo: { command: "afs", args: [], env: {} },
        logger: { appendLine() {}, dispose() {} } as never,
      },
    );

    await commands.executeCommand("afs.index.query");

    assert.deepStrictEqual(
      transport.turnEvents.map((event) => event.event),
      ["begin", "fail"],
    );
    assert.strictEqual(transport.turnEvents[0].prompt, "broken query");
    assert.strictEqual(transport.turnEvents[1].summary, "Context query failed for: broken query");
    assert.strictEqual(transport.turnEvents[1].error, "query exploded");
    assert.deepStrictEqual(errorMessages, ["Query failed: Error: query exploded"]);
  });

  it("prefers the active editor workspace for index.query in multi-root windows", async () => {
    const transport = new MockTransport();
    const workspaceA = "/tmp/afs-vscode-workspace-a";
    const workspaceB = "/tmp/afs-vscode-workspace-b";

    workspace.workspaceFolders = [
      { name: "alpha", uri: { fsPath: workspaceA } },
      { name: "beta", uri: { fsPath: workspaceB } },
    ];
    __setActiveTextEditor(path.join(workspaceB, "notes.md"));
    __setShowInputBox(async () => "sprite state");
    __setShowQuickPick(async (items) => (await Promise.resolve(items))[0]);

    transport.toolResponses["context.query"] = {
      entries: [],
    };

    registerCommands(
      { subscriptions: [] } as never,
      {
        transport,
        contextService: new ContextService(transport),
        fileService: new FileService(transport),
        indexService: new IndexService(transport),
        treeProvider: { refresh() {} } as never,
        binaryInfo: { command: "afs", args: [], env: {} },
        logger: { appendLine() {}, dispose() {} } as never,
      },
    );

    await commands.executeCommand("afs.index.query");

    const queryCall = transport.toolCalls.find((call) => call.name === "context.query");
    assert.ok(queryCall);
    assert.strictEqual(queryCall.args.context_path, path.join(workspaceB, ".context"));
  });

  it("uses selected mount-point data for context unmount", async () => {
    const transport = new MockTransport();
    const infoMessages: string[] = [];

    __setShowWarningMessage(async () => "Unmount");
    __setShowInformationMessage(async (message) => {
      infoMessages.push(String(message));
      return undefined;
    });
    transport.toolResponses["context.unmount"] = { removed: true };

    registerCommands(
      { subscriptions: [] } as never,
      {
        transport,
        contextService: new ContextService(transport),
        fileService: new FileService(transport),
        indexService: new IndexService(transport),
        treeProvider: { refresh() {} } as never,
        binaryInfo: { command: "afs", args: [], env: {} },
        logger: { appendLine() {}, dispose() {} } as never,
      },
    );

    await commands.executeCommand("afs.context.unmount", {
      contextPath: "/tmp/workspace/.context",
      mountType: "knowledge",
      alias: "notes",
    });

    const unmountCall = transport.toolCalls.find((call) => call.name === "context.unmount");
    assert.ok(unmountCall);
    assert.deepStrictEqual(unmountCall.args, {
      mount_type: "knowledge",
      alias: "notes",
      context_path: "/tmp/workspace/.context",
    });
    assert.deepStrictEqual(infoMessages, ["Unmounted notes from knowledge."]);
  });

  it("shows session hints in afs.mcp.status", async () => {
    const transport = new MockTransport();
    const infoMessages: string[] = [];
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

    __setShowInformationMessage(async (message) => {
      infoMessages.push(String(message));
      return undefined;
    });

    registerCommands(
      { subscriptions: [] } as never,
      {
        transport,
        contextService: new ContextService(transport),
        fileService: new FileService(transport),
        indexService: new IndexService(transport),
        treeProvider: { refresh() {} } as never,
        binaryInfo: { command: "afs", args: [], env: {} },
        logger: { appendLine() {}, dispose() {} } as never,
      },
    );

    await commands.executeCommand("afs.mcp.status");

    assert.strictEqual(infoMessages.length, 1);
    assert.match(infoMessages[0], /Connected: true/);
    assert.match(infoMessages[0], /Session workspace: \/tmp\/workspace/);
    assert.match(infoMessages[0], /Query hint: afs query <text> --path \/tmp\/workspace/);
    assert.match(infoMessages[0], /Canonical query hint: afs context query <text> --path \/tmp\/workspace/);
    assert.match(infoMessages[0], /Index hint: afs index rebuild --path \/tmp\/workspace/);
    assert.match(infoMessages[0], /Note: Indexed retrieval may be stale\./);
  });
});
