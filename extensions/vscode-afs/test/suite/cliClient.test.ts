import * as assert from "node:assert";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, it } from "node:test";
import { workspace } from "vscode";
import { CliClient } from "../../src/transport/cliClient";

function writeFakeAfsBinary(tmpDir: string, logPath: string): string {
  const scriptPath = path.join(tmpDir, "fake-afs.js");
  fs.writeFileSync(
    scriptPath,
    `#!/usr/bin/env node
const fs = require("node:fs");
const path = require("node:path");
const args = process.argv.slice(2);
const logPath = process.env.CLI_TEST_LOG;
const root = process.env.CLI_TEST_ROOT;
const promptJson = path.join(root, "session_system_prompt_vscode.json");
const promptText = path.join(root, "session_system_prompt_vscode.txt");
const payloadJson = path.join(root, "session_client_vscode.json");

fs.appendFileSync(
  logPath,
  JSON.stringify({
    args,
    env: {
      AFS_SESSION_ID: process.env.AFS_SESSION_ID || "",
      AFS_SESSION_CLIENT_PAYLOAD_JSON: process.env.AFS_SESSION_CLIENT_PAYLOAD_JSON || "",
      AFS_SESSION_SYSTEM_PROMPT_JSON: process.env.AFS_SESSION_SYSTEM_PROMPT_JSON || "",
      AFS_SESSION_SYSTEM_PROMPT_TEXT: process.env.AFS_SESSION_SYSTEM_PROMPT_TEXT || "",
      AFS_SESSION_QUERY_HINT: process.env.AFS_SESSION_QUERY_HINT || "",
      AFS_SESSION_CONTEXT_QUERY_HINT: process.env.AFS_SESSION_CONTEXT_QUERY_HINT || "",
      AFS_SESSION_INDEX_REBUILD_HINT: process.env.AFS_SESSION_INDEX_REBUILD_HINT || "",
      AFS_SESSION_DEFAULT_TURN_ID: process.env.AFS_SESSION_DEFAULT_TURN_ID || "",
      AFS_ACTIVE_CONTEXT_ROOT: process.env.AFS_ACTIVE_CONTEXT_ROOT || "",
    },
  }) + "\\n",
);

if (args.length === 1 && args[0] === "--help") {
  process.stdout.write("help");
  process.exit(0);
}

if (args[0] === "session" && args[1] === "prepare-client") {
  fs.writeFileSync(promptText, "You are the VS Code AFS harness.\\n", "utf-8");
  fs.writeFileSync(
    promptJson,
    JSON.stringify({ text: "You are the VS Code AFS harness." }, null, 2),
    "utf-8",
  );
  fs.writeFileSync(payloadJson, "{}", "utf-8");
  process.stdout.write(
    JSON.stringify({
      client: "vscode",
      session_id: "sess-test",
      context_path: path.join(root, "workspace", ".context"),
      prompt: {
        artifact_paths: {
          json: promptJson,
          text: promptText,
        },
      },
      cli_hints: {
        workspace_path: path.join(root, "workspace"),
        query_shortcut: "afs query <text> --path " + path.join(root, "workspace"),
        query_canonical: "afs context query <text> --path " + path.join(root, "workspace"),
        index_rebuild: "afs index rebuild --path " + path.join(root, "workspace"),
        notes: ["Index may be stale"],
      },
      artifact_paths: {
        json: payloadJson,
      },
    }),
  );
  process.exit(0);
}

if (args[0] === "session" && (args[1] === "hook" || args[1] === "event")) {
  process.stdout.write("{}");
  process.exit(0);
}

if (args[0] === "fs" && args[1] === "read") {
  if (args[3] === "missing.txt") {
    process.stderr.write("missing file");
    process.exit(1);
  }
  process.stdout.write("from-context");
  process.exit(0);
}

if (args[0] === "context" && args[1] === "query") {
  process.stdout.write(
    JSON.stringify({
      count: 1,
      entries: [
        {
          mount_type: "scratchpad",
          relative_path: "note.md",
          absolute_path: path.join(root, "workspace", ".context", "scratchpad", "note.md"),
          is_dir: false,
          size_bytes: 12,
          modified_at: "2026-04-04T00:00:00+00:00",
          indexed_at: "2026-04-04T00:00:00+00:00",
          content_excerpt: "from query",
        },
      ],
    }),
  );
  process.exit(0);
}

if (args[0] === "index" && args[1] === "rebuild") {
  process.stdout.write(
    JSON.stringify({
      context_path: path.join(root, "workspace", ".context"),
      db_path: path.join(root, "workspace", ".context", "global", "context_index.sqlite3"),
      indexed_at: "2026-04-04T00:00:00+00:00",
      rows_written: 1,
      rows_deleted: 0,
      by_mount_type: { scratchpad: 1 },
      skipped_large_files: 0,
      skipped_binary_files: 0,
      errors: [],
    }),
  );
  process.exit(0);
}

process.stderr.write("unsupported command: " + args.join(" "));
process.exit(1);
`,
    "utf-8",
  );
  fs.chmodSync(scriptPath, 0o755);
  return scriptPath;
}

function readLogEntries(logPath: string): Array<{ args: string[]; env: Record<string, string> }> {
  const raw = fs.readFileSync(logPath, "utf-8").trim();
  if (!raw) {
    return [];
  }
  return raw.split("\n").map((line) => JSON.parse(line));
}

describe("CliClient", () => {
  it("prepares a VS Code session harness and exports prompt artifacts to tool calls", async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "afs-cli-client-"));
    const logPath = path.join(tmpDir, "cli.log");
    const workspaceRoot = path.join(tmpDir, "workspace");
    const notePath = path.join(workspaceRoot, ".context", "scratchpad", "note.md");
    fs.mkdirSync(path.dirname(notePath), { recursive: true });
    fs.writeFileSync(notePath, "", "utf-8");
    workspace.workspaceFolders = [{ uri: { fsPath: workspaceRoot } }];

    const client = new CliClient(
      writeFakeAfsBinary(tmpDir, logPath),
      [],
      {
        CLI_TEST_LOG: logPath,
        CLI_TEST_ROOT: tmpDir,
      },
      {
        appendLine() {},
        dispose() {},
      } as never,
      5_000,
    );

    await client.initialize();
    const result = await client.callTool("context.read", { path: notePath });
    assert.deepStrictEqual(result, { path: notePath, content: "from-context" });
    client.dispose();
    await new Promise((resolve) => setTimeout(resolve, 50));

    const calls = readLogEntries(logPath);
    assert.ok(calls.some((entry) => entry.args[0] === "session" && entry.args[1] === "prepare-client"));
    assert.ok(calls.some((entry) => entry.args[0] === "session" && entry.args[1] === "hook" && entry.args[2] === "session_start"));
    assert.ok(calls.some((entry) => entry.args[0] === "session" && entry.args[1] === "event" && entry.args[2] === "task_created"));
    assert.ok(calls.some((entry) => entry.args[0] === "session" && entry.args[1] === "event" && entry.args[2] === "task_completed"));
    assert.ok(calls.some((entry) => entry.args[0] === "session" && entry.args[1] === "hook" && entry.args[2] === "session_end"));

    const fsReadCall = calls.find((entry) => entry.args[0] === "fs" && entry.args[1] === "read");
    assert.ok(fsReadCall);
    assert.strictEqual(fsReadCall.env.AFS_SESSION_ID.length, 12);
    assert.strictEqual(fsReadCall.env.AFS_SESSION_CLIENT_PAYLOAD_JSON, path.join(tmpDir, "session_client_vscode.json"));
    assert.strictEqual(fsReadCall.env.AFS_SESSION_SYSTEM_PROMPT_JSON, path.join(tmpDir, "session_system_prompt_vscode.json"));
    assert.strictEqual(fsReadCall.env.AFS_SESSION_SYSTEM_PROMPT_TEXT, path.join(tmpDir, "session_system_prompt_vscode.txt"));
    assert.strictEqual(fsReadCall.env.AFS_SESSION_QUERY_HINT, `afs query <text> --path ${workspaceRoot}`);
    assert.strictEqual(
      fsReadCall.env.AFS_SESSION_CONTEXT_QUERY_HINT,
      `afs context query <text> --path ${workspaceRoot}`,
    );
    assert.strictEqual(
      fsReadCall.env.AFS_SESSION_INDEX_REBUILD_HINT,
      `afs index rebuild --path ${workspaceRoot}`,
    );
    assert.strictEqual(fsReadCall.env.AFS_ACTIVE_CONTEXT_ROOT, path.join(tmpDir, "workspace", ".context"));
    assert.deepStrictEqual(client.getSessionInfo()?.cliHints, {
      workspacePath: workspaceRoot,
      queryShortcut: `afs query <text> --path ${workspaceRoot}`,
      queryCanonical: `afs context query <text> --path ${workspaceRoot}`,
      indexRebuild: `afs index rebuild --path ${workspaceRoot}`,
      notes: ["Index may be stale"],
    });
  });

  it("records turn events and carries the active turn into nested tool calls", async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "afs-cli-client-turn-"));
    const logPath = path.join(tmpDir, "cli.log");
    const workspaceRoot = path.join(tmpDir, "workspace");
    const notePath = path.join(workspaceRoot, ".context", "scratchpad", "note.md");
    fs.mkdirSync(path.dirname(notePath), { recursive: true });
    fs.writeFileSync(notePath, "", "utf-8");
    workspace.workspaceFolders = [{ uri: { fsPath: workspaceRoot } }];

    const client = new CliClient(
      writeFakeAfsBinary(tmpDir, logPath),
      [],
      {
        CLI_TEST_LOG: logPath,
        CLI_TEST_ROOT: tmpDir,
      },
      {
        appendLine() {},
        dispose() {},
      } as never,
      5_000,
    );

    await client.initialize();
    const turnId = await client.beginTurn("Inspect scratchpad note", "Search AFS context index");
    const result = await client.callTool("context.read", { path: notePath });
    assert.deepStrictEqual(result, { path: notePath, content: "from-context" });
    await client.completeTurn(turnId, "Context query returned 1 result(s)");
    client.dispose();
    await new Promise((resolve) => setTimeout(resolve, 50));

    const calls = readLogEntries(logPath);
    const fsReadCall = calls.find((entry) => entry.args[0] === "fs" && entry.args[1] === "read");
    assert.ok(fsReadCall);
    assert.strictEqual(fsReadCall.env.AFS_SESSION_DEFAULT_TURN_ID, turnId);
    assert.ok(
      calls.some(
        (entry) =>
          entry.args[0] === "session" &&
          entry.args[1] === "event" &&
          entry.args[2] === "user_prompt_submit" &&
          entry.args.includes("--turn-id") &&
          entry.args.includes(turnId),
      ),
    );
    assert.ok(
      calls.some(
        (entry) =>
          entry.args[0] === "session" &&
          entry.args[1] === "event" &&
          entry.args[2] === "turn_started" &&
          entry.args.includes(turnId),
      ),
    );
    assert.ok(
      calls.some(
        (entry) =>
          entry.args[0] === "session" &&
          entry.args[1] === "event" &&
          entry.args[2] === "task_created" &&
          entry.args.includes(turnId),
      ),
    );
    assert.ok(
      calls.some(
        (entry) =>
          entry.args[0] === "session" &&
          entry.args[1] === "event" &&
          entry.args[2] === "task_completed" &&
          entry.args.includes(turnId),
      ),
    );
    assert.ok(
      calls.some(
        (entry) =>
          entry.args[0] === "session" &&
          entry.args[1] === "event" &&
          entry.args[2] === "turn_completed" &&
          entry.args.includes(turnId),
      ),
    );
  });

  it("emits task_failed when the CLI command errors", async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "afs-cli-client-fail-"));
    const logPath = path.join(tmpDir, "cli.log");
    const workspaceRoot = path.join(tmpDir, "workspace");
    const missingPath = path.join(workspaceRoot, ".context", "scratchpad", "missing.txt");
    fs.mkdirSync(path.dirname(missingPath), { recursive: true });
    workspace.workspaceFolders = [{ uri: { fsPath: workspaceRoot } }];

    const client = new CliClient(
      writeFakeAfsBinary(tmpDir, logPath),
      [],
      {
        CLI_TEST_LOG: logPath,
        CLI_TEST_ROOT: tmpDir,
      },
      {
        appendLine() {},
        dispose() {},
      } as never,
      5_000,
    );

    await client.initialize();
    await assert.rejects(() => client.callTool("context.read", { path: missingPath }));
    client.dispose();
    await new Promise((resolve) => setTimeout(resolve, 50));

    const calls = readLogEntries(logPath);
    assert.ok(calls.some((entry) => entry.args[0] === "session" && entry.args[1] === "event" && entry.args[2] === "task_failed"));
  });

  it("uses the canonical context query CLI path and preserves query options", async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "afs-cli-client-query-"));
    const logPath = path.join(tmpDir, "cli.log");
    const workspaceRoot = path.join(tmpDir, "workspace");
    fs.mkdirSync(path.join(workspaceRoot, ".context", "scratchpad"), { recursive: true });
    workspace.workspaceFolders = [{ uri: { fsPath: workspaceRoot } }];

    const client = new CliClient(
      writeFakeAfsBinary(tmpDir, logPath),
      [],
      {
        CLI_TEST_LOG: logPath,
        CLI_TEST_ROOT: tmpDir,
      },
      {
        appendLine() {},
        dispose() {},
      } as never,
      5_000,
    );

    await client.initialize();
    const result = await client.callTool("context.query", {
      context_path: path.join(workspaceRoot, ".context"),
      query: "needle",
      mount_types: ["scratchpad"],
      relative_prefix: "notes",
      limit: 7,
      include_content: true,
    });
    client.dispose();
    await new Promise((resolve) => setTimeout(resolve, 50));

    assert.strictEqual(Array.isArray(result.entries), true);
    assert.strictEqual(result.count, 1);

    const calls = readLogEntries(logPath);
    const queryCall = calls.find((entry) => entry.args[0] === "context" && entry.args[1] === "query");
    assert.ok(queryCall);
    assert.deepStrictEqual(queryCall.args.slice(0, 10), [
      "context",
      "query",
      "needle",
      "--path",
      workspaceRoot,
      "--json",
      "--mount",
      "scratchpad",
      "--limit",
      "7",
    ]);
    assert.ok(queryCall.args.includes("--prefix"));
    assert.ok(queryCall.args.includes("notes"));
    assert.ok(queryCall.args.includes("--include-content"));
  });
});
