import { ChildProcess, execFile, spawn } from "child_process";
import { randomUUID } from "crypto";
import * as path from "path";
import * as vscode from "vscode";
import { MCP_PROTOCOL_VERSION } from "../constants";
import type {
  ConnectionState,
  McpPrompt,
  McpPromptMessage,
  McpResource,
  McpResourceContent,
  ToolSpec,
} from "../types";
import type {
  ITransportClient,
  JsonRpcResponse,
  ServerCapabilities,
  SessionCliHints,
  TransportSessionInfo,
} from "./types";

interface PendingRequest {
  resolve: (value: JsonRpcResponse) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

interface ExecCliOptions {
  includeSessionEnv?: boolean;
}

export class McpStdioClient implements ITransportClient {
  private process: ChildProcess | null = null;
  private nextId = 1;
  private pending = new Map<number, PendingRequest>();
  private buffer = Buffer.alloc(0);
  private ready = false;
  private caps: ServerCapabilities = { tools: false, resources: false, prompts: false };
  private sessionId = "";
  private sessionPayloadFile = "";
  private sessionContextPath = "";
  private sessionPromptJson = "";
  private sessionPromptText = "";
  private sessionWorkspace = "";
  private sessionCliHints: SessionCliHints = this.defaultCliHints("");
  private toolTaskCounter = 0;
  private turnCounter = 0;
  private activeTurnId = "";

  private readonly _onConnectionStateChanged = new vscode.EventEmitter<ConnectionState>();
  readonly onConnectionStateChanged = this._onConnectionStateChanged.event;

  constructor(
    private readonly command: string,
    private readonly cliArgs: string[],
    private readonly mcpArgs: string[],
    private readonly env: Record<string, string>,
    private readonly logger: vscode.OutputChannel,
    private readonly timeout: number = 30_000,
  ) {}

  async initialize(): Promise<void> {
    await this.ensureSessionHarness();

    this.process = spawn(this.command, this.mcpArgs, {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, ...this.env, ...this.sessionEnv() },
    });

    this.process.stdout!.on("data", (chunk: Buffer) => this.onData(chunk));
    this.process.stderr!.on("data", (chunk: Buffer) => {
      this.logger.appendLine(`[stderr] ${chunk.toString("utf-8").trimEnd()}`);
    });
    this.process.on("exit", (code) => {
      this.ready = false;
      this.rejectAll(new Error(`AFS server exited with code ${code}`));
      this._onConnectionStateChanged.fire("disconnected");
    });
    this.process.on("error", (err) => {
      this.ready = false;
      this.rejectAll(err);
      this._onConnectionStateChanged.fire("error");
    });

    // MCP handshake
    const initResult = await this.sendRequest("initialize", {
      protocolVersion: MCP_PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: { name: "afs-vscode", version: "0.1.0" },
    });

    const serverCaps = (initResult.result as Record<string, unknown>)?.capabilities;
    if (serverCaps && typeof serverCaps === "object") {
      const capsObj = serverCaps as Record<string, unknown>;
      this.caps = {
        tools: !!capsObj.tools,
        resources: !!capsObj.resources,
        prompts: !!capsObj.prompts,
      };
    }

    // Send initialized notification (no response expected)
    this.sendNotification("notifications/initialized", {});

    this.ready = true;
    this._onConnectionStateChanged.fire("connected");
  }

  isReady(): boolean {
    return this.ready;
  }

  capabilities(): ServerCapabilities {
    return { ...this.caps };
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<Record<string, unknown>> {
    await this.ensureSessionHarness();
    const taskId = this.nextTaskId();
    await this.recordToolEvent("task_created", name, taskId, `MCP tool started: ${name}`, "running");
    try {
      const resp = await this.sendRequest("tools/call", { name, arguments: args });
      if (resp.error) {
        throw new Error(resp.error.message);
      }
      const result = resp.result ?? {};
      await this.recordToolEvent(
        "task_completed",
        name,
        taskId,
        `MCP tool completed: ${name}`,
        "completed",
      );
      // Extract structuredContent if present (AFS wraps tool results)
      if ("structuredContent" in result && typeof result.structuredContent === "object") {
        return result.structuredContent as Record<string, unknown>;
      }
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.recordToolEvent("task_failed", name, taskId, message, "failed", message);
      throw err;
    }
  }

  async listTools(): Promise<ToolSpec[]> {
    const resp = await this.sendRequest("tools/list", {});
    if (resp.error) throw new Error(resp.error.message);
    return ((resp.result as Record<string, unknown>)?.tools ?? []) as ToolSpec[];
  }

  async listResources(): Promise<McpResource[]> {
    if (!this.caps.resources) return [];
    const resp = await this.sendRequest("resources/list", {});
    if (resp.error) throw new Error(resp.error.message);
    return ((resp.result as Record<string, unknown>)?.resources ?? []) as McpResource[];
  }

  async readResource(uri: string): Promise<McpResourceContent> {
    if (!this.caps.resources) throw new Error("Server does not support resources");
    const resp = await this.sendRequest("resources/read", { uri });
    if (resp.error) throw new Error(resp.error.message);
    const contents = (resp.result as Record<string, unknown>)?.contents;
    if (Array.isArray(contents) && contents.length > 0) {
      return contents[0] as McpResourceContent;
    }
    return { uri };
  }

  async listPrompts(): Promise<McpPrompt[]> {
    if (!this.caps.prompts) return [];
    const resp = await this.sendRequest("prompts/list", {});
    if (resp.error) throw new Error(resp.error.message);
    return ((resp.result as Record<string, unknown>)?.prompts ?? []) as McpPrompt[];
  }

  async getPrompt(name: string, args?: Record<string, unknown>): Promise<McpPromptMessage[]> {
    if (!this.caps.prompts) throw new Error("Server does not support prompts");
    const resp = await this.sendRequest("prompts/get", { name, arguments: args ?? {} });
    if (resp.error) throw new Error(resp.error.message);
    return ((resp.result as Record<string, unknown>)?.messages ?? []) as McpPromptMessage[];
  }

  async beginTurn(prompt: string, summary?: string): Promise<string> {
    await this.ensureSessionHarness();
    if (!this.sessionId) {
      return "";
    }

    const turnId = this.nextTurnId();
    const promptText = prompt.trim();
    const turnSummary = ((summary ?? promptText) || "VS Code command").trim();
    try {
      await this.runSessionEvent("user_prompt_submit", [
        "--turn-id",
        turnId,
        "--prompt",
        promptText || prompt,
      ]);
      this.activeTurnId = turnId;
      const args = ["--turn-id", turnId];
      if (turnSummary) {
        args.push("--summary", turnSummary);
      }
      await this.runSessionEvent("turn_started", args);
      return turnId;
    } catch (err) {
      this.activeTurnId = "";
      this.logger.appendLine(`[mcp harness] beginTurn failed: ${err}`);
      return "";
    }
  }

  async completeTurn(turnId: string, summary?: string): Promise<void> {
    if (!turnId) {
      return;
    }
    try {
      const args = ["--turn-id", turnId];
      if (summary?.trim()) {
        args.push("--summary", summary.trim());
      }
      args.push("--reason", "vscode_command");
      await this.runSessionEvent("turn_completed", args);
    } catch (err) {
      this.logger.appendLine(`[mcp harness] completeTurn failed: ${err}`);
    } finally {
      if (this.activeTurnId === turnId) {
        this.activeTurnId = "";
      }
    }
  }

  async failTurn(turnId: string, error: unknown, summary?: string): Promise<void> {
    if (!turnId) {
      return;
    }
    try {
      const args = ["--turn-id", turnId];
      if (summary?.trim()) {
        args.push("--summary", summary.trim());
      }
      const reason = error instanceof Error ? error.message : String(error);
      if (reason.trim()) {
        args.push("--reason", reason.trim());
      }
      await this.runSessionEvent("turn_failed", args);
    } catch (err) {
      this.logger.appendLine(`[mcp harness] failTurn failed: ${err}`);
    } finally {
      if (this.activeTurnId === turnId) {
        this.activeTurnId = "";
      }
    }
  }

  getSessionInfo(): TransportSessionInfo | undefined {
    if (!this.sessionId) {
      return undefined;
    }
    return {
      sessionId: this.sessionId,
      payloadFile: this.sessionPayloadFile,
      contextPath: this.sessionContextPath,
      promptJson: this.sessionPromptJson,
      promptText: this.sessionPromptText,
      workspace: this.sessionWorkspace || this.workspaceRoot(),
      cliHints: { ...this.sessionCliHints, notes: [...this.sessionCliHints.notes] },
    };
  }

  dispose(): void {
    if (this.activeTurnId) {
      const turnId = this.activeTurnId;
      this.activeTurnId = "";
      void this.runSessionEvent("turn_failed", [
        "--turn-id",
        turnId,
        "--summary",
        "VS Code command interrupted by client dispose",
        "--reason",
        "client_dispose",
      ]).catch((err) => {
        this.logger.appendLine(`[mcp harness] turn_failed failed: ${err}`);
      });
    }
    if (this.sessionId) {
      void this.runSessionHook("session_end", ["--reason", "client_dispose"]).catch((err) => {
        this.logger.appendLine(`[mcp harness] session_end failed: ${err}`);
      });
    }
    this.rejectAll(new Error("Client disposed"));
    if (this.process) {
      this.process.kill();
      this.process = null;
    }
    this.ready = false;
    this._onConnectionStateChanged.dispose();
  }

  // --- Private ---

  private workspaceRoot(): string {
    const folder = vscode.workspace.workspaceFolders?.[0];
    return folder?.uri?.fsPath ? path.resolve(folder.uri.fsPath) : process.cwd();
  }

  private nextTaskId(): string {
    this.toolTaskCounter += 1;
    return `vscode-tool-${this.toolTaskCounter}`;
  }

  private nextTurnId(): string {
    this.turnCounter += 1;
    return `vscode-turn-${this.turnCounter}`;
  }

  private sessionEnv(): Record<string, string> {
    const env: Record<string, string> = {};
    if (this.sessionId) env.AFS_SESSION_ID = this.sessionId;
    if (this.sessionPayloadFile) env.AFS_SESSION_CLIENT_PAYLOAD_JSON = this.sessionPayloadFile;
    if (this.sessionContextPath) env.AFS_ACTIVE_CONTEXT_ROOT = this.sessionContextPath;
    if (this.sessionPromptJson) env.AFS_SESSION_SYSTEM_PROMPT_JSON = this.sessionPromptJson;
    if (this.sessionPromptText) env.AFS_SESSION_SYSTEM_PROMPT_TEXT = this.sessionPromptText;
    if (this.sessionCliHints.queryShortcut) env.AFS_SESSION_QUERY_HINT = this.sessionCliHints.queryShortcut;
    if (this.sessionCliHints.queryCanonical) {
      env.AFS_SESSION_CONTEXT_QUERY_HINT = this.sessionCliHints.queryCanonical;
    }
    if (this.sessionCliHints.indexRebuild) {
      env.AFS_SESSION_INDEX_REBUILD_HINT = this.sessionCliHints.indexRebuild;
    }
    if (this.activeTurnId) env.AFS_SESSION_DEFAULT_TURN_ID = this.activeTurnId;
    return env;
  }

  private async ensureSessionHarness(): Promise<void> {
    if (this.sessionId) {
      return;
    }

    this.sessionId = randomUUID().replace(/-/g, "").slice(0, 12);
    this.sessionWorkspace = this.workspaceRoot();
    this.sessionCliHints = this.defaultCliHints(this.sessionWorkspace);

    try {
      const payload = await this.execCliJson(
        [
          "session",
          "prepare-client",
          "--client",
          "vscode",
          "--session-id",
          this.sessionId,
          "--cwd",
          this.sessionWorkspace,
          "--path",
          this.sessionWorkspace,
          "--model",
          "generic",
          "--workflow",
          "general",
          "--tool-profile",
          "context_readonly",
          "--pack-mode",
          "focused",
          "--task",
          "VS Code AFS transport session",
          "--skills-prompt",
          "vscode ide transport context filesystem",
          "--json",
        ],
        { includeSessionEnv: false },
      );

      this.sessionPayloadFile = this.payloadArtifactPath(payload);
      this.sessionContextPath = this.stringValue(payload.context_path);
      const promptArtifacts = this.artifactPaths((payload.prompt as Record<string, unknown> | undefined) ?? {});
      this.sessionPromptJson = promptArtifacts.json ?? "";
      this.sessionPromptText = promptArtifacts.text ?? "";
      this.sessionCliHints = this.parseCliHints(payload.cli_hints, this.sessionWorkspace);
    } catch (err) {
      this.logger.appendLine(`[mcp harness] prepare-client failed: ${err}`);
      return;
    }

    try {
      await this.runSessionHook("session_start");
    } catch (err) {
      this.logger.appendLine(`[mcp harness] session_start failed: ${err}`);
    }
  }

  private async runSessionHook(event: string, extraArgs: string[] = []): Promise<void> {
    if (!this.sessionId) {
      return;
    }
    const args = [
      "session",
      "hook",
      event,
      "--client",
      "vscode",
      "--session-id",
      this.sessionId,
      "--cwd",
      this.sessionWorkspace || this.workspaceRoot(),
      "--path",
      this.sessionWorkspace || this.workspaceRoot(),
    ];
    if (this.sessionPayloadFile) {
      args.push("--payload-file", this.sessionPayloadFile);
    }
    args.push(...extraArgs);
    await this.execCli(args, { includeSessionEnv: false });
  }

  private async runSessionEvent(event: string, extraArgs: string[] = []): Promise<void> {
    if (!this.sessionId) {
      return;
    }
    const eventArgs = [...extraArgs];
    if (this.activeTurnId && !this.hasArg(eventArgs, "--turn-id")) {
      eventArgs.push("--turn-id", this.activeTurnId);
    }
    const args = [
      "session",
      "event",
      event,
      "--client",
      "vscode",
      "--session-id",
      this.sessionId,
      "--cwd",
      this.sessionWorkspace || this.workspaceRoot(),
      "--path",
      this.sessionWorkspace || this.workspaceRoot(),
    ];
    if (this.sessionPayloadFile) {
      args.push("--payload-file", this.sessionPayloadFile);
    }
    args.push(...eventArgs);
    await this.execCli(args, { includeSessionEnv: false });
  }

  private async recordToolEvent(
    event: "task_created" | "task_completed" | "task_failed",
    toolName: string,
    taskId: string,
    summary: string,
    status: string,
    reason = "",
  ): Promise<void> {
    try {
      const args = [
        "--task-id",
        taskId,
        "--task-title",
        toolName,
        "--summary",
        summary,
        "--status",
        status,
      ];
      if (reason) {
        args.push("--reason", reason);
      }
      await this.runSessionEvent(event, args);
    } catch (err) {
      this.logger.appendLine(`[mcp harness] ${event} failed: ${err}`);
    }
  }

  private payloadArtifactPath(payload: Record<string, unknown>): string {
    const raw = payload.artifact_paths;
    if (!raw || typeof raw !== "object") {
      return "";
    }
    return this.stringValue((raw as Record<string, unknown>).json);
  }

  private artifactPaths(section: Record<string, unknown>): Record<string, string | undefined> {
    const raw = section.artifact_paths;
    if (!raw || typeof raw !== "object") {
      return {};
    }
    const entries = raw as Record<string, unknown>;
    return {
      json: this.stringValue(entries.json),
      text: this.stringValue(entries.text),
      markdown: this.stringValue(entries.markdown),
    };
  }

  private stringValue(value: unknown): string {
    return typeof value === "string" ? value : "";
  }

  private parseCliHints(value: unknown, workspace: string): SessionCliHints {
    const fallback = this.defaultCliHints(workspace);
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      return fallback;
    }
    const raw = value as Record<string, unknown>;
    return {
      workspacePath: this.stringValue(raw.workspace_path).trim() || fallback.workspacePath,
      queryShortcut: this.stringValue(raw.query_shortcut).trim() || fallback.queryShortcut,
      queryCanonical: this.stringValue(raw.query_canonical).trim() || fallback.queryCanonical,
      indexRebuild: this.stringValue(raw.index_rebuild).trim() || fallback.indexRebuild,
      notes: Array.isArray(raw.notes)
        ? raw.notes
            .filter((entry): entry is string => typeof entry === "string")
            .map((entry) => entry.trim())
            .filter(Boolean)
        : fallback.notes,
    };
  }

  private defaultCliHints(workspace: string): SessionCliHints {
    const resolvedWorkspace = (workspace || this.workspaceRoot()).trim();
    const quotedWorkspace = this.shellQuote(resolvedWorkspace);
    return {
      workspacePath: resolvedWorkspace,
      queryShortcut: resolvedWorkspace ? `afs query <text> --path ${quotedWorkspace}` : "",
      queryCanonical: resolvedWorkspace
        ? `afs context query <text> --path ${quotedWorkspace}`
        : "",
      indexRebuild: resolvedWorkspace ? `afs index rebuild --path ${quotedWorkspace}` : "",
      notes: [],
    };
  }

  private shellQuote(value: string): string {
    if (!value) {
      return "''";
    }
    if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(value)) {
      return value;
    }
    return `'${value.replace(/'/g, `'\\''`)}'`;
  }

  private hasArg(args: string[], flag: string): boolean {
    return args.includes(flag);
  }

  private execCli(extraArgs: string[], options: ExecCliOptions = {}): Promise<string> {
    return new Promise((resolve, reject) => {
      const env = {
        ...process.env,
        ...this.env,
        ...(options.includeSessionEnv === false ? {} : this.sessionEnv()),
      };
      execFile(
        this.command,
        [...this.cliArgs, ...extraArgs],
        { env, timeout: this.timeout },
        (err, stdout, stderr) => {
          if (stderr) this.logger.appendLine(`[cli stderr] ${stderr.trimEnd()}`);
          if (err) return reject(err);
          resolve(stdout);
        },
      );
    });
  }

  private async execCliJson(
    extraArgs: string[],
    options: ExecCliOptions = {},
  ): Promise<Record<string, unknown>> {
    const output = await this.execCli(extraArgs, options);
    return JSON.parse(output) as Record<string, unknown>;
  }

  private sendRequest(method: string, params: Record<string, unknown>): Promise<JsonRpcResponse> {
    return new Promise((resolve, reject) => {
      if (!this.process?.stdin?.writable) {
        return reject(new Error("Transport not connected"));
      }
      const id = this.nextId++;
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Request ${method} (id=${id}) timed out after ${this.timeout}ms`));
      }, this.timeout);

      this.pending.set(id, { resolve, reject, timer });
      this.writeMessage({ jsonrpc: "2.0", id, method, params });
    });
  }

  private sendNotification(method: string, params: Record<string, unknown>): void {
    if (!this.process?.stdin?.writable) return;
    this.writeMessage({ jsonrpc: "2.0", method, params });
  }

  private writeMessage(payload: Record<string, unknown>): void {
    const body = Buffer.from(JSON.stringify(payload), "utf-8");
    const header = Buffer.from(`Content-Length: ${body.length}\r\n\r\n`, "ascii");
    this.process!.stdin!.write(header);
    this.process!.stdin!.write(body);
  }

  private onData(chunk: Buffer): void {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    while (this.tryParseMessage()) {
      // keep parsing
    }
  }

  private tryParseMessage(): boolean {
    const headerEnd = this.buffer.indexOf("\r\n\r\n");
    if (headerEnd === -1) return false;

    const headerStr = this.buffer.subarray(0, headerEnd).toString("ascii");
    const match = /content-length:\s*(\d+)/i.exec(headerStr);
    if (!match) {
      // Discard malformed header
      this.buffer = this.buffer.subarray(headerEnd + 4);
      return true;
    }

    const contentLength = parseInt(match[1], 10);
    const bodyStart = headerEnd + 4;
    if (this.buffer.length < bodyStart + contentLength) return false;

    const bodyBuf = this.buffer.subarray(bodyStart, bodyStart + contentLength);
    this.buffer = this.buffer.subarray(bodyStart + contentLength);

    try {
      const msg = JSON.parse(bodyBuf.toString("utf-8")) as JsonRpcResponse;
      this.handleResponse(msg);
    } catch {
      this.logger.appendLine("[warn] Failed to parse MCP message");
    }
    return true;
  }

  private handleResponse(msg: JsonRpcResponse): void {
    if (msg.id == null) return; // notification, ignore
    const id = typeof msg.id === "string" ? parseInt(msg.id, 10) : msg.id;
    const pending = this.pending.get(id);
    if (!pending) return;
    this.pending.delete(id);
    clearTimeout(pending.timer);
    pending.resolve(msg);
  }

  private rejectAll(err: Error): void {
    for (const [id, pending] of this.pending) {
      clearTimeout(pending.timer);
      pending.reject(err);
      this.pending.delete(id);
    }
  }
}
