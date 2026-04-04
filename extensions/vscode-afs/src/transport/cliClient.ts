import { execFile } from "child_process";
import { randomUUID } from "crypto";
import * as path from "path";
import * as vscode from "vscode";
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
  ServerCapabilities,
  SessionCliHints,
  TransportSessionInfo,
} from "./types";

interface ExecOptions {
  includeSessionEnv?: boolean;
}

/** Fallback transport that invokes the AFS CLI directly. Limited feature set. */
export class CliClient implements ITransportClient {
  private ready = false;
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
    private readonly args: string[],
    private readonly env: Record<string, string>,
    private readonly logger: vscode.OutputChannel,
    private readonly timeout: number = 30_000,
  ) {}

  async initialize(): Promise<void> {
    try {
      await this.exec(["--help"], { includeSessionEnv: false });
      await this.ensureSessionHarness();
      this.ready = true;
      this._onConnectionStateChanged.fire("connected");
    } catch (err) {
      this.ready = false;
      this._onConnectionStateChanged.fire("error");
      throw err;
    }
  }

  isReady(): boolean {
    return this.ready;
  }

  capabilities(): ServerCapabilities {
    return { tools: true, resources: false, prompts: false };
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<Record<string, unknown>> {
    await this.ensureSessionHarness();
    const taskId = this.nextTaskId();
    await this.recordToolEvent("task_created", name, taskId, `CLI tool started: ${name}`, "running");
    try {
      const result = await this.callToolImpl(name, args);
      await this.recordToolEvent("task_completed", name, taskId, `CLI tool completed: ${name}`, "completed");
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.recordToolEvent("task_failed", name, taskId, message, "failed", message);
      throw err;
    }
  }

  private async callToolImpl(name: string, args: Record<string, unknown>): Promise<Record<string, unknown>> {
    // Map MCP tool names to CLI commands
    switch (name) {
      case "context.discover": {
        const cliArgs = ["context", "discover", "--json"];
        const searchPaths = args.search_paths;
        if (Array.isArray(searchPaths)) {
          for (const item of searchPaths) {
            if (typeof item === "string" && item.trim()) {
              cliArgs.push("--path", item.trim());
            }
          }
        }
        if (typeof args.max_depth === "number") {
          cliArgs.push("--max-depth", String(args.max_depth));
        }
        const output = await this.exec(cliArgs);
        return JSON.parse(output);
      }
      case "context.init": {
        const projectPath = this.projectPathFromArgs(args.project_path);
        const cliArgs = ["context", "init", "--path", projectPath];
        if (args.force === true) cliArgs.push("--force");
        if (args.link_context === true) cliArgs.push("--link-context");
        if (typeof args.context_root === "string" && args.context_root.trim()) {
          cliArgs.push("--context-root", args.context_root.trim());
        }
        if (typeof args.context_dir === "string" && args.context_dir.trim()) {
          cliArgs.push("--context-dir", args.context_dir.trim());
        }
        if (typeof args.profile === "string" && args.profile.trim()) {
          cliArgs.push("--profile", args.profile.trim());
        }
        await this.exec(cliArgs);
        return {
          context_path:
            typeof args.context_root === "string" && args.context_root.trim()
              ? args.context_root.trim()
              : path.join(projectPath, ".context"),
          project: path.basename(projectPath),
          valid: true,
          mounts: 0,
        };
      }
      case "context.mount": {
        const source = args.source as string;
        const mountType = args.mount_type as string;
        if (!source || !mountType) {
          throw new Error("context.mount requires source and mount_type");
        }
        const projectPath = this.projectPathFromContextArg(args.context_path);
        const cliArgs = ["context", "mount", mountType, source, "--path", projectPath];
        if (typeof args.alias === "string" && args.alias.trim()) {
          cliArgs.push("--alias", args.alias.trim());
        }
        await this.exec(cliArgs);
        const alias =
          typeof args.alias === "string" && args.alias.trim()
            ? args.alias.trim()
            : path.basename(source);
        return {
          context_path: path.join(projectPath, ".context"),
          mount: {
            name: alias,
            mount_type: mountType,
            source,
            is_symlink: true,
          },
        };
      }
      case "context.unmount": {
        const mountType = args.mount_type as string;
        const alias = args.alias as string;
        if (!mountType || !alias) {
          throw new Error("context.unmount requires mount_type and alias");
        }
        const projectPath = this.projectPathFromContextArg(args.context_path);
        await this.exec([
          "context",
          "unmount",
          mountType,
          alias,
          "--path",
          projectPath,
        ]);
        return {
          context_path: path.join(projectPath, ".context"),
          mount_type: mountType,
          alias,
          removed: true,
        };
      }
      case "context.status": {
        const projectPath = this.projectPathFromContextArg(args.context_path);
        return this.execJson(["status", "--start-dir", projectPath, "--json"]);
      }
      case "context.freshness": {
        const projectPath = this.projectPathFromContextArg(args.context_path);
        const cliArgs = ["context", "freshness", "--path", projectPath, "--json"];
        if (typeof args.mount_type === "string" && args.mount_type.trim()) {
          cliArgs.push("--mount", args.mount_type.trim());
        }
        if (typeof args.decay_hours === "number") {
          cliArgs.push("--decay-hours", String(args.decay_hours));
        }
        if (typeof args.threshold === "number") {
          cliArgs.push("--threshold", String(args.threshold));
        }
        return this.execJson(cliArgs);
      }
      case "memory.status": {
        const projectPath = this.projectPathFromContextArg(args.context_path);
        return this.execJson(["memory", "status", "--path", projectPath, "--json"]);
      }
      case "agent.capabilities": {
        const cliArgs = ["agents", "capabilities", "--json"];
        if (typeof args.agent_name === "string" && args.agent_name.trim()) {
          cliArgs.push("--agent", args.agent_name.trim());
        }
        const agents = await this.execJson(cliArgs);
        return Array.isArray(agents) ? { agents, count: agents.length } : agents;
      }
      case "training.antigravity.status": {
        const cliArgs = ["training", "antigravity-status", "--json"];
        if (typeof args.db_path === "string" && args.db_path.trim()) {
          cliArgs.push("--db-path", args.db_path.trim());
        }
        const stateKeys = args.state_keys;
        if (Array.isArray(stateKeys)) {
          for (const key of stateKeys) {
            if (typeof key === "string" && key.trim()) {
              cliArgs.push("--state-key", key.trim());
            }
          }
        }
        return this.execJson(cliArgs);
      }
      case "context.index.rebuild": {
        const projectPath = this.projectPathFromContextArg(args.context_path);
        const cliArgs = ["index", "rebuild", "--path", projectPath, "--json"];
        if (Array.isArray(args.mount_types)) {
          for (const mt of args.mount_types) {
            if (typeof mt === "string" && mt.trim()) {
              cliArgs.push("--mount", mt.trim());
            }
          }
        }
        return this.execJson(cliArgs);
      }
      case "context.query": {
        const projectPath = this.projectPathFromContextArg(args.context_path);
        const query = args.query as string;
        if (!query) throw new Error("context.query requires a query string");
        const cliArgs = ["context", "query", query, "--path", projectPath, "--json"];
        if (Array.isArray(args.mount_types)) {
          for (const mt of args.mount_types) {
            if (typeof mt === "string" && mt.trim()) {
              cliArgs.push("--mount", mt.trim());
            }
          }
        }
        if (typeof args.limit === "number") {
          cliArgs.push("--limit", String(args.limit));
        }
        if (typeof args.relative_prefix === "string" && args.relative_prefix.trim()) {
          cliArgs.push("--prefix", args.relative_prefix.trim());
        }
        if (args.include_content === true) {
          cliArgs.push("--include-content");
        }
        return this.execJson(cliArgs);
      }
      case "context.read":
      case "fs.read": {
        const filePath = args.path as string;
        const parsed = this.parseMountPath(filePath);
        if (!parsed.relativePath) {
          throw new Error("context.read requires a file path under .context/<mount_type>/...");
        }
        const output = await this.exec([
          "fs",
          "read",
          parsed.mountType,
          parsed.relativePath,
          "--path",
          parsed.projectPath,
        ]);
        return { path: filePath, content: output };
      }
      case "context.write":
      case "fs.write": {
        const filePath = args.path as string;
        const content = args.content as string;
        if (typeof content !== "string") {
          throw new Error("context.write requires string content");
        }
        const parsed = this.parseMountPath(filePath);
        if (!parsed.relativePath) {
          throw new Error("context.write requires a file path under .context/<mount_type>/...");
        }
        const cliArgs = [
          "fs",
          "write",
          parsed.mountType,
          parsed.relativePath,
          "--path",
          parsed.projectPath,
          "--content",
          content,
        ];
        if (args.append === true) cliArgs.push("--append");
        if (args.mkdirs === true) cliArgs.push("--mkdirs");
        await this.exec(cliArgs);
        return { path: filePath, bytes: Buffer.byteLength(content, "utf-8") };
      }
      case "context.delete":
      case "fs.delete": {
        const filePath = args.path as string;
        const parsed = this.parseMountPath(filePath);
        if (!parsed.relativePath) {
          throw new Error("context.delete requires a file path under .context/<mount_type>/...");
        }
        const cliArgs = [
          "fs",
          "delete",
          parsed.mountType,
          parsed.relativePath,
          "--path",
          parsed.projectPath,
        ];
        if (args.recursive === true) cliArgs.push("--recursive");
        await this.exec(cliArgs);
        return { path: filePath, deleted: true, recursive: args.recursive === true };
      }
      case "context.move":
      case "fs.move": {
        const sourcePath = args.source as string;
        const destinationPath = args.destination as string;
        const sourceParsed = this.parseMountPath(sourcePath);
        const destinationParsed = this.parseMountPath(destinationPath);
        if (!sourceParsed.relativePath || !destinationParsed.relativePath) {
          throw new Error(
            "context.move requires source and destination paths under .context/<mount_type>/...",
          );
        }
        if (sourceParsed.projectPath !== destinationParsed.projectPath) {
          throw new Error("CLI transport only supports context.move within the same project");
        }
        const cliArgs = [
          "fs",
          "move",
          sourceParsed.mountType,
          sourceParsed.relativePath,
          destinationParsed.mountType,
          destinationParsed.relativePath,
          "--path",
          sourceParsed.projectPath,
        ];
        if (args.mkdirs === true) cliArgs.push("--mkdirs");
        await this.exec(cliArgs);
        return { source: sourcePath, destination: destinationPath };
      }
      case "context.list":
      case "fs.list": {
        const dirPath = args.path as string;
        const parsed = this.parseMountPath(dirPath);
        const cliArgs = ["fs", "list", parsed.mountType, "--path", parsed.projectPath, "--json"];
        if (parsed.relativePath) {
          cliArgs.push("--relative", parsed.relativePath);
        }
        if (typeof args.max_depth === "number") {
          cliArgs.push("--max-depth", String(args.max_depth));
        }
        const output = await this.exec(cliArgs);
        return JSON.parse(output);
      }
      default:
        throw new Error(`CLI transport does not support tool: ${name}`);
    }
  }

  async listTools(): Promise<ToolSpec[]> {
    // CLI mode has limited tool support
    return [
      { name: "context.discover", description: "Discover .context roots", inputSchema: {} },
      { name: "context.init", description: "Initialize context", inputSchema: {} },
      { name: "context.mount", description: "Mount a path into context", inputSchema: {} },
      { name: "context.unmount", description: "Unmount an alias from context", inputSchema: {} },
      { name: "context.status", description: "Get context status", inputSchema: {} },
      { name: "context.freshness", description: "Get mount freshness scores", inputSchema: {} },
      { name: "context.index.rebuild", description: "Rebuild context index", inputSchema: {} },
      { name: "context.query", description: "Query context index", inputSchema: {} },
      { name: "context.read", description: "Read a context-scoped file", inputSchema: {} },
      { name: "context.write", description: "Write a context-scoped file", inputSchema: {} },
      { name: "context.delete", description: "Delete a context-scoped file", inputSchema: {} },
      { name: "context.move", description: "Move a context-scoped file", inputSchema: {} },
      { name: "context.list", description: "List context-scoped files", inputSchema: {} },
      { name: "memory.status", description: "Get memory status", inputSchema: {} },
      { name: "agent.capabilities", description: "List agent capabilities", inputSchema: {} },
      { name: "training.antigravity.status", description: "Get Antigravity training status", inputSchema: {} },
    ];
  }

  async listResources(): Promise<McpResource[]> {
    return [];
  }

  async readResource(_uri: string): Promise<McpResourceContent> {
    throw new Error("CLI transport does not support resources");
  }

  async listPrompts(): Promise<McpPrompt[]> {
    return [];
  }

  async getPrompt(_name: string, _args?: Record<string, unknown>): Promise<McpPromptMessage[]> {
    throw new Error("CLI transport does not support prompts");
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
      this.logger.appendLine(`[cli harness] beginTurn failed: ${err}`);
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
      this.logger.appendLine(`[cli harness] completeTurn failed: ${err}`);
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
      this.logger.appendLine(`[cli harness] failTurn failed: ${err}`);
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
        this.logger.appendLine(`[cli harness] turn_failed failed: ${err}`);
      });
    }
    if (this.sessionId) {
      void this.runSessionHook("session_end", ["--reason", "client_dispose"]).catch((err) => {
        this.logger.appendLine(`[cli harness] session_end failed: ${err}`);
      });
    }
    this.ready = false;
    this._onConnectionStateChanged.dispose();
  }

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
      const payload = await this.execJson(
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
      this.logger.appendLine(`[cli harness] prepare-client failed: ${err}`);
      return;
    }

    try {
      await this.runSessionHook("session_start");
    } catch (err) {
      this.logger.appendLine(`[cli harness] session_start failed: ${err}`);
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
    await this.exec(args, { includeSessionEnv: false });
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
    await this.exec(args, { includeSessionEnv: false });
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
      this.logger.appendLine(`[cli harness] ${event} failed: ${err}`);
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

  private exec(extraArgs: string[], options: ExecOptions = {}): Promise<string> {
    return new Promise((resolve, reject) => {
      const allArgs = [...this.args, ...extraArgs];
      const env = {
        ...process.env,
        ...this.env,
        ...(options.includeSessionEnv === false ? {} : this.sessionEnv()),
      };
      execFile(
        this.command,
        allArgs,
        { env, timeout: this.timeout },
        (err, stdout, stderr) => {
          if (stderr) this.logger.appendLine(`[cli stderr] ${stderr.trimEnd()}`);
          if (err) return reject(err);
          resolve(stdout);
        },
      );
    });
  }

  private async execJson(extraArgs: string[], options: ExecOptions = {}): Promise<Record<string, unknown>> {
    const output = await this.exec(extraArgs, options);
    return JSON.parse(output) as Record<string, unknown>;
  }

  private parseMountPath(rawPath: string): {
    projectPath: string;
    mountType: string;
    relativePath: string;
  } {
    if (!rawPath || typeof rawPath !== "string") {
      throw new Error("path argument is required");
    }
    const resolved = path.resolve(rawPath);
    const marker = `${path.sep}.context${path.sep}`;
    const markerIndex = resolved.indexOf(marker);
    if (markerIndex < 0) {
      throw new Error(`Path must be inside a .context mount: ${resolved}`);
    }

    const projectPath = resolved.slice(0, markerIndex);
    const remainder = resolved.slice(markerIndex + marker.length);
    const parts = remainder.split(path.sep).filter(Boolean);
    if (parts.length === 0) {
      throw new Error(`Path must include mount type: ${resolved}`);
    }

    const mountType = parts[0];
    const relativePath = parts.slice(1).join("/");
    return { projectPath, mountType, relativePath };
  }

  private projectPathFromContextArg(rawContextPath: unknown): string {
    if (typeof rawContextPath !== "string" || !rawContextPath.trim()) {
      return process.cwd();
    }
    const resolved = path.resolve(rawContextPath);
    const marker = `${path.sep}.context`;
    if (resolved.endsWith(marker)) {
      return resolved.slice(0, -marker.length);
    }
    return path.dirname(resolved);
  }

  private projectPathFromArgs(rawProjectPath: unknown): string {
    if (typeof rawProjectPath === "string" && rawProjectPath.trim()) {
      return path.resolve(rawProjectPath);
    }
    return process.cwd();
  }
}
