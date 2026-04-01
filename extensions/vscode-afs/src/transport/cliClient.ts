import { execFile } from "child_process";
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
import type { ITransportClient, ServerCapabilities } from "./types";

/** Fallback transport that invokes the AFS CLI directly. Limited feature set. */
export class CliClient implements ITransportClient {
  private ready = false;
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
      await this.exec(["--help"]);
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
      { name: "context.read", description: "Read a context-scoped file", inputSchema: {} },
      { name: "context.write", description: "Write a context-scoped file", inputSchema: {} },
      { name: "context.delete", description: "Delete a context-scoped file", inputSchema: {} },
      { name: "context.move", description: "Move a context-scoped file", inputSchema: {} },
      { name: "context.list", description: "List context-scoped files", inputSchema: {} },
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

  dispose(): void {
    this.ready = false;
    this._onConnectionStateChanged.dispose();
  }

  private exec(extraArgs: string[]): Promise<string> {
    return new Promise((resolve, reject) => {
      const allArgs = [...this.args, ...extraArgs];
      execFile(
        this.command,
        allArgs,
        { env: { ...process.env, ...this.env }, timeout: this.timeout },
        (err, stdout, stderr) => {
          if (stderr) this.logger.appendLine(`[cli stderr] ${stderr.trimEnd()}`);
          if (err) return reject(err);
          resolve(stdout);
        },
      );
    });
  }

  private async execJson(extraArgs: string[]): Promise<Record<string, unknown>> {
    const output = await this.exec(extraArgs);
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
