import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import * as path from "node:path";
import * as vscode from "vscode";
import type { BinaryInfo } from "../transport/clientFactory";
import { getConfig } from "../utils/config";
import {
  buildMcpConfigCandidates,
  defaultWorkspaceMcpConfigPath,
  resolveExistingMcpConfigPath,
} from "./configCandidates";

export interface McpServerEntry {
  command: string;
  args?: string[];
  env?: Record<string, string>;
}

interface McpServersConfig {
  mcpServers?: Record<string, McpServerEntry>;
  [key: string]: unknown;
}

interface RegisterOptions {
  configPath?: string;
  interactivePathSelection?: boolean;
}

interface DetectPathOptions {
  preferExisting?: boolean;
}

interface ConfigPathInfo {
  overridePath: string | null;
  existingPath: string | null;
  defaultPath: string | null;
  candidates: string[];
}

const CONFIG_PATH_INFO_TTL_MS = 5000;
let configPathInfoCache:
  | { key: string; expiresAt: number; value: ConfigPathInfo }
  | undefined;

interface ReadConfigResult {
  exists: boolean;
  config: McpServersConfig;
  parseError?: string;
}

export interface RegistrationStatus {
  registered: boolean;
  configPath: string | null;
  entry?: McpServerEntry;
  parseError?: string;
}

export function detectMcpConfigPath(options: DetectPathOptions = {}): string | null {
  const { overridePath, existingPath, defaultPath } = getConfigPathInfo();
  if (overridePath) {
    return overridePath;
  }
  if (existingPath) {
    return existingPath;
  }
  if (options.preferExisting) {
    return null;
  }
  return defaultPath;
}

export async function chooseMcpConfigPath(): Promise<string | null> {
  const { overridePath, existingPath, defaultPath, candidates } = getConfigPathInfo();
  if (overridePath) {
    return overridePath;
  }

  const options = candidates.map((candidate) => ({
    label: shortenPath(candidate),
    description: describeCandidate(candidate, existingPath, defaultPath),
    detail: candidate,
  }));

  if (options.length === 0) {
    return null;
  }
  if (options.length === 1) {
    return options[0].detail;
  }

  const picked = await vscode.window.showQuickPick(options, {
    placeHolder: "Choose the MCP config file to update",
  });
  return picked?.detail ?? null;
}

export function buildServerEntry(binaryInfo: BinaryInfo): McpServerEntry {
  const extraArgs = getConfig<string[]>("server.args", [])
    .map((value) => value.trim())
    .filter(Boolean);
  const extraEnv = normalizeEnv(getConfig<Record<string, unknown>>("server.env", {}));
  const env = normalizeEnv({ ...binaryInfo.env, ...extraEnv });

  return {
    command: binaryInfo.command,
    args: [...binaryInfo.args, "mcp", "serve", ...extraArgs],
    ...(Object.keys(env).length > 0 ? { env } : {}),
  };
}

export function mergeMcpServerConfig(
  existing: McpServersConfig,
  entry: McpServerEntry,
): McpServersConfig {
  return {
    ...existing,
    mcpServers: {
      ...(existing.mcpServers ?? {}),
      afs: entry,
    },
  };
}

export async function registerAfs(
  binaryInfo: BinaryInfo,
  logger: vscode.OutputChannel,
  options: RegisterOptions = {},
): Promise<boolean> {
  const configPath = await resolveRegistrationTarget(options);
  if (!configPath) {
    vscode.window.showErrorMessage(
      "Could not determine MCP config path. Set `afs.mcp.configPath` or choose a target config.",
    );
    return false;
  }

  const loaded = readConfigFile(configPath);
  if (loaded.parseError) {
    vscode.window.showErrorMessage(loaded.parseError);
    logger.appendLine(`[mcp] ${loaded.parseError}`);
    return false;
  }

  const entry = buildServerEntry(binaryInfo);
  const currentEntry = loaded.config.mcpServers?.afs;
  if (currentEntry && entriesEqual(currentEntry, entry)) {
    vscode.window.showInformationMessage(`AFS is already registered in ${configPath}`);
    return true;
  }

  const newConfig = mergeMcpServerConfig(loaded.config, entry);
  const choice = await vscode.window.showInformationMessage(
    `Register AFS MCP server in ${configPath}?`,
    {
      modal: true,
      detail: buildRegistrationPreview(configPath, entry, loaded.exists, !!currentEntry),
    },
    currentEntry ? "Update" : "Register",
    "Cancel",
  );

  if (choice !== "Register" && choice !== "Update") {
    return false;
  }

  backupFileIfPresent(configPath, logger);
  ensureParentDirectory(configPath);
  writeFileSync(configPath, JSON.stringify(newConfig, null, 2), "utf-8");
  invalidateConfigPathInfoCache();
  logger.appendLine(`[mcp] Registered AFS in ${configPath}`);
  vscode.window.showInformationMessage(`AFS registered in ${configPath}`);
  return true;
}

export async function unregisterAfs(
  logger: vscode.OutputChannel,
): Promise<boolean> {
  const configPath = detectMcpConfigPath({ preferExisting: true }) ?? detectMcpConfigPath();
  if (!configPath || !existsSync(configPath)) {
    vscode.window.showInformationMessage("No MCP config found to unregister from.");
    return false;
  }

  const loaded = readConfigFile(configPath);
  if (loaded.parseError) {
    vscode.window.showErrorMessage(loaded.parseError);
    logger.appendLine(`[mcp] ${loaded.parseError}`);
    return false;
  }

  if (!loaded.config.mcpServers?.afs) {
    vscode.window.showInformationMessage("AFS is not registered in MCP config.");
    return false;
  }

  const choice = await vscode.window.showWarningMessage(
    `Remove AFS from ${configPath}?`,
    {
      modal: true,
      detail: "The existing file will be backed up before removing the AFS MCP entry.",
    },
    "Remove",
    "Cancel",
  );

  if (choice !== "Remove") {
    return false;
  }

  backupFileIfPresent(configPath, logger);
  delete loaded.config.mcpServers.afs;
  writeFileSync(configPath, JSON.stringify(loaded.config, null, 2), "utf-8");
  invalidateConfigPathInfoCache();
  logger.appendLine(`[mcp] Unregistered AFS from ${configPath}`);
  vscode.window.showInformationMessage(`AFS removed from ${configPath}`);
  return true;
}

export function checkRegistration(): RegistrationStatus {
  const configPath = detectMcpConfigPath();
  if (!configPath || !existsSync(configPath)) {
    return { registered: false, configPath };
  }

  const loaded = readConfigFile(configPath);
  if (loaded.parseError) {
    return {
      registered: false,
      configPath,
      parseError: loaded.parseError,
    };
  }

  const entry = loaded.config.mcpServers?.afs;
  return {
    registered: !!entry,
    configPath,
    ...(entry ? { entry } : {}),
  };
}

async function resolveRegistrationTarget(
  options: RegisterOptions,
): Promise<string | null> {
  if (options.configPath?.trim()) {
    return options.configPath.trim();
  }

  if (options.interactivePathSelection === false) {
    return detectMcpConfigPath();
  }

  return chooseMcpConfigPath();
}

function getConfigPathInfo(): ConfigPathInfo {
  const overridePath = getConfig<string>("mcp.configPath", "").trim() || null;
  const workspaceFolders = (vscode.workspace.workspaceFolders ?? []).map(
    (folder) => folder.uri.fsPath,
  );
  const home = process.env.HOME ?? process.env.USERPROFILE ?? "";
  const antigravityContextRoot = home ? resolveConfiguredContextRoot(home) : "";
  const cacheKey = [
    overridePath ?? "",
    home,
    antigravityContextRoot,
    ...workspaceFolders,
  ].join("::");
  const now = Date.now();
  if (configPathInfoCache && configPathInfoCache.key === cacheKey && configPathInfoCache.expiresAt > now) {
    return configPathInfoCache.value;
  }

  let info: ConfigPathInfo;
  if (overridePath) {
    info = {
      overridePath,
      existingPath: existsSync(overridePath) ? overridePath : null,
      defaultPath: overridePath,
      candidates: [overridePath],
    };
    configPathInfoCache = {
      key: cacheKey,
      expiresAt: now + CONFIG_PATH_INFO_TTL_MS,
      value: info,
    };
    return info;
  }

  const candidates = home
    ? buildMcpConfigCandidates({
        home,
        workspaceFolders,
        antigravityContextRoot,
      })
    : [];
  const existingPath = resolveExistingMcpConfigPath(candidates, existsSync);
  const defaultPath = defaultWorkspaceMcpConfigPath(workspaceFolders)
    ?? candidates[0]
    ?? null;

  info = {
    overridePath: null,
    existingPath,
    defaultPath,
    candidates: orderCandidates(candidates, existingPath, defaultPath),
  };
  configPathInfoCache = {
    key: cacheKey,
    expiresAt: now + CONFIG_PATH_INFO_TTL_MS,
    value: info,
  };
  return info;
}

function resolveConfiguredContextRoot(home: string): string {
  return getConfig<string>("mcp.contextRoot", "").trim()
    || getConfig<string>("antigravity.contextRoot", "").trim()
    || path.join(home, ".gemini", "antigravity");
}

function orderCandidates(
  candidates: string[],
  existingPath: string | null,
  defaultPath: string | null,
): string[] {
  const ordered = new Map<string, string>();
  if (existingPath) {
    ordered.set(existingPath, existingPath);
  }
  if (defaultPath) {
    ordered.set(defaultPath, defaultPath);
  }
  for (const candidate of candidates) {
    ordered.set(candidate, candidate);
  }
  return Array.from(ordered.values());
}

function describeCandidate(
  candidate: string,
  existingPath: string | null,
  defaultPath: string | null,
): string {
  if (candidate === existingPath) {
    return "existing config";
  }
  if (candidate === defaultPath) {
    return "default target";
  }
  return "available target";
}

function shortenPath(filePath: string): string {
  const home = process.env.HOME ?? process.env.USERPROFILE ?? "";
  return home && filePath.startsWith(home)
    ? `~${filePath.slice(home.length)}`
    : filePath;
}

function normalizeEnv(values: Record<string, unknown>): Record<string, string> {
  return Object.fromEntries(
    Object.entries(values)
      .filter(([, value]) => value != null && `${value}`.trim())
      .map(([key, value]) => [key, String(value)]),
  );
}

function entriesEqual(a: McpServerEntry, b: McpServerEntry): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

function buildRegistrationPreview(
  configPath: string,
  entry: McpServerEntry,
  exists: boolean,
  updating: boolean,
): string {
  const envKeys = Object.keys(entry.env ?? {});
  return [
    `Target: ${configPath}`,
    `Config file: ${exists ? "existing" : "new"}`,
    `Action: ${updating ? "update existing AFS entry" : "add AFS entry"}`,
    `Command: ${entry.command}`,
    `Args: ${(entry.args ?? []).join(" ") || "(none)"}`,
    `Env: ${envKeys.length > 0 ? envKeys.join(", ") : "(none)"}`,
  ].join("\n");
}

function readConfigFile(configPath: string): ReadConfigResult {
  if (!existsSync(configPath)) {
    return { exists: false, config: {} };
  }

  try {
    const parsed = JSON.parse(readFileSync(configPath, "utf-8")) as McpServersConfig;
    return { exists: true, config: parsed };
  } catch {
    return {
      exists: true,
      config: {},
      parseError: `Could not parse MCP config at ${configPath}. Fix the JSON or choose another target.`,
    };
  }
}

function backupFileIfPresent(configPath: string, logger: vscode.OutputChannel): void {
  if (!existsSync(configPath)) {
    return;
  }
  const backupPath = `${configPath}.${Date.now()}.backup`;
  copyFileSync(configPath, backupPath);
  logger.appendLine(`[mcp] Backed up ${configPath} -> ${backupPath}`);
}

function ensureParentDirectory(configPath: string): void {
  const dir = path.dirname(configPath);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
}

function invalidateConfigPathInfoCache(): void {
  configPathInfoCache = undefined;
}
