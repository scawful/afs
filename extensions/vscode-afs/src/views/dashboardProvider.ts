import * as path from "node:path";
import * as vscode from "vscode";
import { buildServerEntry, checkRegistration, type McpServerEntry, type RegistrationStatus } from "../mcp/registration";
import type { BinaryInfo } from "../transport/clientFactory";
import type { ITransportClient, TransportSessionInfo } from "../transport/types";
import { extractToolPayload } from "../utils/toolPayload";
import { resolvePreferredContextPath } from "../utils/workspace";

export class AfsDashboardProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "afs.dashboard";

  private view?: vscode.WebviewView;
  private renderToken = 0;

  constructor(
    private readonly transport: ITransportClient,
    private readonly logger: vscode.OutputChannel,
    private readonly binaryInfo: BinaryInfo,
  ) {}

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken,
  ): void {
    this.view = webviewView;
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.onDidReceiveMessage(async (msg) => {
      switch (msg.command) {
        case "refresh":
          await this.updateContent();
          break;
        case "rebuildIndex":
          await vscode.commands.executeCommand("afs.index.rebuild");
          await this.updateContent();
          break;
        case "queryIndex":
          await vscode.commands.executeCommand("afs.index.query");
          break;
        case "mcpStatus":
          await vscode.commands.executeCommand("afs.mcp.status");
          break;
        case "showLogs":
          await vscode.commands.executeCommand("afs.server.showLogs");
          break;
        case "copyText":
          if (typeof msg.text === "string" && msg.text.trim()) {
            await vscode.env.clipboard.writeText(msg.text);
            const label =
              typeof msg.label === "string" && msg.label.trim()
                ? msg.label.trim()
                : "command";
            vscode.window.showInformationMessage(`Copied ${label} to clipboard.`);
          }
          break;
        case "openCommand":
          if (typeof msg.id === "string" && msg.id.trim()) {
            await vscode.commands.executeCommand(msg.id);
            await this.updateContent();
          }
          break;
      }
    });
    void this.updateContent();
  }

  async refresh(): Promise<void> {
    await this.updateContent();
  }

  private async updateContent(): Promise<void> {
    if (!this.view) {
      return;
    }

    const token = ++this.renderToken;
    const connected = this.transport.isReady();
    const caps = this.transport.capabilities();
    const sessionInfo = this.transport.getSessionInfo();
    const registration = checkRegistration();
    const serverEntry = buildServerEntry(this.binaryInfo);

    this.view.webview.html = this.buildHtml(
      connected,
      caps,
      null,
      null,
      null,
      sessionInfo,
      registration,
      serverEntry,
    );

    void this.hydrateAndRender(token);
  }

  private async hydrateAndRender(token: number): Promise<void> {
    if (!this.view || token !== this.renderToken) {
      return;
    }

    if (!this.transport.isReady()) {
      try {
        await this.transport.initialize();
      } catch (error) {
        this.logger.appendLine(`[dashboard] transport init failed: ${error}`);
      }
    }

    if (!this.view || token !== this.renderToken) {
      return;
    }

    const connected = this.transport.isReady();
    const caps = this.transport.capabilities();
    const sessionInfo = this.transport.getSessionInfo();
    const contextArgs = this.resolveContextArgs(sessionInfo);
    const registration = checkRegistration();
    const serverEntry = buildServerEntry(this.binaryInfo);

    let contextStatus: Record<string, unknown> | null = null;
    let freshnessData: Record<string, unknown> | null = null;
    let memoryStatus: Record<string, unknown> | null = null;

    if (connected) {
      [contextStatus, freshnessData, memoryStatus] = await Promise.all([
        this.callDashboardTool("context.status", contextArgs),
        this.callDashboardTool("context.freshness", contextArgs),
        this.callDashboardTool("memory.status", contextArgs),
      ]);
    }

    if (!this.view || token !== this.renderToken) {
      return;
    }

    this.view.webview.html = this.buildHtml(
      connected,
      caps,
      contextStatus,
      freshnessData,
      memoryStatus,
      sessionInfo,
      registration,
      serverEntry,
    );
  }

  private async callDashboardTool(
    name: string,
    args: Record<string, string>,
  ): Promise<Record<string, unknown> | null> {
    try {
      const result = await this.transport.callTool(name, args);
      return extractToolPayload(result);
    } catch (error) {
      this.logger.appendLine(`[dashboard] ${name} unavailable: ${error}`);
      return null;
    }
  }

  private resolveContextArgs(sessionInfo?: TransportSessionInfo): Record<string, string> {
    const contextPath = this.resolveContextPath(sessionInfo);
    return contextPath ? { context_path: contextPath } : {};
  }

  private resolveContextPath(sessionInfo?: TransportSessionInfo): string {
    const preferredContextPath = resolvePreferredContextPath() ?? "";
    if (preferredContextPath.trim()) {
      return preferredContextPath;
    }

    const sessionContextPath = sessionInfo?.contextPath?.trim();
    if (sessionContextPath) {
      return sessionContextPath;
    }

    const sessionWorkspace =
      sessionInfo?.workspace?.trim() || sessionInfo?.cliHints?.workspacePath?.trim() || "";
    if (sessionWorkspace) {
      return path.join(sessionWorkspace, ".context");
    }

    return "";
  }

  private buildHtml(
    connected: boolean,
    caps: { tools: boolean; resources: boolean; prompts: boolean },
    contextStatus: Record<string, unknown> | null,
    freshnessData: Record<string, unknown> | null,
    memoryStatus: Record<string, unknown> | null,
    sessionInfo: TransportSessionInfo | undefined,
    registration: RegistrationStatus,
    serverEntry: McpServerEntry,
  ): string {
    const statusIcon = connected ? "\u2713" : "\u2717";
    const statusClass = connected ? "connected" : "disconnected";
    const statusLabel = connected ? "Connected" : "Disconnected";

    const capsList = [
      caps.tools ? "tools" : null,
      caps.resources ? "resources" : null,
      caps.prompts ? "prompts" : null,
    ]
      .filter(Boolean)
      .join(", ") || "none";

    const notices: string[] = [];
    if (!connected) {
      notices.push("The AFS backend is disconnected. Chat and context tools may be unavailable until the server reconnects.");
    }
    if (registration.parseError) {
      notices.push(registration.parseError);
    }
    const indexState = this.getRecord(contextStatus?.index);
    if (indexState && indexState.stale === true) {
      notices.push("The context index is stale. Rebuild it before trusting retrieval-heavy answers.");
    }

    const noticeHtml = notices.length > 0
      ? `<div class="banner">${notices.map((notice) => `<div>${this.esc(notice)}</div>`).join("")}</div>`
      : "";

    const contextHtml = this.renderContextSection(contextStatus, freshnessData);
    const memoryHtml = this.renderMemorySection(memoryStatus);
    const sessionHtml = this.renderSessionSection(sessionInfo);
    const mcpHtml = this.renderMcpSection(registration, serverEntry);

    return `<!DOCTYPE html>
<html>
<head>
<style>
  body {
    font-family: var(--vscode-font-family);
    font-size: var(--vscode-font-size);
    color: var(--vscode-foreground);
    padding: 10px;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--vscode-panel-border);
  }
  .header h2 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
  }
  .status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
  }
  .status .icon { font-size: 14px; }
  .connected .icon { color: var(--vscode-testing-iconPassed); }
  .disconnected .icon { color: var(--vscode-testing-iconFailed); }
  .banner {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 10px;
    border-radius: 6px;
    border: 1px solid var(--vscode-editorWarning-border, var(--vscode-panel-border));
    background: color-mix(in srgb, var(--vscode-editorWarning-background, transparent) 60%, transparent);
    font-size: 11px;
    line-height: 1.4;
  }
  .section {
    border: 1px solid var(--vscode-panel-border);
    border-radius: 8px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .section h3 {
    margin: 0;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--vscode-descriptionForeground);
  }
  .row {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    font-size: 12px;
  }
  .label { color: var(--vscode-descriptionForeground); }
  .value {
    font-weight: 500;
    text-align: right;
  }
  .value.fresh { color: var(--vscode-testing-iconPassed); }
  .value.stale { color: var(--vscode-editorWarning-foreground); }
  .value.critical { color: var(--vscode-testing-iconFailed); }
  .sub-row {
    padding-left: 12px;
    font-size: 11px;
  }
  .path-value, code {
    word-break: break-word;
  }
  .path-value {
    max-width: 190px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 11px;
  }
  .hint-block {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .hint-label {
    font-size: 11px;
    color: var(--vscode-descriptionForeground);
  }
  code {
    display: block;
    white-space: pre-wrap;
    font-size: 11px;
    line-height: 1.45;
    padding: 6px 8px;
    border-radius: 4px;
    background: var(--vscode-textCodeBlock-background);
    color: var(--vscode-textPreformat-foreground);
  }
  ul {
    margin: 0;
    padding-left: 16px;
    font-size: 11px;
    color: var(--vscode-descriptionForeground);
  }
  .actions, .inline-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  button {
    background: var(--vscode-button-secondaryBackground);
    color: var(--vscode-button-secondaryForeground);
    border: none;
    padding: 5px 8px;
    cursor: pointer;
    font-size: 12px;
    font-family: var(--vscode-font-family);
    text-align: left;
    border-radius: 4px;
  }
  button:hover {
    background: var(--vscode-button-secondaryHoverBackground);
  }
</style>
</head>
<body>
  <div class="header">
    <h2>AFS</h2>
    <div class="status ${statusClass}">
      <span class="icon">${statusIcon}</span>
      <span>${statusLabel}</span>
    </div>
  </div>

  ${noticeHtml}

  <div class="section">
    <h3>Server</h3>
    <div class="row"><span class="label">Capabilities</span><span class="value">${this.esc(capsList)}</span></div>
    <div class="actions">
      <button onclick="post('refresh')">Refresh</button>
      <button onclick="post('openCommand', { id: 'afs.chat.open' })">Open Chat</button>
      <button onclick="post('queryIndex')">Query Index</button>
      <button onclick="post('rebuildIndex')">Rebuild Index</button>
      <button onclick="post('showLogs')">Show Logs</button>
    </div>
  </div>

  ${contextHtml}
  ${memoryHtml}
  ${mcpHtml}
  ${sessionHtml}

  <script>
    const vscode = acquireVsCodeApi();
    function post(command, data) {
      vscode.postMessage({ command, ...data });
    }
  </script>
</body>
</html>`;
  }

  private renderContextSection(
    contextStatus: Record<string, unknown> | null,
    freshnessData: Record<string, unknown> | null,
  ): string {
    if (!contextStatus) {
      return "";
    }

    const contextPath = typeof contextStatus.context_path === "string"
      ? contextStatus.context_path
      : "";
    const project = contextPath ? path.basename(path.dirname(contextPath)) : "unknown";
    const mountObj = this.getRecord(contextStatus.mount_counts) ?? {};
    const mountKeys = Object.keys(mountObj);
    const totalFiles = typeof contextStatus.total_files === "number"
      ? contextStatus.total_files
      : 0;
    const profile = typeof contextStatus.profile === "string"
      ? contextStatus.profile
      : null;
    const indexState = this.getRecord(contextStatus.index);
    const mountHealth = this.getRecord(contextStatus.mount_health);
    const indexedAt = indexState && typeof indexState.built_at === "string"
      ? indexState.built_at
      : typeof contextStatus.indexed_at === "string"
        ? contextStatus.indexed_at
        : null;
    const suggestedActions = this.collectSuggestedActions(contextStatus, mountHealth);

    const mountRows = mountKeys
      .map((key) => {
        const count = typeof mountObj[key] === "number" ? mountObj[key] : 0;
        const freshness = this.lookupFreshness(freshnessData, key);
        return `<div class="row sub-row"><span class="label">${this.esc(key)}</span><span class="value ${this.freshnessClass(freshness)}">${count}${freshness != null ? ` • ${(freshness * 100).toFixed(0)}%` : ""}</span></div>`;
      })
      .join("");

    const actionHtml = suggestedActions.length > 0
      ? `<ul>${suggestedActions.map((action) => `<li>${this.esc(action)}</li>`).join("")}</ul>`
      : "";

    return `
      <div class="section">
        <h3>Context</h3>
        <div class="row"><span class="label">Project</span><span class="value">${this.esc(project)}</span></div>
        ${profile ? `<div class="row"><span class="label">Profile</span><span class="value">${this.esc(profile)}</span></div>` : ""}
        <div class="row"><span class="label">Path</span><span class="value path-value" title="${this.esc(contextPath)}">${this.esc(this.shortPath(contextPath))}</span></div>
        <div class="row"><span class="label">Mounts</span><span class="value">${mountKeys.length}</span></div>
        ${mountRows}
        <div class="row"><span class="label">Total Files</span><span class="value">${totalFiles}</span></div>
        ${indexedAt ? `<div class="row"><span class="label">Last Indexed</span><span class="value">${this.esc(indexedAt)}</span></div>` : ""}
        ${indexState ? `<div class="row"><span class="label">Index</span><span class="value ${indexState.stale === true ? "stale" : "fresh"}">${indexState.stale === true ? "Stale" : "Ready"}</span></div>` : ""}
        ${indexState && typeof indexState.total_entries === "number" ? `<div class="row"><span class="label">Index Entries</span><span class="value">${indexState.total_entries}</span></div>` : ""}
        ${mountHealth ? `<div class="row"><span class="label">Mount Health</span><span class="value ${mountHealth.healthy === true ? "fresh" : "critical"}">${mountHealth.healthy === true ? "Healthy" : "Needs Attention"}</span></div>` : ""}
        ${actionHtml}
      </div>`;
  }

  private renderMemorySection(memoryStatus: Record<string, unknown> | null): string {
    if (!memoryStatus) {
      return "";
    }

    const entries = typeof memoryStatus.entries_count === "number"
      ? memoryStatus.entries_count
      : 0;
    const stale = memoryStatus.stale === true;
    const memPath = typeof memoryStatus.memory_path === "string"
      ? memoryStatus.memory_path
      : "";

    return `
      <div class="section">
        <h3>Memory</h3>
        <div class="row"><span class="label">Entries</span><span class="value">${entries}</span></div>
        <div class="row"><span class="label">Status</span><span class="value ${stale ? "stale" : "fresh"}">${stale ? "Stale" : "Fresh"}</span></div>
        ${memPath ? `<div class="row"><span class="label">Path</span><span class="value path-value" title="${this.esc(memPath)}">${this.esc(this.shortPath(memPath))}</span></div>` : ""}
      </div>`;
  }

  private renderMcpSection(
    registration: RegistrationStatus,
    serverEntry: McpServerEntry,
  ): string {
    const registrationLabel = registration.parseError
      ? "Config Error"
      : registration.registered
        ? "Registered"
        : "Not Registered";
    const registrationClass = registration.parseError
      ? "critical"
      : registration.registered
        ? "fresh"
        : "stale";
    const launchCommand = [serverEntry.command, ...(serverEntry.args ?? [])].join(" ");
    const envKeys = Object.keys(serverEntry.env ?? {});
    const primaryCommand = registration.registered ? "afs.mcp.register" : "afs.mcp.register";
    const primaryLabel = registration.registered ? "Update MCP Registration" : "Register MCP";

    return `
      <div class="section">
        <h3>MCP</h3>
        <div class="row"><span class="label">Status</span><span class="value ${registrationClass}">${registrationLabel}</span></div>
        <div class="row"><span class="label">Config</span><span class="value path-value" title="${this.esc(registration.configPath ?? "")}">${this.esc(this.shortPath(registration.configPath ?? "No target detected"))}</span></div>
        <div class="hint-block">
          <div class="hint-label">Launch Command</div>
          <code>${this.esc(launchCommand)}</code>
        </div>
        <div class="row"><span class="label">Env</span><span class="value">${this.esc(envKeys.length > 0 ? envKeys.join(", ") : "none")}</span></div>
        <div class="inline-actions">
          <button onclick="post('openCommand', { id: '${primaryCommand}' })">${primaryLabel}</button>
          ${registration.registered ? `<button onclick="post('openCommand', { id: 'afs.mcp.unregister' })">Unregister MCP</button>` : ""}
          <button onclick="post('mcpStatus')">View MCP Status</button>
          ${registration.configPath ? `<button onclick="post('copyText', { text: ${JSON.stringify(registration.configPath)}, label: 'MCP config path' })">Copy Config Path</button>` : ""}
        </div>
      </div>`;
  }

  private renderSessionSection(sessionInfo?: TransportSessionInfo): string {
    if (!sessionInfo) {
      return "";
    }

    const cliHints = sessionInfo.cliHints;
    const notes = cliHints.notes
      .filter((note) => note.trim())
      .map((note) => `<li>${this.esc(note)}</li>`)
      .join("");

    return `
      <div class="section">
        <h3>Session Hints</h3>
        <div class="row"><span class="label">Workspace</span><span class="value path-value" title="${this.esc(cliHints.workspacePath)}">${this.esc(this.shortPath(cliHints.workspacePath))}</span></div>
        <div class="hint-block">
          <div class="hint-label">Query</div>
          <code>${this.esc(cliHints.queryShortcut)}</code>
        </div>
        <div class="hint-block">
          <div class="hint-label">Canonical Query</div>
          <code>${this.esc(cliHints.queryCanonical)}</code>
        </div>
        <div class="hint-block">
          <div class="hint-label">Rebuild</div>
          <code>${this.esc(cliHints.indexRebuild)}</code>
        </div>
        ${notes ? `<ul>${notes}</ul>` : ""}
        <div class="inline-actions">
          <button onclick="post('queryIndex')">Query More</button>
          <button onclick="post('copyText', { text: ${JSON.stringify(cliHints.queryShortcut)}, label: 'query command' })">Copy Query Command</button>
          <button onclick="post('copyText', { text: ${JSON.stringify(cliHints.indexRebuild)}, label: 'index rebuild command' })">Copy Rebuild Command</button>
        </div>
      </div>`;
  }

  private collectSuggestedActions(
    contextStatus: Record<string, unknown>,
    mountHealth: Record<string, unknown> | null,
  ): string[] {
    const statusActions = this.getStringArray(contextStatus.actions);
    const mountActions = this.getStringArray(mountHealth?.suggested_actions);
    return Array.from(new Set([...statusActions, ...mountActions])).slice(0, 4);
  }

  private lookupFreshness(
    freshnessData: Record<string, unknown> | null,
    mount: string,
  ): number | null {
    const scores = this.getRecord(freshnessData?.mount_scores);
    const score = scores?.[mount];
    return typeof score === "number" ? score : null;
  }

  private freshnessClass(score: number | null): string {
    if (score == null) {
      return "";
    }
    if (score >= 0.5) {
      return "fresh";
    }
    if (score >= 0.3) {
      return "stale";
    }
    return "critical";
  }

  private getRecord(value: unknown): Record<string, unknown> | null {
    return value && typeof value === "object" && !Array.isArray(value)
      ? value as Record<string, unknown>
      : null;
  }

  private getStringArray(value: unknown): string[] {
    return Array.isArray(value)
      ? value.filter((entry): entry is string => typeof entry === "string" && entry.trim().length > 0)
      : [];
  }

  private shortPath(filePath: string): string {
    if (!filePath) {
      return "";
    }
    const normalized = filePath.replace(/\\/g, "/");
    const home = (process.env.HOME ?? process.env.USERPROFILE ?? "").replace(/\\/g, "/");
    if (home && normalized.startsWith(home)) {
      return `~${normalized.slice(home.length)}`;
    }
    return normalized.replace(/^.*\//, ".../");
  }

  private esc(value: string): string {
    return value
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }
}
