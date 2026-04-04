import * as vscode from "vscode";
import * as path from "path";
import type { ITransportClient, TransportSessionInfo } from "../transport/types";
import { extractToolPayload } from "../utils/toolPayload";

export class AfsDashboardProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "afs.dashboard";

  private view?: vscode.WebviewView;

  constructor(
    private readonly transport: ITransportClient,
    private readonly logger: vscode.OutputChannel,
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
          if (msg.id) {
            await vscode.commands.executeCommand(msg.id);
          }
          break;
      }
    });
    this.updateContent();
  }

  async refresh(): Promise<void> {
    await this.updateContent();
  }

  private async updateContent(): Promise<void> {
    if (!this.view) return;

    const connected = this.transport.isReady();
    const caps = this.transport.capabilities();

    // Try to get context status via MCP
    let contextStatus: Record<string, unknown> | null = null;
    let freshnessData: Record<string, unknown> | null = null;
    let antigravityStatus: Record<string, unknown> | null = null;
    let memoryStatus: Record<string, unknown> | null = null;
    const sessionInfo = this.transport.getSessionInfo();

    if (connected) {
      try {
        const statusResult = await this.transport.callTool("context.status", {});
        const parsed = extractToolPayload(statusResult);
        if (parsed) {
          contextStatus = parsed;
        }
      } catch {
        // status tool may not exist
      }

      try {
        const freshResult = await this.transport.callTool("context.freshness", {});
        const parsed = extractToolPayload(freshResult);
        if (parsed) {
          freshnessData = parsed;
        }
      } catch {
        // freshness tool may not exist
      }

      try {
        const agResult = await this.transport.callTool("training.antigravity.status", {});
        const parsed = extractToolPayload(agResult);
        if (parsed) {
          antigravityStatus = parsed;
        }
      } catch {
        // antigravity tool may not exist
      }

      try {
        const memResult = await this.transport.callTool("memory.status", {});
        const parsed = extractToolPayload(memResult);
        if (parsed) {
          memoryStatus = parsed;
        }
      } catch {
        // memory tool may not exist
      }
    }

    this.view.webview.html = this.buildHtml(
      connected,
      caps,
      contextStatus,
      freshnessData,
      antigravityStatus,
      memoryStatus,
      sessionInfo,
    );
  }

  private buildHtml(
    connected: boolean,
    caps: { tools: boolean; resources: boolean; prompts: boolean },
    contextStatus: Record<string, unknown> | null,
    freshnessData: Record<string, unknown> | null,
    antigravityStatus: Record<string, unknown> | null,
    memoryStatus: Record<string, unknown> | null,
    sessionInfo?: TransportSessionInfo,
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

    let contextHtml = "";
    if (contextStatus) {
      const contextPath = typeof contextStatus.context_path === "string"
        ? contextStatus.context_path
        : "";
      const project = contextPath ? path.basename(path.dirname(contextPath)) : "unknown";
      const mountCounts = contextStatus.mount_counts;
      const mountObj = mountCounts && typeof mountCounts === "object" && !Array.isArray(mountCounts)
        ? mountCounts as Record<string, unknown>
        : {};
      const mountKeys = Object.keys(mountObj);
      const totalFiles = typeof contextStatus.total_files === "number"
        ? contextStatus.total_files
        : 0;
      const indexedAt = typeof contextStatus.indexed_at === "string"
        ? contextStatus.indexed_at
        : null;

      let mountDetails = "";
      if (mountKeys.length > 0) {
        const mountRows = mountKeys
          .map((k) => {
            const count = typeof mountObj[k] === "number" ? mountObj[k] : 0;
            return `<div class="row sub-row"><span class="label">${this.esc(k)}</span><span class="value">${count}</span></div>`;
          })
          .join("");
        mountDetails = mountRows;
      }

      contextHtml = `
        <div class="section">
          <h3>Context</h3>
          <div class="row"><span class="label">Project</span><span class="value">${this.esc(project)}</span></div>
          <div class="row"><span class="label">Path</span><span class="value path-value" title="${this.esc(contextPath)}">${this.esc(contextPath.replace(/^.*\//, ".../" ))}</span></div>
          <div class="row"><span class="label">Mounts</span><span class="value">${mountKeys.length}</span></div>
          ${mountDetails}
          <div class="row"><span class="label">Total Files</span><span class="value">${totalFiles}</span></div>
          ${indexedAt ? `<div class="row"><span class="label">Last Indexed</span><span class="value">${this.esc(indexedAt)}</span></div>` : ""}
        </div>`;
    }

    let freshnessHtml = "";
    if (freshnessData) {
      const scores = (freshnessData as any).mount_scores ?? {};
      const rows = Object.entries(scores)
        .map(([mount, score]) => {
          const s = typeof score === "number" ? score : 0;
          const cls = s >= 0.5 ? "fresh" : s >= 0.3 ? "stale" : "critical";
          return `<div class="row"><span class="label">${this.esc(mount)}</span><span class="value ${cls}">${(s * 100).toFixed(0)}%</span></div>`;
        })
        .join("");
      if (rows) {
        freshnessHtml = `
          <div class="section">
            <h3>Freshness</h3>
            ${rows}
          </div>`;
      }
    }

    let antigravityHtml = "";
    if (antigravityStatus) {
      const count = (antigravityStatus as any).payload_count ?? 0;
      const dbExists = (antigravityStatus as any).db_exists === true;
      const lastSync = (antigravityStatus as any).last_sync ?? "unknown";
      antigravityHtml = `
        <div class="section">
          <h3>Antigravity</h3>
          <div class="row"><span class="label">Payloads</span><span class="value">${count}</span></div>
          <div class="row"><span class="label">Database</span><span class="value">${dbExists ? "Found" : "Missing"}</span></div>
          <div class="row"><span class="label">Last Sync</span><span class="value">${this.esc(String(lastSync))}</span></div>
        </div>`;
    }

    let memoryHtml = "";
    if (memoryStatus) {
      const entries = typeof (memoryStatus as any).entries_count === "number"
        ? (memoryStatus as any).entries_count
        : 0;
      const stale = (memoryStatus as any).stale === true;
      const cls = stale ? "stale" : "fresh";
      const memPath = typeof (memoryStatus as any).memory_path === "string"
        ? (memoryStatus as any).memory_path
        : "";
      memoryHtml = `
        <div class="section">
          <h3>Memory</h3>
          <div class="row"><span class="label">Entries</span><span class="value">${entries}</span></div>
          <div class="row"><span class="label">Status</span><span class="value ${cls}">${stale ? "Stale" : "Fresh"}</span></div>
          ${memPath ? `<div class="row"><span class="label">Path</span><span class="value path-value" title="${this.esc(memPath)}">${this.esc(memPath.replace(/^.*\//, ".../" ))}</span></div>` : ""}
        </div>`;
    }

    let sessionHtml = "";
    if (sessionInfo) {
      const cliHints = sessionInfo.cliHints;
      const notes = cliHints.notes
        .map((note) => `<li>${this.esc(note)}</li>`)
        .join("");
      sessionHtml = `
        <div class="section">
          <h3>Session Hints</h3>
          <div class="row"><span class="label">Workspace</span><span class="value path-value" title="${this.esc(cliHints.workspacePath)}">${this.esc(cliHints.workspacePath.replace(/^.*\//, ".../"))}</span></div>
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
          ${
            notes
              ? `<ul class="notes">${notes}</ul>`
              : ""
          }
          <div class="actions inline-actions">
            <button onclick="post('queryIndex')">Query More</button>
            <button onclick="post('copyText', { text: ${JSON.stringify(cliHints.queryShortcut)}, label: 'query command' })">Copy Query Command</button>
            <button onclick="post('copyText', { text: ${JSON.stringify(cliHints.indexRebuild)}, label: 'index rebuild command' })">Copy Rebuild Command</button>
          </div>
        </div>`;
    }

    return `<!DOCTYPE html>
<html>
<head>
<style>
  body {
    font-family: var(--vscode-font-family);
    font-size: var(--vscode-font-size);
    color: var(--vscode-foreground);
    padding: 8px;
    margin: 0;
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--vscode-panel-border);
  }
  .header h2 { margin: 0; font-size: 13px; font-weight: 600; }
  .status { display: flex; align-items: center; gap: 6px; font-size: 12px; }
  .status .icon { font-size: 14px; }
  .connected .icon { color: var(--vscode-testing-iconPassed); }
  .disconnected .icon { color: var(--vscode-testing-iconFailed); }
  .section { margin-bottom: 12px; }
  .section h3 {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--vscode-descriptionForeground);
    margin: 0 0 6px 0;
  }
  .row {
    display: flex;
    justify-content: space-between;
    padding: 2px 0;
    font-size: 12px;
  }
  .label { color: var(--vscode-descriptionForeground); }
  .value { font-weight: 500; }
  .value.fresh { color: var(--vscode-testing-iconPassed); }
  .value.stale { color: var(--vscode-editorWarning-foreground); }
  .value.critical { color: var(--vscode-testing-iconFailed); }
  .sub-row { padding-left: 12px; font-size: 11px; }
  .path-value {
    max-width: 140px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 11px;
  }
  .actions {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 12px;
  }
  .inline-actions {
    margin-top: 8px;
  }
  .hint-block {
    margin: 6px 0;
  }
  .hint-label {
    font-size: 11px;
    color: var(--vscode-descriptionForeground);
    margin-bottom: 2px;
  }
  code {
    display: block;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 11px;
    line-height: 1.4;
    padding: 6px 8px;
    border-radius: 4px;
    background: var(--vscode-textCodeBlock-background);
    color: var(--vscode-textPreformat-foreground);
  }
  .notes {
    margin: 8px 0 0 16px;
    padding: 0;
    color: var(--vscode-descriptionForeground);
    font-size: 11px;
  }
  .notes li {
    margin: 4px 0;
  }
  button {
    background: var(--vscode-button-secondaryBackground);
    color: var(--vscode-button-secondaryForeground);
    border: none;
    padding: 4px 8px;
    cursor: pointer;
    font-size: 12px;
    font-family: var(--vscode-font-family);
    text-align: left;
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

  <div class="section">
    <h3>Server</h3>
    <div class="row"><span class="label">Capabilities</span><span class="value">${capsList}</span></div>
  </div>

  ${contextHtml}
  ${freshnessHtml}
  ${memoryHtml}
  ${antigravityHtml}
  ${sessionHtml}

  <div class="actions">
    <button onclick="post('refresh')">Refresh</button>
    <button onclick="post('rebuildIndex')">Rebuild Index</button>
    <button onclick="post('showLogs')">Show Logs</button>
  </div>

  <script>
    const vscode = acquireVsCodeApi();
    function post(cmd, data) {
      vscode.postMessage({ command: cmd, ...data });
    }
  </script>
</body>
</html>`;
  }

  private esc(s: string): string {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
}
