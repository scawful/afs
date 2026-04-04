import * as vscode from "vscode";
import type { ConnectionState } from "../types";
import type { TransportSessionInfo } from "../transport/types";

export class AfsStatusBar implements vscode.Disposable {
  private readonly item: vscode.StatusBarItem;

  constructor() {
    this.item = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 50);
    this.item.command = "afs.mcp.status";
    this.update("disconnected");
  }

  update(state: ConnectionState, contextName?: string, sessionInfo?: TransportSessionInfo): void {
    const sessionLabel =
      contextName?.trim() ||
      this.labelFromWorkspace(sessionInfo?.workspace ?? "");
    switch (state) {
      case "connected":
        this.item.text = `$(check) AFS${sessionLabel ? `: ${sessionLabel}` : ""}`;
        this.item.tooltip = this.connectedTooltip(sessionInfo);
        break;
      case "disconnected":
        this.item.text = "$(circle-slash) AFS";
        this.item.tooltip = `AFS disconnected${sessionLabel ? ` (${sessionLabel})` : ""}`;
        break;
      case "error":
        this.item.text = "$(error) AFS";
        this.item.tooltip = `AFS error${sessionLabel ? ` (${sessionLabel})` : ""} — click for details`;
        break;
    }
    this.item.show();
  }

  private connectedTooltip(sessionInfo?: TransportSessionInfo): string {
    const lines = ["AFS connected — click for MCP status"];
    if (!sessionInfo) {
      return lines.join("\n");
    }
    const label = this.labelFromWorkspace(sessionInfo.workspace);
    if (label) {
      lines.push(`Workspace: ${label}`);
    }
    if (sessionInfo.cliHints.queryShortcut) {
      lines.push(`Query: ${sessionInfo.cliHints.queryShortcut}`);
    }
    if (sessionInfo.cliHints.indexRebuild) {
      lines.push(`Rebuild: ${sessionInfo.cliHints.indexRebuild}`);
    }
    for (const note of sessionInfo.cliHints.notes) {
      if (note.trim()) {
        lines.push(`Note: ${note.trim()}`);
      }
    }
    return lines.join("\n");
  }

  private labelFromWorkspace(workspace: string): string {
    const trimmed = workspace.trim().replace(/[\\/]+$/, "");
    if (!trimmed) {
      return "";
    }
    const segments = trimmed.split(/[\\/]/).filter(Boolean);
    return segments[segments.length - 1] ?? trimmed;
  }

  dispose(): void {
    this.item.dispose();
  }
}
