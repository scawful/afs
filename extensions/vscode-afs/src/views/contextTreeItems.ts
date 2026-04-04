import * as vscode from "vscode";
import type { PolicyType } from "../types";

export abstract class ContextTreeItem extends vscode.TreeItem {}

export class ContextRootItem extends ContextTreeItem {
  constructor(
    public readonly projectName: string,
    public readonly contextPath: string,
    public readonly isValid: boolean,
    public readonly mountCount: number,
  ) {
    super(projectName, vscode.TreeItemCollapsibleState.Collapsed);
    this.description = contextPath;
    this.tooltip = `${projectName} (${mountCount} mounts)${isValid ? "" : " [invalid]"}`;
    this.contextValue = "contextRoot";
    this.iconPath = new vscode.ThemeIcon(isValid ? "folder-library" : "warning");
  }
}

export class MountTypeItem extends ContextTreeItem {
  constructor(
    public readonly mountType: string,
    public readonly contextPath: string,
    public readonly policy: PolicyType,
    public readonly entryCount: number,
    public readonly freshness?: number,
  ) {
    super(
      mountType,
      entryCount > 0
        ? vscode.TreeItemCollapsibleState.Collapsed
        : vscode.TreeItemCollapsibleState.None,
    );
    const freshnessLabel = freshness != null
      ? ` ${freshnessIndicator(freshness)}`
      : "";
    this.description = `${policy} (${entryCount})${freshnessLabel}`;
    this.tooltip = freshness != null
      ? `${mountType} — ${policy} — ${entryCount} entries — freshness: ${(freshness * 100).toFixed(0)}%`
      : `${mountType} — ${policy} — ${entryCount} entries`;
    this.contextValue = "mountType";
    this.iconPath = new vscode.ThemeIcon(policyIcon(policy));
  }
}

export class ContextFileItem extends ContextTreeItem {
  constructor(
    public readonly filePath: string,
    public readonly isDir: boolean,
  ) {
    const name = filePath.split("/").pop() ?? filePath;
    super(
      name,
      isDir
        ? vscode.TreeItemCollapsibleState.Collapsed
        : vscode.TreeItemCollapsibleState.None,
    );
    this.contextValue = isDir ? "contextDir" : "contextFile";
    this.resourceUri = vscode.Uri.file(filePath);
    if (!isDir) {
      this.command = {
        command: "vscode.open",
        title: "Open File",
        arguments: [vscode.Uri.file(filePath)],
      };
    }
    this.iconPath = isDir ? vscode.ThemeIcon.Folder : vscode.ThemeIcon.File;
  }
}

function policyIcon(policy: PolicyType): string {
  switch (policy) {
    case "read_only":
      return "lock";
    case "writable":
      return "edit";
    case "executable":
      return "terminal";
    default:
      return "folder";
  }
}

function freshnessIndicator(score: number): string {
  const pct = (score * 100).toFixed(0);
  if (score >= 0.5) return `[${pct}%]`;
  if (score >= 0.3) return `[${pct}% !]`;
  return `[${pct}% !!]`;
}
