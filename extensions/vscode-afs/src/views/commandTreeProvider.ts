import * as vscode from "vscode";

class CommandGroupItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly icon: string,
    public readonly children: CommandItem[],
  ) {
    super(label, vscode.TreeItemCollapsibleState.Collapsed);
    this.iconPath = new vscode.ThemeIcon(icon);
    this.contextValue = "commandGroup";
  }
}

class CommandItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly commandId: string,
    public readonly icon: string,
  ) {
    super(label, vscode.TreeItemCollapsibleState.None);
    this.iconPath = new vscode.ThemeIcon(icon);
    this.contextValue = "commandItem";
    this.command = {
      command: commandId,
      title: label,
    };
  }
}

type TreeNode = CommandGroupItem | CommandItem;

export class AfsCommandTreeProvider implements vscode.TreeDataProvider<TreeNode> {
  private _onDidChangeTreeData = new vscode.EventEmitter<TreeNode | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private groups: CommandGroupItem[];

  constructor() {
    this.groups = this.buildGroups();
  }

  refresh(): void {
    this.groups = this.buildGroups();
    this._onDidChangeTreeData.fire(undefined);
  }

  getTreeItem(element: TreeNode): vscode.TreeItem {
    return element;
  }

  getChildren(element?: TreeNode): TreeNode[] {
    if (!element) {
      return this.groups;
    }
    if (element instanceof CommandGroupItem) {
      return element.children;
    }
    return [];
  }

  private buildGroups(): CommandGroupItem[] {
    return [
      new CommandGroupItem("Context", "folder-library", [
        new CommandItem("Discover Contexts", "afs.context.discover", "search"),
        new CommandItem("Initialize Context", "afs.context.init", "new-folder"),
        new CommandItem("Mount Source", "afs.context.mount", "plug"),
        new CommandItem("Unmount Source", "afs.context.unmount", "debug-disconnect"),
      ]),
      new CommandGroupItem("Index", "database", [
        new CommandItem("Rebuild Index", "afs.index.rebuild", "refresh"),
        new CommandItem("Query Index", "afs.index.query", "search"),
        new CommandItem("Quick Open", "afs.index.queryQuickOpen", "go-to-file"),
      ]),
      new CommandGroupItem("Chat", "comment-discussion", [
        new CommandItem("Open Chat", "afs.chat.open", "comment-discussion"),
      ]),
      new CommandGroupItem("Server", "server", [
        new CommandItem("MCP Status", "afs.mcp.status", "info"),
        new CommandItem("Register MCP", "afs.mcp.register", "plug"),
        new CommandItem("Unregister MCP", "afs.mcp.unregister", "debug-disconnect"),
        new CommandItem("Restart Server", "afs.server.restart", "debug-restart"),
        new CommandItem("Show Logs", "afs.server.showLogs", "output"),
      ]),
      new CommandGroupItem("Tools", "tools", [
        new CommandItem("Refresh Explorer", "afs.treeView.refresh", "refresh"),
        new CommandItem("Refresh Dashboard", "afs.dashboard.refresh", "refresh"),
      ]),
    ];
  }
}
