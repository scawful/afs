import * as os from "os";
import * as path from "path";
import * as vscode from "vscode";
import type { ContextService } from "../services/contextService";
import type { FileService } from "../services/fileService";
import type { DiscoveredContext, MountType } from "../types";
import { DEFAULT_POLICIES, PolicyType } from "../types";
import {
  ContextTreeItem,
  ContextFileItem,
  ContextRootItem,
  MountTypeItem,
} from "./contextTreeItems";

const MOUNT_TYPE_ORDER: MountType[] = [
  "memory",
  "knowledge",
  "tools",
  "scratchpad",
  "history",
  "hivemind",
  "global",
  "items",
  "monorepo",
] as MountType[];

export class ContextTreeProvider implements vscode.TreeDataProvider<ContextTreeItem> {
  private _onDidChangeTreeData = new vscode.EventEmitter<ContextTreeItem | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private contexts: DiscoveredContext[] = [];
  private freshnessCache = new Map<string, Record<string, number>>();

  constructor(
    private readonly contextService: ContextService,
    private readonly fileService: FileService,
    private readonly showEmptyMounts: boolean,
  ) {}

  refresh(): void {
    this._onDidChangeTreeData.fire(undefined);
  }

  getTreeItem(element: ContextTreeItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: ContextTreeItem): Promise<ContextTreeItem[]> {
    if (!element) {
      return this.getRootItems();
    }
    if (element instanceof ContextRootItem) {
      return this.getMountTypes(element.contextPath);
    }
    if (element instanceof MountTypeItem) {
      return this.getFiles(element.contextPath, element.mountType);
    }
    if (element instanceof ContextFileItem && element.isDir) {
      return this.getDirectoryChildren(element.filePath);
    }
    return [];
  }

  private async getRootItems(): Promise<ContextTreeItem[]> {
    try {
      this.contexts = await this.contextService.discover();
    } catch {
      this.contexts = [];
    }

    // Pre-fetch freshness for all discovered contexts
    this.freshnessCache.clear();
    for (const ctx of this.contexts) {
      try {
        const freshness = await this.contextService.freshness(ctx.path);
        if (freshness?.mount_scores) {
          this.freshnessCache.set(ctx.path, freshness.mount_scores);
        }
      } catch {
        // freshness not available for this context
      }
    }

    const items: ContextTreeItem[] = [];

    // Add global ~/.context if it exists
    const globalContext = path.join(os.homedir(), ".context");
    const alreadyIncluded = this.contexts.some(
      (ctx) => ctx.path === globalContext || ctx.path === path.resolve(globalContext),
    );
    if (!alreadyIncluded) {
      items.push(new ContextRootItem("~/.context", globalContext, true, 0));
    }

    // Add discovered contexts
    for (const ctx of this.contexts) {
      items.push(new ContextRootItem(ctx.project, ctx.path, ctx.valid, ctx.mounts));
    }

    return items;
  }

  private async getMountTypes(contextPath: string): Promise<MountTypeItem[]> {
    const items: MountTypeItem[] = [];
    const scores = this.freshnessCache.get(contextPath);
    for (const mt of MOUNT_TYPE_ORDER) {
      try {
        const entries = await this.fileService.list(`${contextPath}/${mt}`, 1);
        if (entries.length > 0 || this.showEmptyMounts) {
          const policy = DEFAULT_POLICIES[mt] ?? PolicyType.READ_ONLY;
          const freshness = scores?.[mt];
          items.push(new MountTypeItem(mt, contextPath, policy, entries.length, freshness));
        }
      } catch {
        if (this.showEmptyMounts) {
          const policy = DEFAULT_POLICIES[mt] ?? PolicyType.READ_ONLY;
          const freshness = scores?.[mt];
          items.push(new MountTypeItem(mt, contextPath, policy, 0, freshness));
        }
      }
    }
    return items;
  }

  private async getFiles(
    contextPath: string,
    mountType: string,
  ): Promise<ContextFileItem[]> {
    try {
      const entries = await this.fileService.list(`${contextPath}/${mountType}`, 1);
      return entries.map((e) => new ContextFileItem(e.path, e.is_dir));
    } catch {
      return [];
    }
  }

  private async getDirectoryChildren(
    dirPath: string,
  ): Promise<ContextFileItem[]> {
    try {
      const entries = await this.fileService.list(dirPath, 1);
      // Filter out the directory itself (rglob can return the root)
      return entries
        .filter((e) => e.path !== dirPath)
        .map((e) => new ContextFileItem(e.path, e.is_dir));
    } catch {
      return [];
    }
  }
}
