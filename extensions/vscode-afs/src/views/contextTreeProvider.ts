import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import * as vscode from "vscode";
import type { ContextService } from "../services/contextService";
import type { FileService } from "../services/fileService";
import type { DiscoveredContext, MountType } from "../types";
import { DEFAULT_POLICIES, PolicyType } from "../types";
import {
  ContextFileItem,
  ContextRootItem,
  ContextTreeItem,
  MountPointItem,
  MountTypeItem,
} from "./contextTreeItems";
import { getConfig } from "../utils/config";

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

  private manualDiscoveredContexts: DiscoveredContext[] = [];
  private autoDiscoveredContexts: DiscoveredContext[] = [];
  private autoDiscoverPromise: Promise<void> | undefined;
  private autoDiscoverLoaded = false;
  private freshnessCache = new Map<string, Record<string, number>>();
  private freshnessRequests = new Map<string, Promise<Record<string, number>>>();
  private rootStatusCache = new Map<string, { valid: boolean; mounts: number }>();
  private rootStatusRequests = new Map<string, Promise<boolean>>();
  private pathExistsCache = new Map<string, Promise<boolean>>();

  constructor(
    private readonly contextService: ContextService,
    private readonly fileService: FileService,
  ) {}

  refresh(): void {
    this.autoDiscoverPromise = undefined;
    this.autoDiscoveredContexts = [];
    this.autoDiscoverLoaded = false;
    this.freshnessCache.clear();
    this.freshnessRequests.clear();
    this.rootStatusCache.clear();
    this.rootStatusRequests.clear();
    this.pathExistsCache.clear();
    this._onDidChangeTreeData.fire(undefined);
  }

  setManualDiscoveredContexts(contexts: DiscoveredContext[]): void {
    this.manualDiscoveredContexts = dedupeContexts(contexts);
    this.refresh();
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
      return this.getMountPoints(element.contextPath, element.mountType);
    }
    if (element instanceof MountPointItem && element.isDir) {
      return this.getDirectoryChildren(element.filePath);
    }
    if (element instanceof ContextFileItem && element.isDir) {
      return this.getDirectoryChildren(element.filePath);
    }
    return [];
  }

  private async getRootItems(): Promise<ContextTreeItem[]> {
    const autoDiscover = getConfig<boolean>("discovery.autoDiscover", true);
    const contexts = autoDiscover
      ? await this.visibleContextsWithBackgroundDiscovery()
      : await this.visibleContextsWithoutDiscovery();

    return contexts.map(
      (ctx) => {
        const status = this.rootStatusCache.get(ctx.path);
        return new ContextRootItem(
          ctx.project,
          ctx.path,
          status?.valid ?? ctx.valid,
          status?.mounts ?? ctx.mounts,
        );
      },
    );
  }

  private async getMountTypes(contextPath: string): Promise<MountTypeItem[]> {
    void this.ensureRootStatus(contextPath).then((changed) => {
      if (changed) {
        this._onDidChangeTreeData.fire(undefined);
      }
    });

    const items: MountTypeItem[] = [];
    const scores = await this.getFreshnessScores(contextPath);
    const showEmptyMounts = getConfig<boolean>("treeView.showEmptyMounts", false);

    for (const mt of MOUNT_TYPE_ORDER) {
      const mountPath = path.join(contextPath, mt);
      if (!(await this.pathExists(mountPath))) {
        if (showEmptyMounts) {
          const policy = DEFAULT_POLICIES[mt] ?? PolicyType.READ_ONLY;
          const freshness = scores?.[mt];
          items.push(new MountTypeItem(mt, contextPath, policy, 0, freshness));
        }
        continue;
      }

      try {
        const entries = await this.fileService.list(mountPath, 1);
        const mountPoints = entries.filter((entry) => entry.path !== mountPath);
        if (mountPoints.length > 0 || showEmptyMounts) {
          const policy = DEFAULT_POLICIES[mt] ?? PolicyType.READ_ONLY;
          const freshness = scores?.[mt];
          items.push(
            new MountTypeItem(mt, contextPath, policy, mountPoints.length, freshness),
          );
        }
      } catch {
        if (showEmptyMounts) {
          const policy = DEFAULT_POLICIES[mt] ?? PolicyType.READ_ONLY;
          const freshness = scores?.[mt];
          items.push(new MountTypeItem(mt, contextPath, policy, 0, freshness));
        }
      }
    }

    return items;
  }

  private async getMountPoints(
    contextPath: string,
    mountType: string,
  ): Promise<MountPointItem[]> {
    const mountPath = path.join(contextPath, mountType);
    if (!(await this.pathExists(mountPath))) {
      return [];
    }

    try {
      const entries = await this.fileService.list(mountPath, 1);
      return entries
        .filter((entry) => entry.path !== mountPath)
        .map(
          (entry) =>
            new MountPointItem(
              path.basename(entry.path) || entry.path,
              entry.path,
              entry.is_dir,
              mountType,
              contextPath,
            ),
        );
    } catch {
      return [];
    }
  }

  private async getDirectoryChildren(dirPath: string): Promise<ContextFileItem[]> {
    try {
      const entries = await this.fileService.list(dirPath, 1);
      return entries
        .filter((entry) => entry.path !== dirPath)
        .map((entry) => new ContextFileItem(entry.path, entry.is_dir));
    } catch {
      return [];
    }
  }

  private async visibleContextsWithBackgroundDiscovery(): Promise<DiscoveredContext[]> {
    const visibleContexts = await this.localVisibleContexts();
    this.kickoffAutoDiscover();
    return dedupeContexts([...visibleContexts, ...this.autoDiscoveredContexts]);
  }

  private kickoffAutoDiscover(): void {
    if (this.autoDiscoverPromise || this.autoDiscoverLoaded) {
      return;
    }

    const searchPaths = getConfig<string[]>("discovery.searchPaths", [])
      .map((value) => value.trim())
      .filter(Boolean);
    const maxDepth = Math.max(1, getConfig<number>("discovery.maxDepth", 3));

    this.autoDiscoverPromise = (async () => {
      try {
        const discovered = await this.contextService.discover(
          searchPaths.length > 0 ? searchPaths : undefined,
          maxDepth,
        );
        this.autoDiscoveredContexts = dedupeContexts(discovered);
        this.autoDiscoverLoaded = true;
        this._onDidChangeTreeData.fire(undefined);
      } catch {
        this.autoDiscoveredContexts = [];
        this.autoDiscoverLoaded = true;
      } finally {
        this.autoDiscoverPromise = undefined;
      }
    })();
  }

  private async visibleContextsWithoutDiscovery(): Promise<DiscoveredContext[]> {
    const visibleContexts = await this.localVisibleContexts();
    return dedupeContexts([...visibleContexts, ...this.manualDiscoveredContexts]);
  }

  private async localVisibleContexts(): Promise<DiscoveredContext[]> {
    const candidates: Array<{ project: string; contextPath: string }> = [];

    for (const folder of vscode.workspace.workspaceFolders ?? []) {
      const contextPath = path.join(folder.uri.fsPath, ".context");
      if (fs.existsSync(contextPath)) {
        candidates.push({ project: folder.name, contextPath });
      }
    }

    const globalContext = path.join(os.homedir(), ".context");
    if (fs.existsSync(globalContext)) {
      candidates.push({ project: "~/.context", contextPath: globalContext });
    }

    return dedupeContextCandidates(candidates).map(({ project, contextPath }) => ({
      project,
      path: contextPath,
      valid: true,
      mounts: 0,
    }));
  }

  private ensureRootStatus(contextPath: string): Promise<boolean> {
    if (this.rootStatusCache.has(contextPath)) {
      return Promise.resolve(false);
    }
    const pending = this.rootStatusRequests.get(contextPath);
    if (pending) {
      return pending;
    }

    const request = (async () => {
      try {
        const status = await this.contextService.status(contextPath);
        const mountCounts = status?.mount_counts ?? {};
        const mounts = Object.values(mountCounts).reduce(
          (sum, value) => sum + (typeof value === "number" ? value : 0),
          0,
        );
        const next = {
          valid: status !== null,
          mounts,
        };
        const prev = this.rootStatusCache.get(contextPath);
        const changed = !prev || prev.valid !== next.valid || prev.mounts !== next.mounts;
        this.rootStatusCache.set(contextPath, next);
        return changed;
      } catch {
        const prev = this.rootStatusCache.get(contextPath);
        const changed = !prev || prev.valid !== false || prev.mounts !== 0;
        this.rootStatusCache.set(contextPath, { valid: false, mounts: 0 });
        return changed;
      } finally {
        this.rootStatusRequests.delete(contextPath);
      }
    })();

    this.rootStatusRequests.set(contextPath, request);
    return request;
  }

  private async pathExists(filePath: string): Promise<boolean> {
    const cached = this.pathExistsCache.get(filePath);
    if (cached) {
      return cached;
    }

    const check = fs.promises
      .access(filePath, fs.constants.F_OK)
      .then(() => true)
      .catch(() => false);
    this.pathExistsCache.set(filePath, check);
    return check;
  }

  private async getFreshnessScores(contextPath: string): Promise<Record<string, number> | undefined> {
    const cached = this.freshnessCache.get(contextPath);
    if (cached) {
      return cached;
    }

    const pending = this.freshnessRequests.get(contextPath);
    if (pending) {
      return pending;
    }

    const request = (async () => {
      try {
        const freshness = await this.contextService.freshness(contextPath);
        const scores = freshness?.mount_scores ?? {};
        this.freshnessCache.set(contextPath, scores);
        return scores;
      } catch {
        return {};
      } finally {
        this.freshnessRequests.delete(contextPath);
      }
    })();

    this.freshnessRequests.set(contextPath, request);
    return request;
  }
}

function dedupeContexts(contexts: DiscoveredContext[]): DiscoveredContext[] {
  const byPath = new Map<string, DiscoveredContext>();
  for (const context of contexts) {
    byPath.set(path.resolve(context.path), context);
  }
  return Array.from(byPath.values());
}

function dedupeContextCandidates(
  candidates: Array<{ project: string; contextPath: string }>,
): Array<{ project: string; contextPath: string }> {
  const byPath = new Map<string, { project: string; contextPath: string }>();
  for (const candidate of candidates) {
    byPath.set(path.resolve(candidate.contextPath), candidate);
  }
  return Array.from(byPath.values());
}
