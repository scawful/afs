import * as assert from "node:assert";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, it } from "node:test";
import { __resetTestState, __setConfiguration, workspace } from "vscode";
import { ContextService } from "../../src/services/contextService";
import { FileService } from "../../src/services/fileService";
import { ContextTreeProvider } from "../../src/views/contextTreeProvider";
import { MockTransport } from "./mockTransport";

const tempRoots: string[] = [];

describe("ContextTreeProvider", () => {
  beforeEach(() => {
    __resetTestState();
  });

  afterEach(() => {
    for (const tempRoot of tempRoots.splice(0)) {
      fs.rmSync(tempRoot, { recursive: true, force: true });
    }
  });

  it("skips broad discovery when auto-discover is disabled", async () => {
    const workspaceRoot = makeWorkspace();
    const contextPath = path.join(workspaceRoot, ".context");
    fs.mkdirSync(contextPath, { recursive: true });

    const transport = new MockTransport();
    transport.toolHandlers["context.status"] = async (args) => ({
      context_path: String(args.context_path ?? contextPath),
      mount_counts: { knowledge: 1 },
      total_files: 1,
    });
    transport.toolHandlers["context.freshness"] = async () => ({
      mount_scores: { knowledge: 0.9 },
    });

    workspace.workspaceFolders = [{ name: "demo", uri: { fsPath: workspaceRoot } }];
    __setConfiguration("afs.discovery.autoDiscover", false);

    const provider = new ContextTreeProvider(
      new ContextService(transport),
      new FileService(transport),
    );
    const roots = await provider.getChildren();

    assert.ok(roots.some((root) => (root as { contextPath?: string }).contextPath === contextPath));
    assert.strictEqual(
      transport.toolCalls.some((call) => call.name === "context.discover"),
      false,
    );
  });

  it("returns local contexts before broad auto-discovery completes", async () => {
    const workspaceRoot = makeWorkspace();
    const contextPath = path.join(workspaceRoot, ".context");
    fs.mkdirSync(contextPath, { recursive: true });

    let resolveDiscover: ((value: Record<string, unknown>) => void) | undefined;
    const transport = new MockTransport();
    transport.toolHandlers["context.status"] = async (args) => ({
      context_path: String(args.context_path ?? contextPath),
      mount_counts: { knowledge: 1 },
      total_files: 1,
    });
    transport.toolHandlers["context.discover"] = async () =>
      await new Promise<Record<string, unknown>>((resolve) => {
        resolveDiscover = resolve;
      });

    workspace.workspaceFolders = [{ name: "demo", uri: { fsPath: workspaceRoot } }];
    __setConfiguration("afs.discovery.autoDiscover", true);

    const provider = new ContextTreeProvider(
      new ContextService(transport),
      new FileService(transport),
    );
    const roots = await provider.getChildren();

    assert.ok(roots.some((root) => (root as { contextPath?: string }).contextPath === contextPath));
    assert.strictEqual(
      transport.toolCalls.some((call) => call.name === "context.discover"),
      true,
    );
    assert.strictEqual(
      transport.toolCalls.some((call) => call.name === "context.status"),
      false,
    );

    resolveDiscover?.({ contexts: [] });
    await new Promise((resolve) => setTimeout(resolve, 0));
  });

  it("refreshes roots after background auto-discovery completes", async () => {
    const workspaceRoot = makeWorkspace();
    const contextPath = path.join(workspaceRoot, ".context");
    fs.mkdirSync(contextPath, { recursive: true });
    const discoveredPath = path.join(workspaceRoot, "other", ".context");

    let resolveDiscover: ((value: Record<string, unknown>) => void) | undefined;
    const transport = new MockTransport();
    transport.toolHandlers["context.discover"] = async () =>
      await new Promise<Record<string, unknown>>((resolve) => {
        resolveDiscover = resolve;
      });

    workspace.workspaceFolders = [{ name: "demo", uri: { fsPath: workspaceRoot } }];
    __setConfiguration("afs.discovery.autoDiscover", true);

    const provider = new ContextTreeProvider(
      new ContextService(transport),
      new FileService(transport),
    );

    const firstRoots = await provider.getChildren();
    assert.ok(
      firstRoots.some((root) => (root as { contextPath?: string }).contextPath === contextPath),
    );
    assert.strictEqual(
      firstRoots.some((root) => (root as { contextPath?: string }).contextPath === discoveredPath),
      false,
    );

    const changed = new Promise<void>((resolve) => {
      provider.onDidChangeTreeData(() => resolve());
    });
    resolveDiscover?.({
      contexts: [{ project: "other", path: discoveredPath, valid: true, mounts: 2 }],
    });
    await changed;

    const secondRoots = await provider.getChildren();
    assert.ok(
      secondRoots.some((root) => (root as { contextPath?: string }).contextPath === contextPath),
    );
    assert.ok(
      secondRoots.some((root) => (root as { contextPath?: string }).contextPath === discoveredPath),
    );
  });

  it("does not fetch freshness during root loading", async () => {
    const workspaceRoot = makeWorkspace();
    const contextPath = path.join(workspaceRoot, ".context");
    fs.mkdirSync(contextPath, { recursive: true });

    const transport = new MockTransport();
    transport.toolHandlers["context.status"] = async (args) => ({
      context_path: String(args.context_path ?? contextPath),
      mount_counts: { knowledge: 1 },
      total_files: 1,
    });
    transport.toolHandlers["context.freshness"] = async () => ({
      mount_scores: { knowledge: 0.9 },
    });

    workspace.workspaceFolders = [{ name: "demo", uri: { fsPath: workspaceRoot } }];
    __setConfiguration("afs.discovery.autoDiscover", false);

    const provider = new ContextTreeProvider(
      new ContextService(transport),
      new FileService(transport),
    );
    await provider.getChildren();

    assert.strictEqual(
      transport.toolCalls.some((call) => call.name === "context.freshness"),
      false,
    );
  });

  it("exposes mount points as unmountable tree items", async () => {
    const workspaceRoot = makeWorkspace();
    const contextPath = path.join(workspaceRoot, ".context");
    const knowledgePath = path.join(contextPath, "knowledge");
    const mountPath = path.join(knowledgePath, "notes");
    fs.mkdirSync(mountPath, { recursive: true });

    const transport = new MockTransport();
    transport.toolHandlers["context.status"] = async (args) => ({
      context_path: String(args.context_path ?? contextPath),
      mount_counts: { knowledge: 1 },
      total_files: 1,
    });
    transport.toolHandlers["context.freshness"] = async () => ({
      mount_scores: { knowledge: 0.75 },
    });
    transport.toolHandlers["context.list"] = async (args) => {
      const requestedPath = String(args.path ?? "");
      if (requestedPath === knowledgePath) {
        return {
          entries: [
            { path: knowledgePath, is_dir: true },
            { path: mountPath, is_dir: true },
          ],
        };
      }
      if (requestedPath === mountPath) {
        return {
          entries: [
            { path: mountPath, is_dir: true },
            { path: path.join(mountPath, "todo.md"), is_dir: false },
          ],
        };
      }
      return { entries: [] };
    };

    workspace.workspaceFolders = [{ name: "demo", uri: { fsPath: workspaceRoot } }];
    __setConfiguration("afs.discovery.autoDiscover", false);

    const provider = new ContextTreeProvider(
      new ContextService(transport),
      new FileService(transport),
    );

    const roots = await provider.getChildren();
    const demoRoot = roots.find(
      (root) => (root as { contextPath?: string }).contextPath === contextPath,
    );
    assert.ok(demoRoot);

    const mountTypes = await provider.getChildren(demoRoot);
    const knowledge = mountTypes.find(
      (item) => (item as { mountType?: string }).mountType === "knowledge",
    );
    assert.ok(knowledge);

    const mountPoints = await provider.getChildren(knowledge);
    assert.strictEqual(mountPoints.length, 1);
    assert.strictEqual((mountPoints[0] as { contextValue?: string }).contextValue, "mountPoint");
    assert.strictEqual((mountPoints[0] as { alias?: string }).alias, "notes");

    const nestedFiles = await provider.getChildren(mountPoints[0]);
    assert.strictEqual(nestedFiles.length, 1);
    assert.strictEqual((nestedFiles[0] as { label?: string }).label, "todo.md");
  });
});

function makeWorkspace(): string {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "afs-vscode-tree-"));
  tempRoots.push(tempRoot);
  return tempRoot;
}
