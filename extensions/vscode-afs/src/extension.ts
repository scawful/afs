import * as vscode from "vscode";
import { registerChatParticipant } from "./chat/participant";
import { TREE_VIEW_ID } from "./constants";
import { registerCommands } from "./commands/index";
import { checkRegistration, detectMcpConfigPath, registerAfs } from "./mcp/registration";
import { ContextService } from "./services/contextService";
import { FileService } from "./services/fileService";
import { IndexService } from "./services/indexService";
import { createTransport, type BinaryInfo } from "./transport/clientFactory";
import { LazyTransportClient } from "./transport/lazyTransportClient";
import type { ITransportClient } from "./transport/types";
import { locateAfsBinary, resolveAfsBinary } from "./utils/binaryLocator";
import { contextPathFromFilePath, shouldIgnoreAutoRebuildPath } from "./utils/contextPaths";
import { getConfig } from "./utils/config";
import { createLogger } from "./utils/logger";
import { AfsCommandTreeProvider } from "./views/commandTreeProvider";
import { ContextTreeProvider } from "./views/contextTreeProvider";
import { AfsDashboardProvider } from "./views/dashboardProvider";
import { AfsStatusBar } from "./views/statusBar";

let client: ITransportClient | undefined;

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  const activatedAt = Date.now();
  const logger = createLogger();
  context.subscriptions.push(logger);

  const binaryInfo: BinaryInfo = locateAfsBinary(logger);
  const binaryInfoReady = resolveAfsBinary(binaryInfo, logger).then((resolved) => {
    Object.assign(binaryInfo, resolved);
    return binaryInfo;
  });

  const transport: ITransportClient = new LazyTransportClient(async () => {
    try {
      const resolvedBinary = await binaryInfoReady;
      const client = await createTransport(resolvedBinary, logger);
      if (!client.isReady()) {
        await client.initialize();
      }
      return client;
    } catch (err) {
      logger.appendLine(`[transport] Deferred init failed: ${err}`);
      vscode.window.showWarningMessage(
        "AFS: Could not connect to backend. Some features may be limited.",
      );
      throw err;
    }
  }, logger);
  client = transport;
  context.subscriptions.push(transport);

  const contextService = new ContextService(transport);
  const fileService = new FileService(transport);
  const indexService = new IndexService(transport);

  const statusBar = new AfsStatusBar(getConfig<boolean>("statusBar.enabled", true));
  context.subscriptions.push(statusBar);
  transport.onConnectionStateChanged((state) =>
    statusBar.update(state, undefined, transport.getSessionInfo()),
  );
  statusBar.update(
    transport.isReady() ? "connected" : "disconnected",
    undefined,
    transport.getSessionInfo(),
  );

  // --- Context Explorer (tree view) ---
  const treeProvider = new ContextTreeProvider(contextService, fileService);
  context.subscriptions.push(
    vscode.window.registerTreeDataProvider(TREE_VIEW_ID, treeProvider),
  );

  // --- Dashboard (webview) ---
  const dashboardProvider = new AfsDashboardProvider(transport, logger, binaryInfo);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      AfsDashboardProvider.viewType,
      dashboardProvider,
    ),
  );

  // --- Commands (tree view) ---
  const commandTreeProvider = new AfsCommandTreeProvider();
  context.subscriptions.push(
    vscode.window.registerTreeDataProvider("afs.commands", commandTreeProvider),
  );

  registerChatParticipant(context, { transport, logger });

  registerCommands(context, {
    transport,
    contextService,
    fileService,
    indexService,
    treeProvider,
    binaryInfo,
    logger,
  });

  // Dashboard and command refresh commands
  context.subscriptions.push(
    vscode.commands.registerCommand("afs.dashboard.refresh", () => {
      dashboardProvider.refresh();
    }),
    vscode.commands.registerCommand("afs.commands.refresh", () => {
      commandTreeProvider.refresh();
    }),
    vscode.commands.registerCommand("afs.chat.open", async () => {
      const attempts: Array<[string, unknown[]]> = [
        ["workbench.action.chat.open", [{ query: "@afs " }]],
        ["chat.open", [{ query: "@afs " }]],
        ["workbench.action.chat.open", []],
        ["chat.open", []],
      ];

      for (const [command, args] of attempts) {
        try {
          await vscode.commands.executeCommand(command, ...args);
          return;
        } catch {
          // try the next known host command
        }
      }

      vscode.window.showInformationMessage(
        "Open the editor chat and mention @afs to use AFS chat.",
      );
    }),
  );

  const watcher = vscode.workspace.createFileSystemWatcher(
    "**/.context/metadata.json",
  );
  watcher.onDidChange(() => treeProvider.refresh());
  watcher.onDidCreate(() => treeProvider.refresh());
  watcher.onDidDelete(() => treeProvider.refresh());
  context.subscriptions.push(watcher);

  let autoRebuildWatcher: vscode.FileSystemWatcher | undefined;
  const autoRebuildTimers = new Map<string, ReturnType<typeof setTimeout>>();
  const autoRebuildBusy = new Set<string>();

  const clearAutoRebuildTimers = (): void => {
    for (const timer of autoRebuildTimers.values()) {
      clearTimeout(timer);
    }
    autoRebuildTimers.clear();
  };

  const scheduleIndexRebuild = (contextPath: string): void => {
    const existing = autoRebuildTimers.get(contextPath);
    if (existing) {
      clearTimeout(existing);
    }

    autoRebuildTimers.set(
      contextPath,
      setTimeout(async () => {
        autoRebuildTimers.delete(contextPath);
        if (!transport.isReady() || autoRebuildBusy.has(contextPath)) {
          return;
        }

        autoRebuildBusy.add(contextPath);
        try {
          logger.appendLine(`[index] Auto-rebuilding index for ${contextPath}`);
          await indexService.rebuild(contextPath);
          treeProvider.refresh();
        } catch (error) {
          logger.appendLine(`[index] Auto-rebuild failed for ${contextPath}: ${error}`);
        } finally {
          autoRebuildBusy.delete(contextPath);
        }
      }, 900),
    );
  };

  const onContextFileChanged = (uri: vscode.Uri): void => {
    if (!getConfig<boolean>("index.autoRebuild", false)) {
      return;
    }

    const fsPath = uri.fsPath;
    if (!fsPath || shouldIgnoreAutoRebuildPath(fsPath)) {
      return;
    }

    const contextPath = contextPathFromFilePath(fsPath);
    if (!contextPath) {
      return;
    }

    scheduleIndexRebuild(contextPath);
  };

  const refreshAutoRebuildWatcher = (): void => {
    autoRebuildWatcher?.dispose();
    autoRebuildWatcher = undefined;
    clearAutoRebuildTimers();

    if (!getConfig<boolean>("index.autoRebuild", false)) {
      return;
    }

    autoRebuildWatcher = vscode.workspace.createFileSystemWatcher("**/.context/**/*");
    autoRebuildWatcher.onDidChange(onContextFileChanged);
    autoRebuildWatcher.onDidCreate(onContextFileChanged);
    autoRebuildWatcher.onDidDelete(onContextFileChanged);
    context.subscriptions.push(autoRebuildWatcher);
  };

  refreshAutoRebuildWatcher();
  context.subscriptions.push({
    dispose: () => {
      autoRebuildWatcher?.dispose();
      clearAutoRebuildTimers();
    },
  });

  const maybeAutoRegister = async (): Promise<void> => {
    if (!getConfig<boolean>("mcp.autoRegister", false)) {
      return;
    }

    const registration = checkRegistration();
    if (registration.registered) {
      return;
    }

    const configPath = detectMcpConfigPath({ preferExisting: true });
    if (!configPath) {
      logger.appendLine("[mcp] Auto-register skipped: no existing MCP config path detected");
      return;
    }

    try {
      const resolvedBinary = await binaryInfoReady;
      await registerAfs(resolvedBinary, logger, {
        configPath,
        interactivePathSelection: false,
      });
    } catch (error) {
      logger.appendLine(`[mcp] Auto-register failed: ${error}`);
    }
  };

  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("afs.statusBar.enabled")) {
        statusBar.setEnabled(getConfig<boolean>("statusBar.enabled", true));
      }

      if (
        event.affectsConfiguration("afs.discovery.autoDiscover") ||
        event.affectsConfiguration("afs.discovery.searchPaths") ||
        event.affectsConfiguration("afs.discovery.maxDepth") ||
        event.affectsConfiguration("afs.treeView.showEmptyMounts")
      ) {
        treeProvider.refresh();
      }

      if (event.affectsConfiguration("afs.index.autoRebuild")) {
        refreshAutoRebuildWatcher();
      }

      if (event.affectsConfiguration("afs.mcp.autoRegister")) {
        void maybeAutoRegister();
      }
    }),
  );

  const caps = transport.capabilities();
  vscode.commands.executeCommand("setContext", "afs.active", transport.isReady());
  vscode.commands.executeCommand("setContext", "afs.mcp.hasResources", caps.resources);
  vscode.commands.executeCommand("setContext", "afs.mcp.hasPrompts", caps.prompts);

  logger.appendLine(
    `[activate] AFS extension ready. Connected: ${transport.isReady()}, ` +
      `Capabilities: tools=${caps.tools} resources=${caps.resources} prompts=${caps.prompts}`,
  );
  logger.appendLine(`[perf] Activation setup completed in ${Date.now() - activatedAt}ms`);

  void maybeAutoRegister();
}

export function deactivate(): void {
  client?.dispose();
  client = undefined;
}
