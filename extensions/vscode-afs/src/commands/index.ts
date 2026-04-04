import * as path from "path";
import * as vscode from "vscode";
import type { BinaryInfo } from "../transport/clientFactory";
import type { ITransportClient, TurnLifecycleClient } from "../transport/types";
import type { ContextService } from "../services/contextService";
import type { FileService } from "../services/fileService";
import type { IndexService } from "../services/indexService";
import { MountType } from "../types";
import type { ContextTreeProvider } from "../views/contextTreeProvider";
import { registerAfs, unregisterAfs, checkRegistration } from "../mcp/registration";

interface CommandDeps {
  transport: ITransportClient;
  contextService: ContextService;
  fileService: FileService;
  indexService: IndexService;
  treeProvider: ContextTreeProvider;
  binaryInfo: BinaryInfo;
  logger: vscode.OutputChannel;
}

type TurnAwareTransport = ITransportClient & TurnLifecycleClient;

function isTurnAwareTransport(transport: ITransportClient): transport is TurnAwareTransport {
  return (
    typeof transport.beginTurn === "function" &&
    typeof transport.completeTurn === "function" &&
    typeof transport.failTurn === "function"
  );
}

async function withRecordedTurn<T>(
  transport: ITransportClient,
  prompt: string,
  summary: string,
  run: () => Promise<T>,
  options: {
    successSummary?: string | ((result: T) => string);
    failureSummary?: string | ((error: unknown) => string);
  } = {},
): Promise<T> {
  if (!isTurnAwareTransport(transport)) {
    return run();
  }

  const turnId = await transport.beginTurn(prompt, summary);
  try {
    const result = await run();
    const successSummary =
      typeof options.successSummary === "function"
        ? options.successSummary(result)
        : options.successSummary ?? `${summary} completed`;
    await transport.completeTurn(turnId, successSummary);
    return result;
  } catch (error) {
    const failureSummary =
      typeof options.failureSummary === "function"
        ? options.failureSummary(error)
        : options.failureSummary ?? `${summary} failed`;
    await transport.failTurn(turnId, error, failureSummary);
    throw error;
  }
}

async function pickWorkspaceFolder(): Promise<vscode.WorkspaceFolder | undefined> {
  const folders = vscode.workspace.workspaceFolders ?? [];
  if (folders.length === 0) {
    vscode.window.showWarningMessage("No workspace folder is open.");
    return undefined;
  }
  if (folders.length === 1) {
    return folders[0];
  }
  const picked = await vscode.window.showQuickPick(
    folders.map((folder) => ({ label: folder.name, detail: folder.uri.fsPath, folder })),
    { placeHolder: "Select workspace folder" },
  );
  return picked?.folder;
}

export function registerCommands(
  context: vscode.ExtensionContext,
  deps: CommandDeps,
): void {
  const { transport, contextService, indexService, treeProvider, binaryInfo, logger } =
    deps;

  context.subscriptions.push(
    vscode.commands.registerCommand("afs.treeView.refresh", () => {
      treeProvider.refresh();
    }),

    vscode.commands.registerCommand("afs.context.discover", async () => {
      try {
        const contexts = await withRecordedTurn(
          transport,
          "Discover AFS contexts for the open VS Code workspace.",
          "Discover AFS contexts",
          async () => contextService.discover(),
          {
            successSummary: (result) => `Discovered ${result.length} context(s)`,
          },
        );
        treeProvider.refresh();
        vscode.window.showInformationMessage(`Found ${contexts.length} context(s)`);
      } catch (err) {
        vscode.window.showErrorMessage(`Discovery failed: ${err}`);
      }
    }),

    vscode.commands.registerCommand("afs.context.init", async () => {
      const folder = await pickWorkspaceFolder();
      if (!folder) return;

      const contextPath = path.join(folder.uri.fsPath, ".context");
      let force = false;
      try {
        await vscode.workspace.fs.stat(vscode.Uri.file(contextPath));
        const choice = await vscode.window.showWarningMessage(
          `.context already exists in ${folder.name}. Recreate it with --force?`,
          { modal: true },
          "Force Recreate",
          "Cancel",
        );
        if (choice !== "Force Recreate") return;
        force = true;
      } catch {
        // context does not exist yet
      }

      try {
        const result = await withRecordedTurn(
          transport,
          `Initialize AFS context for ${folder.uri.fsPath}${force ? " (force)" : ""}`,
          "Initialize AFS context",
          async () => contextService.init(folder.uri.fsPath, { force }),
          {
            successSummary: (created) => `Initialized context at ${created.context_path}`,
          },
        );
        treeProvider.refresh();
        vscode.window.showInformationMessage(
          `Initialized context at ${result.context_path}`,
        );
      } catch (err) {
        vscode.window.showErrorMessage(`Context init failed: ${err}`);
      }
    }),

    vscode.commands.registerCommand("afs.context.mount", async () => {
      const folder = await pickWorkspaceFolder();
      if (!folder) return;

      const sourcePick = await vscode.window.showOpenDialog({
        canSelectFiles: true,
        canSelectFolders: true,
        canSelectMany: false,
        openLabel: "Select Source to Mount",
      });
      if (!sourcePick?.length) return;
      const sourcePath = sourcePick[0].fsPath;

      const mountType = await vscode.window.showQuickPick(Object.values(MountType), {
        placeHolder: "Select mount type",
      });
      if (!mountType) return;

      const suggestedAlias = path.basename(sourcePath);
      const aliasInput = await vscode.window.showInputBox({
        prompt: "Mount alias (optional)",
        value: suggestedAlias,
      });
      if (aliasInput === undefined) return;
      const alias = aliasInput.trim() || undefined;

      try {
        const contextPath = path.join(folder.uri.fsPath, ".context");
        const mounted = await withRecordedTurn(
          transport,
          `Mount ${sourcePath} as ${alias ?? suggestedAlias} in ${mountType}`,
          "Mount AFS context source",
          async () =>
            contextService.mount(
              sourcePath,
              mountType as MountType,
              contextPath,
              alias,
            ),
          {
            successSummary: (result) => `Mounted ${result.name} in ${result.mount_type}`,
          },
        );
        treeProvider.refresh();
        vscode.window.showInformationMessage(
          `Mounted ${mounted.name} in ${mounted.mount_type}`,
        );
      } catch (err) {
        vscode.window.showErrorMessage(`Mount failed: ${err}`);
      }
    }),

    vscode.commands.registerCommand("afs.context.unmount", async () => {
      const folder = await pickWorkspaceFolder();
      if (!folder) return;

      const mountType = await vscode.window.showQuickPick(Object.values(MountType), {
        placeHolder: "Select mount type",
      });
      if (!mountType) return;

      const alias = await vscode.window.showInputBox({
        prompt: `Alias to unmount from ${mountType}`,
      });
      if (!alias?.trim()) return;

      try {
        const contextPath = path.join(folder.uri.fsPath, ".context");
        const trimmedAlias = alias.trim();
        const removed = await withRecordedTurn(
          transport,
          `Unmount ${trimmedAlias} from ${mountType}`,
          "Unmount AFS context source",
          async () =>
            contextService.unmount(
              mountType as MountType,
              trimmedAlias,
              contextPath,
            ),
          {
            successSummary: (result) =>
              result
                ? `Unmounted ${trimmedAlias} from ${mountType}`
                : `No mount named ${trimmedAlias} found in ${mountType}`,
          },
        );
        treeProvider.refresh();
        if (!removed) {
          vscode.window.showWarningMessage(
            `No mount named "${trimmedAlias}" found in ${mountType}.`,
          );
          return;
        }
        vscode.window.showInformationMessage(
          `Unmounted ${trimmedAlias} from ${mountType}.`,
        );
      } catch (err) {
        vscode.window.showErrorMessage(`Unmount failed: ${err}`);
      }
    }),

    vscode.commands.registerCommand("afs.index.rebuild", async () => {
      const folders = vscode.workspace.workspaceFolders;
      if (!folders?.length) return;
      const contextPath = `${folders[0].uri.fsPath}/.context`;
      try {
        await withRecordedTurn(
          transport,
          `Rebuild AFS context index for ${contextPath}`,
          "Rebuild AFS index",
          async () =>
            vscode.window.withProgress(
              {
                location: vscode.ProgressLocation.Notification,
                title: "Rebuilding AFS index...",
              },
              async () => {
                const summary = await indexService.rebuild(contextPath);
                vscode.window.showInformationMessage(
                  `Index rebuilt: ${summary.rows_written} rows, ${summary.errors.length} errors`,
                );
                return summary;
              },
            ),
          {
            successSummary: (summary) =>
              `Index rebuilt with ${summary.rows_written} rows and ${summary.errors.length} errors`,
          },
        );
        treeProvider.refresh();
      } catch (err) {
        vscode.window.showErrorMessage(`Index rebuild failed: ${err}`);
      }
    }),

    vscode.commands.registerCommand("afs.index.query", async () => {
      const query = await vscode.window.showInputBox({
        prompt: "Search context index",
      });
      if (!query) return;
      const folders = vscode.workspace.workspaceFolders;
      if (!folders?.length) return;
      const contextPath = `${folders[0].uri.fsPath}/.context`;
      try {
        await withRecordedTurn(
          transport,
          query,
          "Search AFS context index",
          async () => {
            const entries = await indexService.query(contextPath, query);
            if (entries.length === 0) {
              vscode.window.showInformationMessage("No results found.");
              return entries;
            }
            const picked = await vscode.window.showQuickPick(
              entries.map((e) => ({
                label: e.relative_path,
                description: `${e.mount_type} (${e.size_bytes} bytes)`,
                detail: e.absolute_path,
              })),
              { placeHolder: `${entries.length} results for "${query}"` },
            );
            if (picked?.detail) {
              const doc = await vscode.workspace.openTextDocument(
                vscode.Uri.file(picked.detail),
              );
              await vscode.window.showTextDocument(doc);
            }
            return entries;
          },
          {
            successSummary: (entries) => `Context query returned ${entries.length} result(s)`,
            failureSummary: `Context query failed for: ${query}`,
          },
        );
      } catch (err) {
        vscode.window.showErrorMessage(`Query failed: ${err}`);
      }
    }),

    vscode.commands.registerCommand("afs.index.queryQuickOpen", async () => {
      await vscode.commands.executeCommand("afs.index.query");
    }),

    vscode.commands.registerCommand("afs.mcp.register", async () => {
      await registerAfs(binaryInfo, logger);
    }),

    vscode.commands.registerCommand("afs.mcp.unregister", async () => {
      await unregisterAfs(logger);
    }),

    vscode.commands.registerCommand("afs.mcp.status", async () => {
      const reg = checkRegistration();
      const caps = transport.capabilities();
      const session = transport.getSessionInfo();
      const lines = [
        `Connected: ${transport.isReady()}`,
        `Capabilities: tools=${caps.tools}, resources=${caps.resources}, prompts=${caps.prompts}`,
        `MCP registered: ${reg.registered}`,
        `Config path: ${reg.configPath ?? "none"}`,
      ];
      if (session) {
        lines.push(`Session workspace: ${session.workspace || "unknown"}`);
        lines.push(`Session payload: ${session.payloadFile || "none"}`);
        if (session.cliHints.queryShortcut) {
          lines.push(`Query hint: ${session.cliHints.queryShortcut}`);
        }
        if (session.cliHints.queryCanonical) {
          lines.push(`Canonical query hint: ${session.cliHints.queryCanonical}`);
        }
        if (session.cliHints.indexRebuild) {
          lines.push(`Index hint: ${session.cliHints.indexRebuild}`);
        }
        for (const note of session.cliHints.notes) {
          if (note.trim()) {
            lines.push(`Note: ${note.trim()}`);
          }
        }
      }
      vscode.window.showInformationMessage(lines.join("\n"), { modal: true });
    }),

    vscode.commands.registerCommand("afs.server.restart", async () => {
      const choice = await vscode.window.showWarningMessage(
        "Restart AFS server by reloading the window?",
        { modal: true },
        "Reload Window",
        "Cancel",
      );
      if (choice !== "Reload Window") return;
      logger.appendLine("[cmd] Reloading window to restart AFS server");
      await vscode.commands.executeCommand("workbench.action.reloadWindow");
    }),

    vscode.commands.registerCommand("afs.server.showLogs", () => {
      logger.show(true);
    }),
  );
}
