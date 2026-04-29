import * as path from "path";
import * as vscode from "vscode";
import type { BinaryInfo } from "../transport/clientFactory";
import type { ITransportClient, TurnLifecycleClient } from "../transport/types";
import type { ContextService } from "../services/contextService";
import type { FileService } from "../services/fileService";
import type { IndexService } from "../services/indexService";
import { MountType } from "../types";
import { getConfig } from "../utils/config";
import { pickWorkspaceFolder } from "../utils/workspace";
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

interface MountPointSelection {
  contextPath: string;
  mountType: string;
  alias: string;
}

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

export function registerCommands(
  context: vscode.ExtensionContext,
  deps: CommandDeps,
): void {
  const { transport, contextService, fileService, indexService, treeProvider, binaryInfo, logger } =
    deps;

  context.subscriptions.push(
    vscode.commands.registerCommand("afs.treeView.refresh", () => {
      treeProvider.refresh();
    }),

    vscode.commands.registerCommand("afs.context.discover", async () => {
      try {
        const searchPaths = getConfig<string[]>("discovery.searchPaths", [])
          .map((value) => value.trim())
          .filter(Boolean);
        const maxDepth = Math.max(1, getConfig<number>("discovery.maxDepth", 3));
        const contexts = await withRecordedTurn(
          transport,
          "Discover AFS contexts for the open VS Code workspace.",
          "Discover AFS contexts",
          async () => contextService.discover(
            searchPaths.length > 0 ? searchPaths : undefined,
            maxDepth,
          ),
          {
            successSummary: (result) => `Discovered ${result.length} context(s)`,
          },
        );
        treeProvider.setManualDiscoveredContexts(contexts);
        vscode.window.showInformationMessage(`Found ${contexts.length} context(s)`);
      } catch (err) {
        vscode.window.showErrorMessage(`Discovery failed: ${err}`);
      }
    }),

    vscode.commands.registerCommand("afs.context.init", async () => {
      const folder = await pickWorkspaceFolder("Select workspace folder to initialize");
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
      const folder = await pickWorkspaceFolder("Select workspace folder to mount into");
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

    vscode.commands.registerCommand("afs.context.unmount", async (selection?: unknown) => {
      const selectedMount = getMountPointSelection(selection);

      let contextPath: string;
      let mountType: string;
      let alias: string;

      if (selectedMount) {
        const choice = await vscode.window.showWarningMessage(
          `Unmount ${selectedMount.alias} from ${selectedMount.mountType}?`,
          { modal: true },
          "Unmount",
          "Cancel",
        );
        if (choice !== "Unmount") {
          return;
        }
        contextPath = selectedMount.contextPath;
        mountType = selectedMount.mountType;
        alias = selectedMount.alias;
      } else {
        const folder = await pickWorkspaceFolder("Select workspace folder to unmount from");
        if (!folder) return;

        contextPath = path.join(folder.uri.fsPath, ".context");
        const pickedMountType = await vscode.window.showQuickPick(Object.values(MountType), {
          placeHolder: "Select mount type",
        });
        if (!pickedMountType) return;

        mountType = pickedMountType;
        const pickedAlias = await pickMountAlias(
          fileService,
          contextPath,
          pickedMountType as MountType,
        );
        if (!pickedAlias) return;
        alias = pickedAlias;
      }

      try {
        const trimmedAlias = alias.trim();
        const removed = await withRecordedTurn(
          transport,
          `Unmount ${trimmedAlias} from ${mountType} in ${contextPath}`,
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
      const folder = await pickWorkspaceFolder("Select workspace folder to rebuild index for");
      if (!folder) return;
      const contextPath = path.join(folder.uri.fsPath, ".context");
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
      const folder = await pickWorkspaceFolder("Select workspace folder to query");
      if (!folder) return;

      const query = await vscode.window.showInputBox({
        prompt: "Search context index",
      });
      if (!query) return;
      const contextPath = path.join(folder.uri.fsPath, ".context");
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

    vscode.commands.registerCommand("afs.work.communication", async () => {
      const folder = await pickWorkspaceFolder("Select workspace folder for work communication guidance");
      if (!folder) return;
      const contextPath = path.join(folder.uri.fsPath, ".context");
      try {
        const guidance = await withRecordedTurn(
          transport,
          "Inspect AFS work communication samples and approval guardrails.",
          "Review AFS work communication guidance",
          async () =>
            transport.callTool("work.communication.guide", {
              context_path: contextPath,
              limit: 8,
            }),
          {
            successSummary: "Reviewed work communication guidance",
          },
        );
        vscode.window.showInformationMessage(
          formatWorkCommunicationGuidance(guidance),
          { modal: true },
        );
      } catch (err) {
        vscode.window.showErrorMessage(`Work communication guidance failed: ${err}`);
      }
    }),

    vscode.commands.registerCommand("afs.work.approvals", async () => {
      const folder = await pickWorkspaceFolder("Select workspace folder for work approvals");
      if (!folder) return;
      const contextPath = path.join(folder.uri.fsPath, ".context");
      try {
        const payload = await withRecordedTurn(
          transport,
          "Inspect pending AFS work approvals before external writes.",
          "Review AFS work approvals",
          async () =>
            transport.callTool("work.approvals.list", {
              context_path: contextPath,
              status: "pending",
              limit: 20,
            }),
          {
            successSummary: "Reviewed pending work approvals",
          },
        );
        vscode.window.showInformationMessage(formatWorkApprovals(payload), { modal: true });
      } catch (err) {
        vscode.window.showErrorMessage(`Work approvals failed: ${err}`);
      }
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
        if (session.cliHints.workSummary) {
          lines.push(`Work hint: ${session.cliHints.workSummary}`);
        }
        if (session.cliHints.workApprovals) {
          lines.push(`Work approvals hint: ${session.cliHints.workApprovals}`);
        }
        if (session.cliHints.workCommunication) {
          lines.push(`Work communication hint: ${session.cliHints.workCommunication}`);
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

function formatWorkCommunicationGuidance(payload: Record<string, unknown>): string {
  const sampleCount = typeof payload.sample_count === "number" ? payload.sample_count : 0;
  const lines = [`Work communication samples: ${sampleCount}`];
  const styleNotes = Array.isArray(payload.style_notes)
    ? payload.style_notes.filter((note): note is string => typeof note === "string" && note.trim().length > 0)
    : [];
  if (styleNotes.length > 0) {
    lines.push(`Style notes: ${styleNotes.slice(0, 8).join(", ")}`);
  }
  const guidance = Array.isArray(payload.guidance)
    ? payload.guidance.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    : [];
  for (const item of guidance.slice(0, 5)) {
    lines.push(`- ${item}`);
  }
  return lines.join("\n");
}

function formatWorkApprovals(payload: Record<string, unknown>): string {
  const approvals = Array.isArray(payload.approvals)
    ? payload.approvals.filter(
        (approval): approval is Record<string, unknown> =>
          !!approval && typeof approval === "object" && !Array.isArray(approval),
      )
    : [];
  if (approvals.length === 0) {
    return "No pending AFS work approvals.";
  }
  const lines = [`Pending AFS work approvals: ${approvals.length}`];
  for (const approval of approvals.slice(0, 10)) {
    const id = stringValue(approval.approval_id) || "approval";
    const system = stringValue(approval.target_system) || "external";
    const action = stringValue(approval.action) || "write";
    const summary = stringValue(approval.summary);
    lines.push(`- ${id}: ${system}/${action}${summary ? ` - ${summary}` : ""}`);
  }
  return lines.join("\n");
}

function stringValue(value: unknown): string {
  return typeof value === "string" && value.trim() ? value.trim() : "";
}

function getMountPointSelection(selection: unknown): MountPointSelection | undefined {
  if (!selection || typeof selection !== "object") {
    return undefined;
  }

  const candidate = selection as Partial<MountPointSelection>;
  if (
    typeof candidate.contextPath === "string" &&
    typeof candidate.mountType === "string" &&
    typeof candidate.alias === "string"
  ) {
    return {
      contextPath: candidate.contextPath,
      mountType: candidate.mountType,
      alias: candidate.alias,
    };
  }

  return undefined;
}

async function pickMountAlias(
  fileService: FileService,
  contextPath: string,
  mountType: MountType,
): Promise<string | undefined> {
  const mountPath = path.join(contextPath, mountType);

  try {
    const entries = await fileService.list(mountPath, 1);
    const aliases = Array.from(
      new Set(
        entries
          .filter((entry) => entry.path !== mountPath)
          .map((entry) => path.basename(entry.path))
          .filter(Boolean),
      ),
    );

    if (aliases.length > 0) {
      const picked = await vscode.window.showQuickPick(
        aliases.map((label) => ({ label })),
        { placeHolder: `Select mount alias to unmount from ${mountType}` },
      );
      return picked?.label;
    }
  } catch {
    // fall back to manual entry below
  }

  const alias = await vscode.window.showInputBox({
    prompt: `Alias to unmount from ${mountType}`,
  });
  return alias?.trim() || undefined;
}
