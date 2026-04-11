import * as path from "node:path";
import * as vscode from "vscode";
import { MountType, type QueryEntry } from "../types";
import type { ITransportClient } from "../transport/types";
import { getConfig } from "../utils/config";
import { extractToolPayload } from "../utils/toolPayload";
import { resolvePreferredContextPath } from "../utils/workspace";
import { buildAfsContextMessage } from "./contextFormatter";
import {
  resolveAfsModelProfile,
  type AfsModelProfile,
  type AfsModelProfileSetting,
} from "./modelProfile";

type ChatCommand = "" | "pack" | "query" | "scratchpad" | "status";
type ChatWorkflow = "general" | "scan_fast" | "edit_fast" | "review_deep" | "root_cause_deep";
type ChatToolProfile = "default" | "context_readonly" | "context_repair" | "edit_and_verify" | "handoff_only";
type ChatPackMode = "focused" | "retrieval" | "full_slice";

interface ChatDeps {
  transport: ITransportClient;
  logger: vscode.OutputChannel;
}

interface ChatSettings {
  modelProfile: AfsModelProfileSetting;
  workflow: ChatWorkflow;
  toolProfile: ChatToolProfile;
  packMode: ChatPackMode;
  maxQueryResults: number;
  includeContent: boolean;
  includeSessionPrompt: boolean;
  mountTypes: MountType[];
}

interface GatheredContext {
  contextPath: string;
  sessionPrompt: string;
  status: Record<string, unknown> | null;
  freshness: Record<string, unknown> | null;
  pack: Record<string, unknown> | null;
  queryEntries: QueryEntry[];
  scratchpadState: string;
  scratchpadDeferred: string;
  references: string[];
}

export function registerChatParticipant(
  context: vscode.ExtensionContext,
  deps: ChatDeps,
): void {
  const chatApi = getChatApi();
  if (!chatApi) {
    deps.logger.appendLine("[chat] Host chat API unavailable; skipping AFS chat participant.");
    return;
  }

  const participant = chatApi.createChatParticipant(
    "afs.chat",
    async (request, chatContext, stream, token) =>
      handleChatRequest(request, chatContext, stream, token, deps),
  );
  if (context.extensionUri) {
    participant.iconPath = vscode.Uri.joinPath(
      context.extensionUri,
      "media",
      "icons",
      "afs-logo.svg",
    );
  }
  context.subscriptions.push(participant);
}

async function handleChatRequest(
  request: vscode.ChatRequest,
  _chatContext: vscode.ChatContext,
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
  deps: ChatDeps,
): Promise<{ metadata: { command: string; modelProfile: AfsModelProfile } }> {
  const command = normalizeCommand(request.command);
  const prompt = normalizePrompt(command, request.prompt);
  const settings = readChatSettings();
  const model = await resolveModel(request.model, deps.logger);
  const modelProfile = resolveAfsModelProfile(settings.modelProfile, model);
  const summary = summarizeCommand(command);

  const turnId = await deps.transport.beginTurn?.(prompt, summary);
  try {
    if (!model) {
      stream.markdown(
        "AFS chat could not find a host chat model. Select a model in the editor chat UI and try again.",
      );
      return { metadata: { command, modelProfile } };
    }

    if (typeof stream.progress === "function") {
      stream.progress("Reading AFS context");
    }
    const gathered = await gatherChatContext(command, prompt, settings, modelProfile, deps);
    if (typeof stream.reference === "function") {
      for (const referencePath of gathered.references.slice(0, 6)) {
        stream.reference(vscode.Uri.file(referencePath));
      }
    }

    const contextMessage = buildAfsContextMessage({
      sessionPrompt: gathered.sessionPrompt,
      status: gathered.status,
      freshness: gathered.freshness,
      pack: gathered.pack,
      queryEntries: gathered.queryEntries,
      scratchpadState: gathered.scratchpadState,
      scratchpadDeferred: gathered.scratchpadDeferred,
    });
    const response = await model.sendRequest(
      [
        vscode.LanguageModelChatMessage.User(buildInstruction(command)),
        vscode.LanguageModelChatMessage.User(contextMessage),
        vscode.LanguageModelChatMessage.User(`User request: ${prompt}`),
      ],
      {},
      token,
    );

    if (typeof stream.progress === "function") {
      stream.progress("Using the selected chat model");
    }
    for await (const fragment of response.text) {
      stream.markdown(fragment);
    }

    await deps.transport.completeTurn?.(
      turnId ?? "",
      `AFS chat completed${command ? ` (${command})` : ""}`,
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    deps.logger.appendLine(`[chat] request failed: ${message}`);
    stream.markdown(`AFS chat failed: ${message}`);
    await deps.transport.failTurn?.(
      turnId ?? "",
      error,
      `AFS chat failed${command ? ` (${command})` : ""}`,
    );
  }

  return { metadata: { command, modelProfile } };
}

async function gatherChatContext(
  command: ChatCommand,
  prompt: string,
  settings: ChatSettings,
  modelProfile: AfsModelProfile,
  deps: ChatDeps,
): Promise<GatheredContext> {
  const contextPath = resolveContextPath(deps.transport);
  const contextArgs = contextPath ? { context_path: contextPath } : {};

  const status = await callOptionalTool(deps.transport, deps.logger, "context.status", contextArgs);
  const freshness = command === "status"
    ? await callOptionalTool(deps.transport, deps.logger, "context.freshness", contextArgs)
    : null;

  const queryEntries = prompt
    ? await queryContext(prompt, contextArgs, settings, deps)
    : [];

  const pack = await callOptionalTool(deps.transport, deps.logger, "session.pack", {
    ...contextArgs,
    query: prompt,
    task: taskForCommand(command, prompt),
    model: modelProfile,
    workflow: settings.workflow,
    tool_profile: settings.toolProfile,
    pack_mode: settings.packMode,
    include_content: settings.includeContent,
    max_query_results: settings.maxQueryResults,
  });

  let scratchpadState = "";
  let scratchpadDeferred = "";
  const references = queryEntries
    .map((entry) => entry.absolute_path)
    .filter((entry): entry is string => typeof entry === "string" && entry.trim().length > 0);

  if (command === "scratchpad" && contextPath) {
    scratchpadState = await readContextText(
      deps.transport,
      deps.logger,
      path.join(contextPath, "scratchpad", "state.md"),
    );
    scratchpadDeferred = await readContextText(
      deps.transport,
      deps.logger,
      path.join(contextPath, "scratchpad", "deferred.md"),
    );

    if (scratchpadState) {
      references.push(path.join(contextPath, "scratchpad", "state.md"));
    }
    if (scratchpadDeferred) {
      references.push(path.join(contextPath, "scratchpad", "deferred.md"));
    }
  }

  return {
    contextPath,
    sessionPrompt: settings.includeSessionPrompt
      ? deps.transport.getSessionInfo()?.promptText ?? ""
      : "",
    status,
    freshness,
    pack,
    queryEntries,
    scratchpadState,
    scratchpadDeferred,
    references: Array.from(new Set(references)),
  };
}

async function queryContext(
  prompt: string,
  contextArgs: Record<string, unknown>,
  settings: ChatSettings,
  deps: ChatDeps,
): Promise<QueryEntry[]> {
  const payload = await callOptionalTool(deps.transport, deps.logger, "context.query", {
    ...contextArgs,
    query: prompt,
    limit: settings.maxQueryResults,
    include_content: settings.includeContent,
    ...(settings.mountTypes.length > 0 ? { mount_types: settings.mountTypes } : {}),
  });

  if (!payload || !Array.isArray(payload.entries)) {
    return [];
  }

  return payload.entries.filter(
    (entry): entry is QueryEntry =>
      !!entry && typeof entry === "object" && !Array.isArray(entry),
  );
}

async function readContextText(
  transport: ITransportClient,
  logger: vscode.OutputChannel,
  filePath: string,
): Promise<string> {
  const payload = await callOptionalTool(transport, logger, "context.read", { path: filePath });
  return typeof payload?.content === "string" ? payload.content : "";
}

async function callOptionalTool(
  transport: ITransportClient,
  logger: vscode.OutputChannel,
  name: string,
  args: Record<string, unknown>,
): Promise<Record<string, unknown> | null> {
  try {
    const result = await transport.callTool(name, args);
    return extractToolPayload(result);
  } catch (error) {
    logger.appendLine(`[chat] ${name} unavailable: ${error}`);
    return null;
  }
}

function normalizeCommand(command?: string): ChatCommand {
  return command === "pack" || command === "query" || command === "scratchpad" || command === "status"
    ? command
    : "";
}

function normalizePrompt(command: ChatCommand, prompt: string): string {
  const trimmed = prompt.trim();
  if (trimmed) {
    return trimmed;
  }
  switch (command) {
    case "status":
      return "Summarize the active AFS context health, freshness, and next actions.";
    case "scratchpad":
      return "Summarize the current scratchpad state and deferred notes for this workspace.";
    case "pack":
      return "Explain the currently prepared AFS session pack and the most relevant context for this workspace.";
    case "query":
      return "Find the most relevant indexed AFS context for this workspace and summarize it.";
    default:
      return "Help with the current workspace using the available AFS context.";
  }
}

function summarizeCommand(command: ChatCommand): string {
  switch (command) {
    case "status":
      return "Summarize AFS context health";
    case "scratchpad":
      return "Review AFS scratchpad state";
    case "pack":
      return "Review AFS session pack";
    case "query":
      return "Search indexed AFS context";
    default:
      return "Answer with AFS context";
  }
}

function taskForCommand(command: ChatCommand, prompt: string): string {
  switch (command) {
    case "status":
      return "Summarize context health, freshness, and recommended next steps for the current workspace.";
    case "scratchpad":
      return "Use scratchpad state and deferred notes to answer the current workspace question.";
    case "pack":
      return "Explain the current AFS session pack and answer the user with grounded workspace context.";
    case "query":
      return `Use indexed AFS context to answer: ${prompt}`;
    default:
      return `Answer the user using the active AFS workspace context: ${prompt}`;
  }
}

function buildInstruction(command: ChatCommand): string {
  const base = [
    "You are AFS, a workspace-context assistant inside the editor.",
    "Use the provided AFS context blocks as the main grounding source.",
    "Prefer concise, file-grounded answers.",
    "If context appears stale or missing, say so briefly before answering.",
  ];
  if (command === "status") {
    base.push("Focus on health, freshness, risk, and the next best action.");
  }
  if (command === "query") {
    base.push("Prioritize the indexed query hits and call out the most relevant files.");
  }
  if (command === "scratchpad") {
    base.push("Treat scratchpad state and deferred notes as the primary context source.");
  }
  if (command === "pack") {
    base.push("Explain what the current session pack contributes before answering the request.");
  }
  return base.join(" ");
}

async function resolveModel(
  requested: vscode.LanguageModelChat | undefined,
  logger: vscode.OutputChannel,
): Promise<vscode.LanguageModelChat | undefined> {
  if (requested) {
    return requested;
  }

  const lmApi = getLanguageModelApi();
  if (!lmApi) {
    return undefined;
  }

  try {
    const models = await lmApi.selectChatModels({});
    return models[0];
  } catch (error) {
    logger.appendLine(`[chat] failed to resolve fallback model: ${error}`);
    return undefined;
  }
}

function resolveContextPath(transport: ITransportClient): string {
  const preferredContextPath = resolvePreferredContextPath() ?? "";
  if (preferredContextPath.trim()) {
    return preferredContextPath;
  }

  const sessionContextPath = transport.getSessionInfo()?.contextPath ?? "";
  if (sessionContextPath.trim()) {
    return sessionContextPath;
  }
  return "";
}

function readChatSettings(): ChatSettings {
  return {
    modelProfile: getConfig<AfsModelProfileSetting>("chat.modelProfile", "auto"),
    workflow: getConfig<ChatWorkflow>("chat.workflow", "general"),
    toolProfile: getConfig<ChatToolProfile>("chat.toolProfile", "context_readonly"),
    packMode: getConfig<ChatPackMode>("chat.packMode", "focused"),
    maxQueryResults: Math.max(1, getConfig<number>("chat.maxQueryResults", 6)),
    includeContent: getConfig<boolean>("chat.includeContent", false),
    includeSessionPrompt: getConfig<boolean>("chat.includeSessionPrompt", true),
    mountTypes: parseMountTypes(getConfig<string[]>("chat.mountTypes", [])),
  };
}

function parseMountTypes(values: string[]): MountType[] {
  const allowed = new Set<string>(Object.values(MountType));
  return values.filter((value): value is MountType => allowed.has(value));
}

function getChatApi(): {
  createChatParticipant: typeof vscode.chat.createChatParticipant;
} | null {
  const candidate = (vscode as unknown as {
    chat?: { createChatParticipant?: typeof vscode.chat.createChatParticipant };
  }).chat;
  return candidate && typeof candidate.createChatParticipant === "function"
    ? { createChatParticipant: candidate.createChatParticipant }
    : null;
}

function getLanguageModelApi(): {
  selectChatModels: typeof vscode.lm.selectChatModels;
} | null {
  const candidate = (vscode as unknown as {
    lm?: { selectChatModels?: typeof vscode.lm.selectChatModels };
  }).lm;
  return candidate && typeof candidate.selectChatModels === "function"
    ? { selectChatModels: candidate.selectChatModels }
    : null;
}
