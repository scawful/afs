import * as vscode from "vscode";
import type {
  ConnectionState,
  McpPrompt,
  McpPromptMessage,
  McpResource,
  McpResourceContent,
  ToolSpec,
} from "../types";

export interface JsonRpcRequest {
  jsonrpc: "2.0";
  id: number | string;
  method: string;
  params?: Record<string, unknown>;
}

export interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: number | string | null;
  result?: Record<string, unknown>;
  error?: { code: number; message: string };
}

export interface JsonRpcNotification {
  jsonrpc: "2.0";
  method: string;
  params?: Record<string, unknown>;
}

/** Server capabilities detected during MCP handshake. */
export interface ServerCapabilities {
  tools: boolean;
  resources: boolean;
  prompts: boolean;
}

export interface SessionCliHints {
  workspacePath: string;
  queryShortcut: string;
  queryCanonical: string;
  indexRebuild: string;
  notes: string[];
}

export interface TransportSessionInfo {
  sessionId: string;
  payloadFile: string;
  contextPath: string;
  promptJson: string;
  promptText: string;
  workspace: string;
  cliHints: SessionCliHints;
}

/** Optional turn lifecycle surface for host-driven session events. */
export interface TurnLifecycleClient {
  beginTurn(prompt: string, summary?: string): Promise<string>;
  completeTurn(turnId: string, summary?: string): Promise<void>;
  failTurn(turnId: string, error: unknown, summary?: string): Promise<void>;
}

/** Transport abstraction for communicating with AFS backend. */
export interface ITransportClient extends vscode.Disposable {
  initialize(): Promise<void>;
  isReady(): boolean;
  capabilities(): ServerCapabilities;

  // Tools
  callTool(name: string, args: Record<string, unknown>): Promise<Record<string, unknown>>;
  listTools(): Promise<ToolSpec[]>;

  // Resources
  listResources(): Promise<McpResource[]>;
  readResource(uri: string): Promise<McpResourceContent>;

  // Prompts
  listPrompts(): Promise<McpPrompt[]>;
  getPrompt(name: string, args?: Record<string, unknown>): Promise<McpPromptMessage[]>;

  // Optional host turn lifecycle
  beginTurn?(prompt: string, summary?: string): Promise<string>;
  completeTurn?(turnId: string, summary?: string): Promise<void>;
  failTurn?(turnId: string, error: unknown, summary?: string): Promise<void>;
  getSessionInfo(): TransportSessionInfo | undefined;

  onConnectionStateChanged: vscode.Event<ConnectionState>;
}
