import type {
  ConnectionState,
  McpPrompt,
  McpPromptMessage,
  McpResource,
  McpResourceContent,
  ToolSpec,
} from "../../src/types";
import type {
  ITransportClient,
  ServerCapabilities,
  TransportSessionInfo,
} from "../../src/transport/types";

/** Minimal event emitter for tests (no vscode dependency). */
class SimpleEventEmitter<T> {
  private listeners: Array<(e: T) => void> = [];
  event = (listener: (e: T) => void) => {
    this.listeners.push(listener);
    return { dispose: () => { this.listeners = this.listeners.filter(l => l !== listener); } };
  };
  fire(data: T): void {
    for (const listener of this.listeners) listener(data);
  }
  dispose(): void {
    this.listeners = [];
  }
}

export class MockTransport implements ITransportClient {
  private ready = true;
  private turnCounter = 0;
  private _onConnectionStateChanged = new SimpleEventEmitter<ConnectionState>();
  readonly onConnectionStateChanged = this._onConnectionStateChanged.event;

  public toolResponses: Record<string, Record<string, unknown>> = {};
  public toolErrors: Record<string, Error> = {};
  public toolCalls: Array<{ name: string; args: Record<string, unknown> }> = [];
  public toolHandlers: Record<
    string,
    (args: Record<string, unknown>) => Record<string, unknown> | Promise<Record<string, unknown>>
  > = {};
  public resourceList: McpResource[] = [];
  public promptList: McpPrompt[] = [];
  public turnEvents: Array<Record<string, unknown>> = [];
  public sessionInfo: TransportSessionInfo | undefined;

  async initialize(): Promise<void> {
    this.ready = true;
  }

  isReady(): boolean {
    return this.ready;
  }

  capabilities(): ServerCapabilities {
    return { tools: true, resources: true, prompts: true };
  }

  async callTool(
    name: string,
    args: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    this.toolCalls.push({ name, args });
    if (this.toolErrors[name]) {
      throw this.toolErrors[name];
    }
    if (this.toolHandlers[name]) {
      return await this.toolHandlers[name](args);
    }
    return this.toolResponses[name] ?? {};
  }

  async listTools(): Promise<ToolSpec[]> {
    return [];
  }

  async listResources(): Promise<McpResource[]> {
    return this.resourceList;
  }

  async readResource(uri: string): Promise<McpResourceContent> {
    return { uri, text: "{}" };
  }

  async listPrompts(): Promise<McpPrompt[]> {
    return this.promptList;
  }

  async getPrompt(_name: string): Promise<McpPromptMessage[]> {
    return [];
  }

  async beginTurn(prompt: string, summary?: string): Promise<string> {
    this.turnCounter += 1;
    const turnId = `mock-turn-${this.turnCounter}`;
    this.turnEvents.push({ event: "begin", turnId, prompt, summary: summary ?? "" });
    return turnId;
  }

  async completeTurn(turnId: string, summary?: string): Promise<void> {
    this.turnEvents.push({ event: "complete", turnId, summary: summary ?? "" });
  }

  async failTurn(turnId: string, error: unknown, summary?: string): Promise<void> {
    this.turnEvents.push({
      event: "fail",
      turnId,
      summary: summary ?? "",
      error: error instanceof Error ? error.message : String(error),
    });
  }

  getSessionInfo(): TransportSessionInfo | undefined {
    return this.sessionInfo;
  }

  dispose(): void {
    this.ready = false;
    this._onConnectionStateChanged.dispose();
  }
}
