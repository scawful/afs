import * as vscode from "vscode";
import type {
  ConnectionState,
  McpPrompt,
  McpPromptMessage,
  McpResource,
  McpResourceContent,
  ToolSpec,
} from "../types";
import type { ITransportClient, ServerCapabilities, TransportSessionInfo } from "./types";

const DISCONNECTED_CAPABILITIES: ServerCapabilities = {
  tools: false,
  resources: false,
  prompts: false,
};

export class LazyTransportClient implements ITransportClient {
  private client: ITransportClient | undefined;
  private initPromise: Promise<ITransportClient> | undefined;
  private disposed = false;
  private readonly connectionStateEmitter = new vscode.EventEmitter<ConnectionState>();

  readonly onConnectionStateChanged = this.connectionStateEmitter.event;

  constructor(
    private readonly factory: () => Promise<ITransportClient>,
    private readonly logger: vscode.OutputChannel,
  ) {}

  async initialize(): Promise<void> {
    await this.ensureClient();
  }

  isReady(): boolean {
    return this.client?.isReady() ?? false;
  }

  capabilities(): ServerCapabilities {
    return this.client?.capabilities() ?? DISCONNECTED_CAPABILITIES;
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<Record<string, unknown>> {
    const client = await this.ensureClient();
    return client.callTool(name, args);
  }

  async listTools(): Promise<ToolSpec[]> {
    const client = await this.ensureClient();
    return client.listTools();
  }

  async listResources(): Promise<McpResource[]> {
    const client = await this.ensureClient();
    return client.listResources();
  }

  async readResource(uri: string): Promise<McpResourceContent> {
    const client = await this.ensureClient();
    return client.readResource(uri);
  }

  async listPrompts(): Promise<McpPrompt[]> {
    const client = await this.ensureClient();
    return client.listPrompts();
  }

  async getPrompt(name: string, args?: Record<string, unknown>): Promise<McpPromptMessage[]> {
    const client = await this.ensureClient();
    return client.getPrompt(name, args);
  }

  async beginTurn(prompt: string, summary?: string): Promise<string> {
    const client = await this.ensureClient();
    return client.beginTurn ? client.beginTurn(prompt, summary) : "";
  }

  async completeTurn(turnId: string, summary?: string): Promise<void> {
    if (!turnId && !this.client) {
      return;
    }
    const client = await this.ensureClient();
    await client.completeTurn?.(turnId, summary);
  }

  async failTurn(turnId: string, error: unknown, summary?: string): Promise<void> {
    if (!turnId && !this.client) {
      return;
    }
    const client = await this.ensureClient();
    await client.failTurn?.(turnId, error, summary);
  }

  getSessionInfo(): TransportSessionInfo | undefined {
    return this.client?.getSessionInfo();
  }

  dispose(): void {
    this.disposed = true;
    this.client?.dispose();
    this.connectionStateEmitter.dispose();
  }

  private async ensureClient(): Promise<ITransportClient> {
    if (this.disposed) {
      throw new Error("Transport disposed");
    }

    if (this.client) {
      return this.client;
    }

    if (this.initPromise) {
      return this.initPromise;
    }

    const startedAt = Date.now();
    this.initPromise = this.factory()
      .then(async (client) => {
        if (this.disposed) {
          client.dispose();
          throw new Error("Transport disposed");
        }

        const wasReady = client.isReady();
        client.onConnectionStateChanged((state) => {
          this.connectionStateEmitter.fire(state);
        });
        if (!wasReady) {
          await client.initialize();
        }

        this.client = client;
        if (wasReady && client.isReady()) {
          this.connectionStateEmitter.fire("connected");
        }
        this.logger.appendLine(
          `[perf] Transport initialized in ${Date.now() - startedAt}ms`,
        );
        return client;
      })
      .catch((error) => {
        this.initPromise = undefined;
        this.connectionStateEmitter.fire("error");
        throw error;
      });

    return this.initPromise;
  }
}
