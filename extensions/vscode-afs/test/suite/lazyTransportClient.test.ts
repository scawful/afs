import * as assert from "node:assert";
import { describe, it } from "node:test";
import { EventEmitter } from "vscode";
import { LazyTransportClient } from "../../src/transport/lazyTransportClient";
import type {
  ITransportClient,
  ServerCapabilities,
  TransportSessionInfo,
} from "../../src/transport/types";
import type {
  ConnectionState,
  McpPrompt,
  McpPromptMessage,
  McpResource,
  McpResourceContent,
  ToolSpec,
} from "../../src/types";

class StubTransport implements ITransportClient {
  private ready = false;
  private readonly emitter = new EventEmitter<ConnectionState>();

  readonly onConnectionStateChanged = this.emitter.event;
  initializeCalls = 0;
  toolCalls = 0;
  disposeCalls = 0;

  async initialize(): Promise<void> {
    this.initializeCalls += 1;
    this.ready = true;
    this.emitter.fire("connected");
  }

  isReady(): boolean {
    return this.ready;
  }

  capabilities(): ServerCapabilities {
    return { tools: true, resources: false, prompts: false };
  }

  async callTool(): Promise<Record<string, unknown>> {
    this.toolCalls += 1;
    return { ok: true };
  }

  async listTools(): Promise<ToolSpec[]> {
    return [];
  }

  async listResources(): Promise<McpResource[]> {
    return [];
  }

  async readResource(uri: string): Promise<McpResourceContent> {
    return { uri };
  }

  async listPrompts(): Promise<McpPrompt[]> {
    return [];
  }

  async getPrompt(): Promise<McpPromptMessage[]> {
    return [];
  }

  getSessionInfo(): TransportSessionInfo | undefined {
    return undefined;
  }

  dispose(): void {
    this.disposeCalls += 1;
    this.ready = false;
    this.emitter.dispose();
  }
}

describe("LazyTransportClient", () => {
  it("defers transport creation until first use", async () => {
    let factoryCalls = 0;
    const inner = new StubTransport();
    const logs: string[] = [];
    const transport = new LazyTransportClient(
      async () => {
        factoryCalls += 1;
        return inner;
      },
      { appendLine(value: string) { logs.push(value); }, dispose() {} } as never,
    );

    assert.strictEqual(factoryCalls, 0);
    assert.strictEqual(transport.isReady(), false);
    assert.deepStrictEqual(transport.capabilities(), {
      tools: false,
      resources: false,
      prompts: false,
    });

    const result = await transport.callTool("context.status", {});

    assert.deepStrictEqual(result, { ok: true });
    assert.strictEqual(factoryCalls, 1);
    assert.strictEqual(inner.initializeCalls, 1);
    assert.strictEqual(inner.toolCalls, 1);
    assert.strictEqual(transport.isReady(), true);
    assert.strictEqual(
      logs.some((line) => line.includes("[perf] Transport initialized in")),
      true,
    );
  });

  it("shares a single in-flight initialization", async () => {
    let factoryCalls = 0;
    const inner = new StubTransport();
    const transport = new LazyTransportClient(
      async () => {
        factoryCalls += 1;
        await new Promise((resolve) => setTimeout(resolve, 10));
        return inner;
      },
      { appendLine() {}, dispose() {} } as never,
    );

    await Promise.all([
      transport.callTool("context.status", {}),
      transport.listTools(),
    ]);

    assert.strictEqual(factoryCalls, 1);
    assert.strictEqual(inner.initializeCalls, 1);
    assert.strictEqual(inner.toolCalls, 1);
  });

  it("retries initialization after factory failure", async () => {
    let factoryCalls = 0;
    const inner = new StubTransport();
    const transport = new LazyTransportClient(
      async () => {
        factoryCalls += 1;
        if (factoryCalls === 1) {
          throw new Error("factory exploded");
        }
        return inner;
      },
      { appendLine() {}, dispose() {} } as never,
    );

    await assert.rejects(async () => transport.callTool("context.status", {}), /factory exploded/);
    const result = await transport.callTool("context.status", {});

    assert.deepStrictEqual(result, { ok: true });
    assert.strictEqual(factoryCalls, 2);
    assert.strictEqual(inner.initializeCalls, 1);
  });

  it("disposes the created client if transport is disposed during init", async () => {
    let resolveFactory: ((value: ITransportClient) => void) | undefined;
    const inner = new StubTransport();
    const transport = new LazyTransportClient(
      async () =>
      await new Promise<ITransportClient>((resolve) => {
        resolveFactory = resolve;
      }),
      { appendLine() {}, dispose() {} } as never,
    );

    const pendingCall = transport.callTool("context.status", {});
    transport.dispose();
    resolveFactory?.(inner);

    await assert.rejects(async () => pendingCall, /Transport disposed/);
    assert.strictEqual(inner.disposeCalls, 1);
  });
});
