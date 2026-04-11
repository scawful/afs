import "vscode";

declare module "vscode" {
  export interface ChatRequest {
    prompt: string;
    command?: string;
    model?: LanguageModelChat;
  }

  export interface ChatContext {
    history: readonly unknown[];
  }

  export interface ChatResponseStream {
    markdown(value: string): void;
    progress(value: string): void;
    reference(value: unknown): void;
  }

  export interface ChatParticipant extends Disposable {
    iconPath?: Uri;
  }

  export type ChatRequestHandler = (
    request: ChatRequest,
    context: ChatContext,
    stream: ChatResponseStream,
    token: CancellationToken,
  ) => Promise<unknown>;

  export namespace chat {
    function createChatParticipant(id: string, handler: ChatRequestHandler): ChatParticipant;
  }

  export interface LanguageModelChat {
    id: string;
    name?: string;
    family?: string;
    vendor?: string;
    maxInputTokens?: number;
    sendRequest(
      messages: readonly LanguageModelChatMessage[],
      options: Record<string, unknown>,
      token: CancellationToken,
    ): Thenable<LanguageModelChatResponse>;
  }

  export interface LanguageModelChatResponse {
    text: AsyncIterable<string>;
  }

  export class LanguageModelChatMessage {
    static User(content: string): LanguageModelChatMessage;
    static Assistant(content: string): LanguageModelChatMessage;
  }

  export namespace lm {
    function selectChatModels(
      selector?: Record<string, string>,
    ): Thenable<readonly LanguageModelChat[]>;
  }

  export class LanguageModelError extends Error {
    readonly code: string;
    readonly cause?: unknown;
  }
}
