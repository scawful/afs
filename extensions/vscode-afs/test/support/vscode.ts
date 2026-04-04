type Listener<T> = (event: T) => unknown;

export interface Disposable {
  dispose(): void;
}

export type Event<T> = (listener: Listener<T>) => Disposable;

export class EventEmitter<T> {
  private listeners = new Set<Listener<T>>();

  readonly event: Event<T> = (listener: Listener<T>) => {
    this.listeners.add(listener);
    return {
      dispose: () => {
        this.listeners.delete(listener);
      },
    };
  };

  fire(event: T): void {
    for (const listener of this.listeners) {
      listener(event);
    }
  }

  dispose(): void {
    this.listeners.clear();
  }
}

export interface OutputChannel {
  appendLine(value: string): void;
  dispose(): void;
  show?(preserveFocus?: boolean): void;
}

export interface StatusBarItem {
  text: string;
  tooltip: string;
  command?: string;
  show(): void;
  dispose(): void;
}

type CommandCallback = (...args: unknown[]) => unknown;

const commandRegistry = new Map<string, CommandCallback>();

let showInputBoxImpl: (options?: unknown) => Promise<string | undefined> = async () => undefined;
let showQuickPickImpl: <T>(items: readonly T[] | Thenable<readonly T[]>, options?: unknown) => Promise<T | undefined> =
  async (items) => {
    const resolved = await Promise.resolve(items);
    return resolved[0];
  };
let showInformationMessageImpl: (...args: unknown[]) => Promise<unknown> = async () => undefined;
let showWarningMessageImpl: (...args: unknown[]) => Promise<unknown> = async () => undefined;
let showErrorMessageImpl: (...args: unknown[]) => Promise<unknown> = async () => undefined;
let showOpenDialogImpl: (options?: unknown) => Promise<Array<{ fsPath: string }> | undefined> = async () => undefined;
let showTextDocumentImpl: (document: unknown) => Promise<unknown> = async (document) => document;
let withProgressImpl: <T>(options: unknown, task: () => Promise<T>) => Promise<T> =
  async (_options, task) => task();
let openTextDocumentImpl: (uri: { fsPath: string }) => Promise<unknown> = async (uri) => ({ uri });
let clipboardText = "";
let lastStatusBarItem: StatusBarItem | undefined;

export const ProgressLocation = {
  Notification: 15,
};

export const StatusBarAlignment = {
  Left: 1,
  Right: 2,
};

export const Uri = {
  file(fsPath: string): { fsPath: string } {
    return { fsPath };
  },
};

export const commands = {
  registerCommand(name: string, callback: CommandCallback): Disposable {
    commandRegistry.set(name, callback);
    return {
      dispose(): void {
        commandRegistry.delete(name);
      },
    };
  },

  async executeCommand<T = unknown>(name: string, ...args: unknown[]): Promise<T | undefined> {
    const callback = commandRegistry.get(name);
    if (!callback) {
      return undefined;
    }
    return (await callback(...args)) as T;
  },
};

export const workspace = {
  workspaceFolders: [] as Array<{ name?: string; uri: { fsPath: string } }>,
  fs: {
    async stat(_uri: { fsPath: string }): Promise<void> {
      throw new Error("ENOENT");
    },
  },
  getConfiguration: () => ({
    get<T>(_section: string, defaultValue: T): T {
      return defaultValue;
    },
  }),
  async openTextDocument(uri: { fsPath: string }): Promise<unknown> {
    return openTextDocumentImpl(uri);
  },
};

export const window = {
  createStatusBarItem(): StatusBarItem {
    const item: StatusBarItem = {
      text: "",
      tooltip: "",
      command: undefined,
      show(): void {},
      dispose(): void {},
    };
    lastStatusBarItem = item;
    return item;
  },
  createOutputChannel(): OutputChannel {
    return {
      appendLine(): void {},
      dispose(): void {},
      show(): void {},
    };
  },
  async showInputBox(options?: unknown): Promise<string | undefined> {
    return showInputBoxImpl(options);
  },
  async showQuickPick<T>(
    items: readonly T[] | Thenable<readonly T[]>,
    options?: unknown,
  ): Promise<T | undefined> {
    return showQuickPickImpl(items, options);
  },
  async showInformationMessage(...args: unknown[]): Promise<unknown> {
    return showInformationMessageImpl(...args);
  },
  async showWarningMessage(...args: unknown[]): Promise<unknown> {
    return showWarningMessageImpl(...args);
  },
  async showErrorMessage(...args: unknown[]): Promise<unknown> {
    return showErrorMessageImpl(...args);
  },
  async showOpenDialog(options?: unknown): Promise<Array<{ fsPath: string }> | undefined> {
    return showOpenDialogImpl(options);
  },
  async showTextDocument(document: unknown): Promise<unknown> {
    return showTextDocumentImpl(document);
  },
  async withProgress<T>(options: unknown, task: () => Promise<T>): Promise<T> {
    return withProgressImpl(options, task);
  },
};

export const env = {
  clipboard: {
    async writeText(value: string): Promise<void> {
      clipboardText = value;
    },
    async readText(): Promise<string> {
      return clipboardText;
    },
  },
};

export function __resetTestState(): void {
  commandRegistry.clear();
  workspace.workspaceFolders = [];
  showInputBoxImpl = async () => undefined;
  showQuickPickImpl = async (items) => {
    const resolved = await Promise.resolve(items);
    return resolved[0];
  };
  showInformationMessageImpl = async () => undefined;
  showWarningMessageImpl = async () => undefined;
  showErrorMessageImpl = async () => undefined;
  showOpenDialogImpl = async () => undefined;
  showTextDocumentImpl = async (document) => document;
  withProgressImpl = async (_options, task) => task();
  openTextDocumentImpl = async (uri) => ({ uri });
  clipboardText = "";
  lastStatusBarItem = undefined;
}

export function __setShowInputBox(
  impl: (options?: unknown) => Promise<string | undefined>,
): void {
  showInputBoxImpl = impl;
}

export function __setShowQuickPick<T>(
  impl: (items: readonly T[] | Thenable<readonly T[]>, options?: unknown) => Promise<T | undefined>,
): void {
  showQuickPickImpl = impl as typeof showQuickPickImpl;
}

export function __setShowInformationMessage(
  impl: (...args: unknown[]) => Promise<unknown>,
): void {
  showInformationMessageImpl = impl;
}

export function __setShowWarningMessage(
  impl: (...args: unknown[]) => Promise<unknown>,
): void {
  showWarningMessageImpl = impl;
}

export function __setShowErrorMessage(
  impl: (...args: unknown[]) => Promise<unknown>,
): void {
  showErrorMessageImpl = impl;
}

export function __setShowOpenDialog(
  impl: (options?: unknown) => Promise<Array<{ fsPath: string }> | undefined>,
): void {
  showOpenDialogImpl = impl;
}

export function __setShowTextDocument(
  impl: (document: unknown) => Promise<unknown>,
): void {
  showTextDocumentImpl = impl;
}

export function __setWithProgress(
  impl: <T>(options: unknown, task: () => Promise<T>) => Promise<T>,
): void {
  withProgressImpl = impl;
}

export function __setOpenTextDocument(
  impl: (uri: { fsPath: string }) => Promise<unknown>,
): void {
  openTextDocumentImpl = impl;
}

export function __getLastStatusBarItem(): StatusBarItem | undefined {
  return lastStatusBarItem;
}
