import { execFile } from "child_process";
import { existsSync } from "fs";
import * as path from "path";
import * as vscode from "vscode";
import type { BinaryInfo } from "../transport/clientFactory";

const HELP_PROBE_TIMEOUT_MS = 5000;

/** Locate the AFS binary following the same resolution as scripts/afs. */
export function locateAfsBinary(logger: vscode.OutputChannel): BinaryInfo {
  const config = vscode.workspace.getConfiguration("afs");

  // 1. Explicit command setting
  const explicitCommand = config.get<string>("server.command", "").trim();
  if (explicitCommand) {
    logger.appendLine(`[binary] Using explicit command: ${explicitCommand}`);
    return { command: explicitCommand, args: [], env: {} };
  }

  // 2. Explicit Python path → run as module
  const pythonPath = config.get<string>("server.pythonPath", "").trim();
  if (pythonPath) {
    logger.appendLine(`[binary] Using explicit Python: ${pythonPath}`);
    return { command: pythonPath, args: ["-m", "afs"], env: {} };
  }

  const workspaceFolders = vscode.workspace.workspaceFolders ?? [];

  for (const folder of workspaceFolders) {
    const root = folder.uri.fsPath;

    // 3. Workspace .venv/bin/python with afs installed
    const venvPython = path.join(root, ".venv", "bin", "python");
    if (existsSync(venvPython)) {
      logger.appendLine(`[binary] Found workspace venv: ${venvPython}`);
      return { command: venvPython, args: ["-m", "afs"], env: {} };
    }

    // 4. Workspace scripts/afs (dev mode)
    const scriptsAfs = path.join(root, "scripts", "afs");
    if (existsSync(scriptsAfs)) {
      logger.appendLine(`[binary] Found workspace scripts/afs: ${scriptsAfs}`);
      return {
        command: scriptsAfs,
        args: [],
        env: { AFS_ROOT: root, PYTHONPATH: path.join(root, "src") },
      };
    }
  }

  // PATH probing is deferred to avoid blocking activation.
  logger.appendLine("[binary] Deferring PATH probe until first backend use");
  return { command: "afs", args: [], env: {} };
}

let deferredProbe: Promise<BinaryInfo> | undefined;

export async function resolveAfsBinary(
  initial: BinaryInfo,
  logger: vscode.OutputChannel,
): Promise<BinaryInfo> {
  if (!shouldProbePath(initial)) {
    return initial;
  }
  if (deferredProbe) {
    return deferredProbe;
  }

  deferredProbe = (async () => {
    if (await commandAvailable("afs", ["--help"])) {
      logger.appendLine("[binary] Found afs on PATH");
      return { command: "afs", args: [], env: {} };
    }
    if (await commandAvailable("python3", ["-m", "afs", "--help"])) {
      logger.appendLine("[binary] Using system python3 -m afs");
      return { command: "python3", args: ["-m", "afs"], env: {} };
    }
    logger.appendLine("[binary] No AFS binary found — extension will run in degraded mode");
    return initial;
  })();

  return deferredProbe;
}

function shouldProbePath(binaryInfo: BinaryInfo): boolean {
  return binaryInfo.command === "afs"
    && binaryInfo.args.length === 0
    && Object.keys(binaryInfo.env).length === 0;
}

async function commandAvailable(command: string, args: string[]): Promise<boolean> {
  return await new Promise<boolean>((resolve) => {
    execFile(command, args, { timeout: HELP_PROBE_TIMEOUT_MS }, (error) => {
      resolve(!error);
    });
  });
}
