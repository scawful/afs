import * as path from "node:path";
import * as vscode from "vscode";

export function getActiveWorkspaceFolder(): vscode.WorkspaceFolder | undefined {
  const activeUri = vscode.window.activeTextEditor?.document?.uri;
  if (!activeUri) {
    return undefined;
  }
  return vscode.workspace.getWorkspaceFolder(activeUri);
}

export async function pickWorkspaceFolder(
  placeHolder = "Select workspace folder",
): Promise<vscode.WorkspaceFolder | undefined> {
  const folders = vscode.workspace.workspaceFolders ?? [];
  if (folders.length === 0) {
    vscode.window.showWarningMessage("No workspace folder is open.");
    return undefined;
  }

  const activeFolder = getActiveWorkspaceFolder();
  if (activeFolder) {
    return activeFolder;
  }

  if (folders.length === 1) {
    return folders[0];
  }

  const picked = await vscode.window.showQuickPick(
    folders.map((folder) => ({
      label: folder.name,
      detail: folder.uri.fsPath,
      folder,
    })),
    { placeHolder },
  );
  return picked?.folder;
}

export function resolvePreferredContextPath(): string | undefined {
  const activeFolder = getActiveWorkspaceFolder();
  if (activeFolder) {
    return path.join(activeFolder.uri.fsPath, ".context");
  }

  const folders = vscode.workspace.workspaceFolders ?? [];
  if (folders.length === 1) {
    return path.join(folders[0].uri.fsPath, ".context");
  }

  return undefined;
}
