import * as path from "path";

export interface McpConfigCandidateOptions {
  home: string;
  workspaceFolders?: string[];
  antigravityContextRoot?: string;
}

export function buildMcpConfigCandidates(options: McpConfigCandidateOptions): string[] {
  const candidates: string[] = [];
  const workspaceFolders = options.workspaceFolders ?? [];

  for (const folder of workspaceFolders) {
    candidates.push(
      path.join(folder, ".cursor", "mcp.json"),
      path.join(folder, ".vscode", "mcp.json"),
    );
  }

  if (options.antigravityContextRoot?.trim()) {
    const root = options.antigravityContextRoot.trim();
    candidates.push(
      path.join(root, "mcp.json"),
      path.join(root, "mcp_config.json"),
    );
  }

  candidates.push(
    path.join(options.home, ".cursor", "mcp.json"),
    path.join(options.home, ".vscode", "mcp.json"),
    path.join(options.home, ".config", "cursor", "mcp.json"),
    path.join(options.home, "Library", "Application Support", "Cursor", "User", "mcp.json"),
    path.join(options.home, "Library", "Application Support", "Code", "User", "mcp.json"),
    path.join(options.home, "Library", "Application Support", "Antigravity", "User", "mcp.json"),
    path.join(
      options.home,
      "Library",
      "Application Support",
      "Claude",
      "claude_desktop_config.json",
    ),
  );

  return Array.from(new Set(candidates));
}

export function resolveExistingMcpConfigPath(
  candidates: string[],
  exists: (candidate: string) => boolean,
): string | null {
  for (const candidate of candidates) {
    if (exists(candidate)) {
      return candidate;
    }
  }
  return null;
}

export function defaultWorkspaceMcpConfigPath(workspaceFolders?: string[]): string | null {
  const firstFolder = workspaceFolders?.[0];
  return firstFolder ? path.join(firstFolder, ".cursor", "mcp.json") : null;
}
