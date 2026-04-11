export function contextPathFromFilePath(filePath: string): string | null {
  const normalized = normalizePath(filePath);
  const marker = "/.context/";
  const index = normalized.indexOf(marker);
  if (index >= 0) {
    return normalized.slice(0, index + marker.length - 1);
  }
  if (normalized.endsWith("/.context")) {
    return normalized;
  }
  return null;
}

export function shouldIgnoreAutoRebuildPath(filePath: string): boolean {
  const normalized = normalizePath(filePath);
  return (
    normalized.includes("/.context/global/context_index.sqlite3") ||
    normalized.endsWith("/.context/global/context_index.sqlite3")
  );
}

function normalizePath(filePath: string): string {
  return filePath.replace(/\\/g, "/");
}
