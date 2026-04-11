import type { QueryEntry } from "../types";

export interface AfsChatContextPayload {
  sessionPrompt?: string;
  status?: Record<string, unknown> | null;
  freshness?: Record<string, unknown> | null;
  pack?: Record<string, unknown> | null;
  queryEntries?: QueryEntry[];
  scratchpadState?: string;
  scratchpadDeferred?: string;
}

export function buildAfsContextMessage(payload: AfsChatContextPayload): string {
  const blocks: string[] = [];

  if (payload.sessionPrompt?.trim()) {
    blocks.push([
      "## Prepared AFS Session Prompt",
      truncateText(payload.sessionPrompt.trim(), 3200),
    ].join("\n"));
  }

  const statusBlock = renderStatus(payload.status, payload.freshness);
  if (statusBlock) {
    blocks.push(statusBlock);
  }

  const packBlock = renderPack(payload.pack);
  if (packBlock) {
    blocks.push(packBlock);
  }

  const queryBlock = renderQueryEntries(payload.queryEntries ?? []);
  if (queryBlock) {
    blocks.push(queryBlock);
  }

  const scratchpadBlock = renderScratchpad(
    payload.scratchpadState ?? "",
    payload.scratchpadDeferred ?? "",
  );
  if (scratchpadBlock) {
    blocks.push(scratchpadBlock);
  }

  if (blocks.length === 0) {
    return "No AFS context payload was available for this request.";
  }

  return blocks.join("\n\n");
}

function renderStatus(
  status: Record<string, unknown> | null | undefined,
  freshness: Record<string, unknown> | null | undefined,
): string {
  if (!status) {
    return "";
  }

  const lines = ["## AFS Context Status"];
  pushIfString(lines, "Context", status.context_path);
  pushIfString(lines, "Profile", status.profile);

  const mountCounts = recordValue(status.mount_counts);
  if (mountCounts) {
    lines.push(`Mounts: ${formatKeyValueRecord(mountCounts)}`);
  }

  const totalFiles = numberValue(status.total_files);
  if (totalFiles != null) {
    lines.push(`Total files: ${totalFiles}`);
  }

  const index = recordValue(status.index);
  if (index) {
    const fields: string[] = [];
    if (boolValue(index.enabled) != null) {
      fields.push(`enabled=${boolValue(index.enabled) ? "yes" : "no"}`);
    }
    if (boolValue(index.has_entries) != null) {
      fields.push(`has_entries=${boolValue(index.has_entries) ? "yes" : "no"}`);
    }
    if (boolValue(index.stale) != null) {
      fields.push(`stale=${boolValue(index.stale) ? "yes" : "no"}`);
    }
    const totalEntries = numberValue(index.total_entries);
    if (totalEntries != null) {
      fields.push(`entries=${totalEntries}`);
    }
    if (fields.length > 0) {
      lines.push(`Index: ${fields.join(", ")}`);
    }
  }

  const mountHealth = recordValue(status.mount_health);
  if (mountHealth) {
    const fields: string[] = [];
    if (boolValue(mountHealth.healthy) != null) {
      fields.push(`healthy=${boolValue(mountHealth.healthy) ? "yes" : "no"}`);
    }
    const broken = arrayLength(mountHealth.broken_mounts);
    if (broken != null) {
      fields.push(`broken=${broken}`);
    }
    const missing = arrayLength(mountHealth.missing_dirs);
    if (missing != null) {
      fields.push(`missing_dirs=${missing}`);
    }
    if (fields.length > 0) {
      lines.push(`Mount health: ${fields.join(", ")}`);
    }
  }

  const actions = Array.from(
    new Set([
      ...stringArray(status.recommended_actions),
      ...stringArray(status.actions),
      ...stringArray(recordValue(status.mount_health)?.suggested_actions),
    ]),
  );
  if (actions.length > 0) {
    lines.push("Recommended actions:");
    for (const action of actions.slice(0, 5)) {
      lines.push(`- ${action}`);
    }
  }

  const mountScores = recordValue(freshness?.mount_scores);
  if (mountScores) {
    lines.push(`Freshness: ${formatPercentRecord(mountScores)}`);
  }

  return lines.join("\n");
}

function renderPack(pack: Record<string, unknown> | null | undefined): string {
  if (!pack) {
    return "";
  }

  const lines = ["## AFS Session Pack"];
  pushIfString(lines, "Project", pack.project);
  pushIfString(lines, "Profile", pack.profile);
  pushIfString(lines, "Model profile", pack.model);
  pushIfString(lines, "Pack mode", pack.pack_mode);

  const estimatedTokens = numberValue(pack.estimated_tokens);
  if (estimatedTokens != null) {
    lines.push(`Estimated tokens: ${estimatedTokens}`);
  }

  const omittedSections = stringArray(pack.omitted_sections);
  if (omittedSections.length > 0) {
    lines.push(`Omitted sections: ${omittedSections.join(", ")}`);
  }

  const sections = Array.isArray(pack.sections)
    ? pack.sections.filter(
        (section): section is Record<string, unknown> =>
          !!section && typeof section === "object" && !Array.isArray(section),
      )
    : [];
  if (sections.length > 0) {
    lines.push("Sections:");
    for (const section of sections.slice(0, 6)) {
      const title = stringValue(section.title) || "Untitled";
      const body = truncateText(stringValue(section.body) || "", 1800);
      lines.push(`### ${title}`);
      if (body) {
        lines.push(body);
      }
    }
  }

  return lines.join("\n");
}

function renderQueryEntries(entries: QueryEntry[]): string {
  if (entries.length === 0) {
    return "";
  }

  const lines = ["## Indexed Context Hits"];
  for (const entry of entries.slice(0, 6)) {
    lines.push(`- ${entry.mount_type}/${entry.relative_path}`);
    const preview = entry.content?.trim() || entry.content_excerpt?.trim() || "";
    if (preview) {
      lines.push(`  ${truncateText(preview, 320)}`);
    }
  }
  return lines.join("\n");
}

function renderScratchpad(state: string, deferred: string): string {
  const parts: string[] = [];
  if (state.trim()) {
    parts.push(["### scratchpad/state.md", truncateText(state.trim(), 2000)].join("\n"));
  }
  if (deferred.trim()) {
    parts.push(["### scratchpad/deferred.md", truncateText(deferred.trim(), 2000)].join("\n"));
  }
  if (parts.length === 0) {
    return "";
  }
  return ["## Scratchpad Notes", ...parts].join("\n\n");
}

function truncateText(value: string, limit: number): string {
  if (value.length <= limit) {
    return value;
  }
  return `${value.slice(0, Math.max(0, limit - 3)).trimEnd()}...`;
}

function pushIfString(lines: string[], label: string, value: unknown): void {
  const text = stringValue(value);
  if (text) {
    lines.push(`${label}: ${text}`);
  }
}

function stringValue(value: unknown): string {
  return typeof value === "string" && value.trim() ? value.trim() : "";
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function boolValue(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((entry): entry is string => typeof entry === "string" && entry.trim().length > 0);
}

function arrayLength(value: unknown): number | null {
  return Array.isArray(value) ? value.length : null;
}

function formatKeyValueRecord(record: Record<string, unknown>): string {
  return Object.entries(record)
    .map(([key, value]) => `${key}=${formatScalar(value)}`)
    .join(", ");
}

function formatPercentRecord(record: Record<string, unknown>): string {
  return Object.entries(record)
    .map(([key, value]) => {
      const numeric = numberValue(value);
      if (numeric == null) {
        return `${key}=n/a`;
      }
      return `${key}=${Math.round(numeric * 100)}%`;
    })
    .join(", ");
}

function formatScalar(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return "n/a";
}
