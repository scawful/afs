/** AFS mount directory roles. Mirrors src/afs/models.py MountType. */
export enum MountType {
  MEMORY = "memory",
  KNOWLEDGE = "knowledge",
  TOOLS = "tools",
  SCRATCHPAD = "scratchpad",
  HISTORY = "history",
  HIVEMIND = "hivemind",
  GLOBAL = "global",
  ITEMS = "items",
  MONOREPO = "monorepo",
}

/** AFS directory policies. Mirrors src/afs/schema.py PolicyType. */
export enum PolicyType {
  READ_ONLY = "read_only",
  WRITABLE = "writable",
  EXECUTABLE = "executable",
}

export interface MountPoint {
  name: string;
  source: string;
  mount_type: MountType;
  is_symlink: boolean;
}

export interface ProjectMetadata {
  created_at: string;
  description: string;
  agents: string[];
  directories: Record<string, string>;
  manual_only: string[];
}

export interface ContextRoot {
  path: string;
  project_name: string;
  is_valid: boolean;
  total_mounts: number;
  metadata: ProjectMetadata;
  mounts: Record<string, MountPoint[]>;
}

export interface DiscoveredContext {
  project: string;
  path: string;
  valid: boolean;
  mounts: number;
}

export interface QueryEntry {
  mount_type: string;
  relative_path: string;
  absolute_path: string;
  is_dir: boolean;
  size_bytes: number;
  modified_at: string | null;
  indexed_at?: string | null;
  content_excerpt?: string;
  content?: string;
  relevance_score?: number;
}

export interface IndexSummary {
  context_path: string;
  db_path: string;
  indexed_at: string;
  rows_written: number;
  rows_deleted: number;
  by_mount_type: Record<string, number>;
  skipped_large_files: number;
  skipped_binary_files: number;
  errors: string[];
}

/** MCP resource definition from resources/list. */
export interface McpResource {
  uri: string;
  name: string;
  description?: string;
  mimeType?: string;
}

/** MCP resource content from resources/read. */
export interface McpResourceContent {
  uri: string;
  mimeType?: string;
  text?: string;
}

/** MCP prompt definition from prompts/list. */
export interface McpPrompt {
  name: string;
  description?: string;
  arguments?: McpPromptArgument[];
}

export interface McpPromptArgument {
  name: string;
  description?: string;
  required?: boolean;
}

/** MCP prompt message from prompts/get. */
export interface McpPromptMessage {
  role: "user" | "assistant";
  content: { type: "text"; text: string };
}

/** Tool spec from tools/list. */
export interface ToolSpec {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
}

/** Default policies per mount type. Mirrors schema.py default_directory_configs. */
export const DEFAULT_POLICIES: Record<MountType, PolicyType> = {
  [MountType.MEMORY]: PolicyType.READ_ONLY,
  [MountType.KNOWLEDGE]: PolicyType.READ_ONLY,
  [MountType.TOOLS]: PolicyType.EXECUTABLE,
  [MountType.SCRATCHPAD]: PolicyType.WRITABLE,
  [MountType.HISTORY]: PolicyType.READ_ONLY,
  [MountType.HIVEMIND]: PolicyType.WRITABLE,
  [MountType.GLOBAL]: PolicyType.WRITABLE,
  [MountType.ITEMS]: PolicyType.WRITABLE,
  [MountType.MONOREPO]: PolicyType.READ_ONLY,
};

export type ConnectionState = "connected" | "disconnected" | "error";
