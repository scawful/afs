import type { ITransportClient } from "../transport/types";
import type { DiscoveredContext, MountPoint, MountType } from "../types";

export interface ContextInitResult {
  context_path: string;
  project: string;
  valid: boolean;
  mounts: number;
}

export interface ContextStatus {
  context_path: string;
  mount_counts: Record<string, number>;
  total_files: number;
  [key: string]: unknown;
}

export interface FreshnessResult {
  mount_scores: Record<string, number>;
  [key: string]: unknown;
}

export class ContextService {
  constructor(private readonly transport: ITransportClient) {}

  async discover(searchPaths?: string[], maxDepth?: number): Promise<DiscoveredContext[]> {
    const args: Record<string, unknown> = {};
    if (searchPaths?.length) args.search_paths = searchPaths;
    if (maxDepth != null) args.max_depth = maxDepth;
    const result = await this.transport.callTool("context.discover", args);
    return (result.contexts ?? []) as DiscoveredContext[];
  }

  async mount(
    source: string,
    mountType: MountType,
    contextPath?: string,
    alias?: string,
  ): Promise<MountPoint> {
    const args: Record<string, unknown> = { source, mount_type: mountType };
    if (contextPath) args.context_path = contextPath;
    if (alias) args.alias = alias;
    const result = await this.transport.callTool("context.mount", args);
    return result.mount as MountPoint;
  }

  async init(
    projectPath: string,
    options?: {
      contextRoot?: string;
      contextDir?: string;
      profile?: string;
      linkContext?: boolean;
      force?: boolean;
    },
  ): Promise<ContextInitResult> {
    const args: Record<string, unknown> = { project_path: projectPath };
    if (options?.contextRoot) args.context_root = options.contextRoot;
    if (options?.contextDir) args.context_dir = options.contextDir;
    if (options?.profile) args.profile = options.profile;
    if (options?.linkContext != null) args.link_context = options.linkContext;
    if (options?.force != null) args.force = options.force;
    return (await this.transport.callTool(
      "context.init",
      args,
    )) as unknown as ContextInitResult;
  }

  async unmount(
    mountType: MountType,
    alias: string,
    contextPath?: string,
  ): Promise<boolean> {
    const args: Record<string, unknown> = {
      mount_type: mountType,
      alias,
    };
    if (contextPath) args.context_path = contextPath;
    const result = await this.transport.callTool("context.unmount", args);
    return Boolean(result.removed);
  }

  async status(contextPath?: string): Promise<ContextStatus | null> {
    try {
      const args: Record<string, unknown> = {};
      if (contextPath) args.context_path = contextPath;
      const result = await this.transport.callTool("context.status", args);
      return result as unknown as ContextStatus;
    } catch {
      return null;
    }
  }

  async freshness(
    contextPath?: string,
    mountType?: string,
  ): Promise<FreshnessResult | null> {
    try {
      const args: Record<string, unknown> = {};
      if (contextPath) args.context_path = contextPath;
      if (mountType) args.mount_type = mountType;
      const result = await this.transport.callTool("context.freshness", args);
      return result as unknown as FreshnessResult;
    } catch {
      return null;
    }
  }
}
