export type AfsModelProfile = "generic" | "gemini" | "claude" | "codex";
export type AfsModelProfileSetting = AfsModelProfile | "auto";

export interface ChatModelDescriptor {
  id?: string;
  name?: string;
  family?: string;
  vendor?: string;
}

export function resolveAfsModelProfile(
  configured: AfsModelProfileSetting,
  model?: ChatModelDescriptor,
): AfsModelProfile {
  if (configured !== "auto") {
    return configured;
  }

  const haystack = [model?.id, model?.name, model?.family, model?.vendor]
    .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
    .join(" ")
    .toLowerCase();

  if (haystack.includes("gemini")) {
    return "gemini";
  }
  if (haystack.includes("claude")) {
    return "claude";
  }
  if (haystack.includes("codex")) {
    return "codex";
  }
  return "generic";
}
