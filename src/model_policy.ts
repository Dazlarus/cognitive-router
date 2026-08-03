const DEFAULT_OLLAMA_TOOL_MODELS = new Set(["gemma4:latest"]);
const ZAI_AGENT_GENERATION_MODELS = new Set([
  "glm-5.2",
  "glm-5.1",
  "glm-4.7-flash",
]);
const OLLAMA_EMERGENCY_GENERATION_MODELS = new Set([
  "gemma4:latest",
  "qwen2.5-coder:7b",
  "deepseek-coder-v2:latest",
  "phi4:14b",
  "mistral-nemo:12b",
]);

export function configuredOllamaToolModels(): Set<string> {
  const raw = process.env.ROUTER_OLLAMA_TOOL_MODELS;
  if (!raw) return DEFAULT_OLLAMA_TOOL_MODELS;
  return new Set(raw.split(",").map((m) => m.trim()).filter(Boolean));
}

export function modelSupportsTools(provider: string, model: string): boolean {
  if (provider === "zai" || provider === "openrouter" || provider === "gemini") return true;
  if (provider === "ollama") return configuredOllamaToolModels().has(model);
  return false;
}

export function isEmbeddingOnlyModel(provider: string, model: string): boolean {
  const id = `${provider}/${model}`.toLowerCase();
  return (
    id.includes("embedding") ||
    id.includes("embed-text") ||
    id.includes("bge-")
  );
}

export function isVisionModel(provider: string, model: string): boolean {
  const id = `${provider}/${model}`.toLowerCase();
  return (
    id.includes("vision") ||
    id.includes("llava") ||
    id.includes("minicpm-v") ||
    id.includes("qwen2.5vl") ||
    id.includes("moondream") ||
    /^zai\/glm-[0-9.]+v/.test(id)
  );
}

export function isOllamaEmergencyGenerationModel(model: string): boolean {
  return OLLAMA_EMERGENCY_GENERATION_MODELS.has(model);
}

export function isZaiAgentGenerationModel(model: string): boolean {
  return ZAI_AGENT_GENERATION_MODELS.has(model);
}

export function modelTaskTypes(provider: string, model: string): string[] {
  if (isEmbeddingOnlyModel(provider, model)) return ["embedding"];

  const taskTypes = new Set<string>();
  if (provider === "zai") {
    if (isZaiAgentGenerationModel(model)) {
      taskTypes.add("chat");
      taskTypes.add("generation");
    } else if (isVisionModel(provider, model)) {
      taskTypes.add("vision");
    } else {
      taskTypes.add("remote-experimental");
    }
  } else if (provider === "ollama") {
    if (isOllamaEmergencyGenerationModel(model)) {
      taskTypes.add("chat");
      taskTypes.add("generation");
      taskTypes.add("local-emergency");
    } else if (isVisionModel(provider, model)) {
      taskTypes.add("vision");
    } else {
      taskTypes.add("local-experimental");
    }
  } else {
    taskTypes.add("chat");
    taskTypes.add("generation");
    if (isVisionModel(provider, model)) {
      taskTypes.add("vision");
    }
  }

  return Array.from(taskTypes);
}

export function isGenerationModel(provider: string, model: string): boolean {
  return modelTaskTypes(provider, model).includes("generation");
}

export function generationRoutingExclusionReason(provider: string, model: string): string | null {
  if (isEmbeddingOnlyModel(provider, model)) return "embedding_only";
  if (provider === "zai" && isVisionModel(provider, model) && !isZaiAgentGenerationModel(model)) {
    return "vision_only_route";
  }
  if (provider === "zai" && !isZaiAgentGenerationModel(model)) {
    return "not_in_zai_agent_generation_pool";
  }
  if (provider === "ollama" && isVisionModel(provider, model) && !isOllamaEmergencyGenerationModel(model)) {
    return "vision_only_route";
  }
  if (provider === "ollama" && !isOllamaEmergencyGenerationModel(model)) {
    return "not_in_ollama_emergency_generation_pool";
  }
  return null;
}
