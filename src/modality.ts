// src/modality.ts — Multimodal request detection
// Scans request messages for non-text content blocks (images, audio)
// and returns the set of modalities required by the request.

import { logger } from "./logger.js";

/** The modalities a request may require. Always includes "text". */
export type Modality = "text" | "vision" | "audio";

/** Result of scanning a request for multimodal content. */
export interface ModalityDetectionResult {
  /** All modalities detected, always includes "text". */
  modalities: Modality[];
  /** True if any non-text modality was found. */
  isMultimodal: boolean;
  /** Count of image content blocks found. */
  imageCount: number;
  /** Count of audio content blocks found. */
  audioCount: number;
  /** Human-readable summary for logging. */
  summary: string;
}

/**
 * Detect modalities required by a chat request.
 *
 * Scans all messages for:
 * - `image_url` content blocks (OpenAI vision format)
 * - `image_file` content blocks (Anthropic/OpenAI file format)
 * - `input_audio` content blocks (OpenAI audio format)
 * - Inline base64 image data in string content (data URI detection)
 *
 * Returns a result with the full set of modalities needed.
 * Text-only requests return `{ modalities: ["text"], isMultimodal: false, ... }`.
 */
export function detectModalities(messages: any[]): ModalityDetectionResult {
  const modalities = new Set<Modality>(["text"]);
  let imageCount = 0;
  let audioCount = 0;

  if (!Array.isArray(messages)) {
    return {
      modalities: ["text"],
      isMultimodal: false,
      imageCount: 0,
      audioCount: 0,
      summary: "text-only (no messages)",
    };
  }

  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;

    // Content can be a string or an array of content blocks
    const content = msg.content;

    if (typeof content === "string") {
      // Check for inline base64 image data URIs in string content
      // e.g. "data:image/png;base64,..."
      if (isDataUriImage(content)) {
        modalities.add("vision");
        imageCount++;
      }
      continue;
    }

    if (!Array.isArray(content)) continue;

    for (const block of content) {
      if (!block || typeof block !== "object") continue;

      switch (block.type) {
        case "image_url":
          modalities.add("vision");
          imageCount++;
          break;

        case "image":
          // Anthropic-style image block
          if (block.source || block.image_url) {
            modalities.add("vision");
            imageCount++;
          }
          break;

        case "image_file":
          // OpenAI file-based image
          if (block.image_file) {
            modalities.add("vision");
            imageCount++;
          }
          break;

        case "input_audio":
          // OpenAI audio input format
          if (block.input_audio) {
            modalities.add("audio");
            audioCount++;
          }
          break;

        case "audio":
          // Generic audio block
          if (block.source || block.audio || block.input_audio) {
            modalities.add("audio");
            audioCount++;
          }
          break;

        default:
          // Unknown content block type — check if it looks like an image/audio
          if (block.image_url) {
            modalities.add("vision");
            imageCount++;
          } else if (block.input_audio || block.audio) {
            modalities.add("audio");
            audioCount++;
          }
          break;
      }
    }
  }

  const modalityList = Array.from(modalities);
  const isMultimodal = modalityList.length > 1 || (modalityList.length === 1 && modalityList[0] !== "text");

  const parts: string[] = [];
  if (imageCount > 0) parts.push(`${imageCount} image(s)`);
  if (audioCount > 0) parts.push(`${audioCount} audio(s)`);
  const summary = isMultimodal
    ? `multimodal: ${parts.join(", ")}`
    : "text-only";

  return {
    modalities: modalityList,
    isMultimodal,
    imageCount,
    audioCount,
    summary,
  };
}

/**
 * Quick check: does a data URI string contain an image?
 * Matches patterns like "data:image/png;base64,..." or "data:image/jpeg;base64,..."
 */
function isDataUriImage(text: string): boolean {
  // Quick length check to avoid regex on short strings
  if (text.length < 30) return false;
  return /^data:image\//i.test(text);
}

/**
 * Check whether a model's modalities include the required modality.
 * @param modelModalities - The modalities supported by the model (e.g. ["text", "vision"])
 * @param required - The modality to check for
 * @returns true if the model supports the required modality
 */
export function modelSupportsModality(
  modelModalities: string[],
  required: Modality,
): boolean {
  return modelModalities.includes(required);
}

/**
 * Check whether a model supports ALL required modalities.
 * @param modelModalities - The modalities supported by the model
 * @param requiredModalities - All modalities the request needs
 * @returns true if the model supports every required modality
 */
export function modelSupportsAllModalities(
  modelModalities: string[],
  requiredModalities: Modality[],
): boolean {
  return requiredModalities.every((m) => modelModalities.includes(m));
}
