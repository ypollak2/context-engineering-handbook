/**
 * Shared metric calculations for context engineering benchmarks.
 */

import { encode } from "gpt-tokenizer";

export function countTokens(text: string): number {
  return encode(text).length;
}

export function precision(truePositives: number, falsePositives: number): number {
  const total = truePositives + falsePositives;
  if (total === 0) return 0;
  return truePositives / total;
}

export function recall(truePositives: number, falseNegatives: number): number {
  const total = truePositives + falseNegatives;
  if (total === 0) return 0;
  return truePositives / total;
}

export function f1Score(prec: number, rec: number): number {
  const total = prec + rec;
  if (total === 0) return 0;
  return (2 * prec * rec) / total;
}

export function compressionRatio(
  originalTokens: number,
  compressedTokens: number
): number {
  if (originalTokens === 0) return 1;
  return compressedTokens / originalTokens;
}

export function effectiveTokenRatio(
  totalTokens: number,
  contributingTokens: number
): number {
  if (totalTokens === 0) return 0;
  return contributingTokens / totalTokens;
}

export function jaccardSimilarity(setA: Set<string>, setB: Set<string>): number {
  if (setA.size === 0 && setB.size === 0) return 1;
  const intersection = new Set([...setA].filter((x) => setB.has(x)));
  const union = new Set([...setA, ...setB]);
  return intersection.size / union.size;
}
