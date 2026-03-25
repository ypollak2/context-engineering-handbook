/**
 * Needle in Haystack Benchmark.
 *
 * Inserts a specific fact at various positions in a large context and tests
 * if the LLM can retrieve it. Measures attention degradation as context grows.
 */

import type { LLMClient } from "../utils/llm-client.js";
import { countTokens } from "../utils/metrics.js";

const FILLER_PARAGRAPHS: readonly string[] = [
  "The history of paper manufacturing dates back to ancient China, where Cai Lun is credited with inventing the process around 105 AD. The technique involved macerating plant fibers in water, forming them into sheets, and drying them in the sun. This revolutionary invention eventually spread along the Silk Road to the Islamic world and then to Europe.",
  "Modern agriculture relies heavily on crop rotation to maintain soil health. Farmers alternate between nitrogen-fixing legumes and heavy-feeding crops like corn. This practice reduces the need for synthetic fertilizers and helps break pest cycles that develop when the same crop is planted year after year.",
  "The development of municipal water treatment systems in the 19th century dramatically reduced waterborne diseases. Chlorination, first used in Jersey City in 1908, became the standard method for disinfecting public water supplies. Sand filtration and sedimentation remain important steps in the treatment process.",
  "Lighthouse construction evolved significantly over the centuries. Early lighthouses were simple fire towers, while modern ones use sophisticated Fresnel lenses that can project light visible from over 20 nautical miles. Automation has replaced most lighthouse keepers since the late 20th century.",
  "The postal system has been a cornerstone of communication for millennia. The Persian Empire established one of the first organized postal systems around 550 BC. Benjamin Franklin served as the first Postmaster General of the United States, establishing routes that still influence mail delivery today.",
  "Volcanic eruptions have shaped Earth's geography and climate throughout history. The eruption of Mount Tambora in 1815 caused the 'Year Without a Summer' in 1816, leading to crop failures across the Northern Hemisphere. Understanding volcanic activity remains crucial for disaster preparedness.",
  "The invention of the printing press by Johannes Gutenberg around 1440 transformed European society. Mass production of books made knowledge accessible beyond the clergy and nobility. The Gutenberg Bible, printed around 1455, is considered one of the most important books in Western civilization.",
  "Coral reefs support approximately 25% of all marine species despite covering less than 1% of the ocean floor. These complex ecosystems are built by tiny organisms called coral polyps that secrete calcium carbonate. Rising ocean temperatures pose a significant threat through coral bleaching.",
  "The development of railways in the 19th century revolutionized transportation and commerce. The Liverpool and Manchester Railway, opened in 1830, was the first intercity passenger railway. Railways enabled rapid industrialization by connecting raw materials to factories and products to markets.",
  "Beekeeping, or apiculture, has been practiced for thousands of years. Ancient Egyptians kept bees in cylindrical clay hives. Modern beekeeping uses movable frame hives invented by Lorenzo Langstroth in 1851, which allow inspection without destroying the colony.",
  "The science of cartography has evolved from hand-drawn maps to satellite-based geographic information systems. Gerardus Mercator's 1569 world map introduced a projection still widely used for navigation today, though it distorts the relative size of land masses near the poles.",
  "Tea cultivation originated in China and became a global commodity through trade routes established during the Tang Dynasty. The British introduced tea cultivation to India in the 1830s to break China's monopoly. Today, tea is the second most consumed beverage worldwide after water.",
];

interface Needle {
  readonly text: string;
  readonly question: string;
  readonly expectedFragment: string;
}

const NEEDLES: readonly Needle[] = [
  {
    text: "The secret project code name is 'Operation Tangerine Dream' and it launches on March 15th, 2025.",
    question:
      "What is the secret project code name and when does it launch?",
    expectedFragment: "Operation Tangerine Dream",
  },
  {
    text: "The quarterly revenue target was revised to exactly $47.3 million after the board meeting on Tuesday.",
    question: "What is the revised quarterly revenue target?",
    expectedFragment: "$47.3 million",
  },
  {
    text: "The database migration password is 'celestial-fox-9921' and must be rotated every 30 days.",
    question: "What is the database migration password?",
    expectedFragment: "celestial-fox-9921",
  },
  {
    text: "Dr. Elena Vasquez discovered that compound XR-7 reduces inflammation by 73% in clinical trials.",
    question:
      "What compound did Dr. Vasquez study and what was the inflammation reduction?",
    expectedFragment: "XR-7",
  },
  {
    text: "The API rate limit for premium tier users was increased to 15,000 requests per minute effective immediately.",
    question: "What is the API rate limit for premium tier users?",
    expectedFragment: "15,000 requests per minute",
  },
];

interface NeedleTrialResult {
  readonly needleIndex: number;
  readonly position: string;
  readonly positionFraction: number;
  readonly haystackTokens: number;
  readonly found: boolean;
  readonly latencyMs: number;
}

export interface NeedleInHaystackResult {
  readonly benchmark: "needle_in_haystack";
  readonly model: string;
  readonly overallRecall: number;
  readonly recallByPosition: Record<string, number>;
  readonly recallBySize: Record<string, number>;
  readonly totalTrials: number;
  readonly trials: readonly NeedleTrialResult[];
}

function buildHaystack(targetTokens: number): string {
  const paragraphs: string[] = [];
  let currentTokens = 0;
  while (currentTokens < targetTokens) {
    const para =
      FILLER_PARAGRAPHS[Math.floor(Math.random() * FILLER_PARAGRAPHS.length)];
    paragraphs.push(para);
    currentTokens += countTokens(para);
  }
  return paragraphs.join("\n\n");
}

function insertNeedle(
  haystack: string,
  needle: string,
  positionFraction: number
): string {
  const paragraphs = haystack.split("\n\n");
  const insertIndex = Math.max(
    0,
    Math.min(
      Math.floor(paragraphs.length * positionFraction),
      paragraphs.length
    )
  );
  const newParagraphs = [...paragraphs];
  newParagraphs.splice(insertIndex, 0, needle);
  return newParagraphs.join("\n\n");
}

function checkNeedleFound(response: string, expectedFragment: string): boolean {
  return response.toLowerCase().includes(expectedFragment.toLowerCase());
}

const HAYSTACK_SIZES = [1000, 2000, 4000, 8000] as const;
const POSITIONS: readonly [string, number][] = [
  ["beginning", 0.1],
  ["middle", 0.5],
  ["end", 0.9],
];

export async function runNeedleInHaystack(
  client: LLMClient
): Promise<NeedleInHaystackResult> {
  const trials: NeedleTrialResult[] = [];

  for (const size of HAYSTACK_SIZES) {
    const haystack = buildHaystack(size);

    for (const [posName, posFraction] of POSITIONS) {
      const needle = NEEDLES[Math.floor(Math.random() * NEEDLES.length)];
      const context = insertNeedle(haystack, needle.text, posFraction);
      const totalTokens = countTokens(context);

      const response = await client.complete(
        [{ role: "user", content: needle.question }],
        `You are a helpful assistant. Answer the question based only on the following context. Be specific and include exact values.\n\nContext:\n${context}`
      );

      const found = checkNeedleFound(response.content, needle.expectedFragment);

      trials.push({
        needleIndex: NEEDLES.indexOf(needle),
        position: posName,
        positionFraction: posFraction,
        haystackTokens: totalTokens,
        found,
        latencyMs: response.latencyMs,
      });
    }
  }

  const overallRecall =
    trials.length > 0
      ? trials.filter((t) => t.found).length / trials.length
      : 0;

  const recallByPosition: Record<string, number> = {};
  for (const [posName] of POSITIONS) {
    const posTrials = trials.filter((t) => t.position === posName);
    if (posTrials.length > 0) {
      recallByPosition[posName] =
        posTrials.filter((t) => t.found).length / posTrials.length;
    }
  }

  const recallBySize: Record<string, number> = {};
  for (const size of HAYSTACK_SIZES) {
    const sizeTrials = trials.filter((t) => t.haystackTokens >= size * 0.8);
    if (sizeTrials.length > 0) {
      recallBySize[String(size)] =
        sizeTrials.filter((t) => t.found).length / sizeTrials.length;
    }
  }

  return {
    benchmark: "needle_in_haystack",
    model: client.model,
    overallRecall: Math.round(overallRecall * 10000) / 10000,
    recallByPosition: Object.fromEntries(
      Object.entries(recallByPosition).map(([k, v]) => [
        k,
        Math.round(v * 10000) / 10000,
      ])
    ),
    recallBySize: Object.fromEntries(
      Object.entries(recallBySize).map(([k, v]) => [
        k,
        Math.round(v * 10000) / 10000,
      ])
    ),
    totalTrials: trials.length,
    trials,
  };
}
