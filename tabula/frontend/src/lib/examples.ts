// Client-side example parsing, mirroring the backend rules in
// transform/classify.py so users get line-numbered errors before any request.

export interface ExamplePair {
  input: string;
  output: string;
}

export interface ExampleParseResult {
  pairs: ExamplePair[];
  errors: string[];
}

// Split at the first recognized arrow ("->" or "→"); everything after it is
// the output, so commas/spaces/Unicode inside values are preserved.
function splitAtArrow(line: string): { src: string; tgt: string } | null {
  const positions = [line.indexOf("->"), line.indexOf("\u2192")].filter(
    (i) => i >= 0
  );
  if (positions.length === 0) return null;
  const i = Math.min(...positions);
  if (line.startsWith("->", i)) {
    return { src: line.slice(0, i), tgt: line.slice(i + 2) };
  }
  return { src: line.slice(0, i), tgt: line.slice(i + 1) };
}

function validateLabeled(
  labeled: { label: string; src: string; tgt: string }[]
): ExampleParseResult {
  const errors: string[] = [];
  const seen = new Map<string, { output: string; label: string }>();
  const pairs: ExamplePair[] = [];

  for (const { label, src, tgt } of labeled) {
    const s = src.trim();
    const t = tgt.trim();
    if (!s) {
      errors.push(`${label}: missing input before the arrow.`);
      continue;
    }
    if (!t) {
      errors.push(`${label}: missing output after the arrow.`);
      continue;
    }
    const prev = seen.get(s);
    if (prev) {
      if (prev.output !== t) {
        errors.push(
          `${label}: contradictory duplicate - input '${s}' maps to ` +
            `'${prev.output}' (${prev.label}) and '${t}' (${label}).`
        );
      }
      continue; // identical duplicate: keep the first occurrence
    }
    seen.set(s, { output: t, label });
    pairs.push({ input: s, output: t });
  }
  return { pairs, errors };
}

export function parseExamplesText(text: string): ExampleParseResult {
  const labeled: { label: string; src: string; tgt: string }[] = [];
  const errors: string[] = [];

  text.split("\n").forEach((line, idx) => {
    const lineno = idx + 1;
    const stripped = line.trim();
    if (!stripped) return;
    const split = splitAtArrow(stripped);
    if (!split) {
      if (stripped.includes(",")) {
        const comma = stripped.indexOf(",");
        labeled.push({
          label: `Line ${lineno}`,
          src: stripped.slice(0, comma),
          tgt: stripped.slice(comma + 1),
        });
      } else {
        errors.push(
          `Line ${lineno}: expected 'input -> output' or 'input \u2192 output' ` +
            `but found '${stripped}'.`
        );
      }
      return;
    }
    labeled.push({ label: `Line ${lineno}`, src: split.src, tgt: split.tgt });
  });

  const result = validateLabeled(labeled);
  errors.push(...result.errors);
  if (errors.length > 0) {
    return { pairs: [], errors };
  }
  if (result.pairs.length === 0) {
    return { pairs: [], errors: ["At least one example pair is required."] };
  }
  return result;
}
