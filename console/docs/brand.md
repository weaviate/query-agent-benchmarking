# Weaviate Console — Brand & Design System

This document is the source of truth for the visual language of the Weaviate
Console. When styling components, follow this guide. Tokens defined here are
implemented as CSS custom properties in `app/globals.css`.

## Voice & positioning

The product positions Weaviate as **"the AI-native database for a new generation
of software."** Visuals should feel clean, contemporary, and technical without
being sterile. Lean on depth (gradients), structure (hexagons), and clarity
(generous whitespace, strong typographic hierarchy).

## Color

### Main palette
The three colors used most often — backgrounds, surfaces, primary actions, body text.

| Token              | Hex       | Usage                                              |
| ------------------ | --------- | -------------------------------------------------- |
| `--color-green`    | `#61BD73` | Primary brand color; CTAs, accents, success states |
| `--color-navy`     | `#130C49` | Primary dark surface; body text on light bg        |
| `--color-off-white`| `#ECF4F8` | Primary light surface                              |

### Secondary palette
Use when more variety is needed (charts, diagrams, status, secondary surfaces).

| Token                  | Hex       |
| ---------------------- | --------- |
| `--color-teal`         | `#7AC7C0` |
| `--color-mint`         | `#BCF0A7` |
| `--color-sky`          | `#7AD6EB` |
| `--color-lavender`     | `#A590DD` |
| `--color-paper`        | `#F7F9FD` |
| `--color-navy-deep`    | `#262262` |
| `--color-periwinkle`   | `#B9C8DE` |

### Accents
Reserved for highlighting / drawing attention. Use sparingly.

| Token             | Hex       | Usage                              |
| ----------------- | --------- | ---------------------------------- |
| `--color-coral`   | `#F4404E` | Error / destructive / critical     |
| `--color-yellow`  | `#F9F15D` | Highlight, warning                 |

### Gradients
Use to add depth and dimension. The first three are signature "Weaviate" gradients.

| Token                    | From → To             |
| ------------------------ | --------------------- |
| `--gradient-green-blue`  | `#3AB54A → #0069B4`   |
| `--gradient-green-yellow`| `#06CD00 → #FFF000`   |
| `--gradient-teal-green`  | `#55CDE0 → #06CD00`   |
| `--gradient-navy`        | `#0C1428 → #1B1464`   |

## Typography

| Role            | Family                  | Weight   | Token                |
| --------------- | ----------------------- | -------- | -------------------- |
| Display / H1–H2 | Plus Jakarta Sans       | 700 Bold | `--font-display`     |
| Subhead / H3–H4 | Plus Jakarta Sans       | 500 Med  | `--font-display`     |
| Body            | Inter                   | 400 Reg  | `--font-body`        |
| Code / eyebrow  | Menlo (fallback: monospace) | 400  | `--font-mono`        |

Rules:
- Headers and titles are **bold and large**. Don't be timid with hero type.
- Body type stays comfortable; aim for ~16px base, 1.5–1.6 line-height.
- Code blocks always use `--font-mono` on a `--color-navy` (or deep navy) background, with green syntax highlights for strings/keys.
- Eyebrow / overline labels use `--font-mono`, uppercase or small caps, tracked.

## Logo usage

- **Horizontal lockup** is preferred (most layouts are wider than tall).
- **Square lockup** for square containers (avatars, app icons).
- The **"W" mark alone** is acceptable only when the Weaviate brand context is already clear.
- Use the **dark logo on light backgrounds**, **white logo on dark backgrounds**.
- **Don't use the colorful logo on colorful (gradient) backgrounds** — switch to single-color (white or dark).
- Maintain clear space equal to the **x-height** of the wordmark on all sides.
- Reserve **3D logo** for large displays only.

## Iconography

- Line-based icons with **flat green fills** for emphasis.
- Two stylesheets: green-on-light and green-on-navy. Pick one per surface; don't mix.
- Stroke weight is consistent across the set. Don't reweight individual icons.
- Hexagonal frames are part of the icon language — embrace them for "data," "node," "object," "module" concepts.

## Hexagon visual language

Hexagons are core to Weaviate's identity — the W mark is built from them, and they recur in icons, backgrounds, and illustrations.

- **Always flat-top hexagons** (horizontal top and bottom edges). Never pointy-top.
- Use as background texture, content frames, node shapes in diagrams, and isometric "cube" structures in illustrations.
- Keep hex outlines thin; fill with brand greens or gradients.

## Illustration & diagram style

- Dark background (`--color-navy`) is the default canvas for illustrations.
- Combine: hexagons, gradient panels, dotted/dashed connector lines, vector-coordinate annotations (`[0.5789]`, `[0.1023]`), small UI fragments (windowed cards, search bubbles).
- Human figures are **silhouettes filled with brand green** — never photographic.
- Technical diagrams (RAG, GFL, multi-tenancy, deployment wheel) use:
  - Light or navy background
  - Hex-framed central node (often the Weaviate W)
  - Green/teal/sky for data flow paths
  - Mono labels for variables and code-like values

## Layout & surfaces

- Default light surface: `--color-off-white` (`#ECF4F8`), with cards on `--color-paper` (`#F7F9FD`) or pure white.
- Default dark surface: `--color-navy` (`#130C49`); deeper accent panels use `--color-navy-deep` (`#262262`).
- Generous whitespace; large content blocks separated by clear margins rather than dividers where possible.
- Rounded corners are subtle (4–8px for cards, 6px for buttons).

## Do / Don't

**Do**
- Pair green CTAs with navy surfaces for maximum brand recognition.
- Use gradients on hero panels, feature cards, and accent backgrounds.
- Use Menlo for code blocks and short eyebrow labels.
- Lead with hexagons when introducing data, vectors, or modules visually.

**Don't**
- Don't put the colorful logo on a gradient or photographic background.
- Don't use pointy-top hexagons.
- Don't use the coral or yellow accents for large surfaces — they're highlights.
- Don't mix green-on-light and green-on-navy icons in the same view.
- Don't replace Plus Jakarta Sans / Inter / Menlo with substitutes.

## Implementation pointer

CSS custom properties for every token above live in `app/globals.css`. When adding a new component, consume tokens (`var(--color-green)`) — do not hardcode hex values.