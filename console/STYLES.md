# Weaviate Query Agent — Brand & Design System

This document is the source of truth for the visual language of the Weaviate
Query Agent. When styling components, follow this guide. Tokens defined here
are implemented as CSS custom properties in `app/globals.css`.

## Voice & positioning

The Query Agent extends Weaviate's identity as **"the AI-native database for a
new generation of software"** into an agentic, conversational surface. Visuals
should feel intelligent, friendly, and technical: dark, deep-space canvases;
luminous cyan-to-green gradients; and the hexagonal agent mascot at the center.
Lean on depth (gradients on dark), personality (the agent face), and clarity
(generous whitespace, strong typographic hierarchy).

## Color

### Main palette
The colors used most often — backgrounds, surfaces, primary actions, body text.

| Token                 | Hex       | Usage                                                    |
| --------------------- | --------- | -------------------------------------------------------- |
| `--color-navy-darkest`| `#000335` | Deepest canvas; hero panels, illustration backgrounds     |
| `--color-navy`        | `#130C49` | Primary dark surface; cards on darkest navy; body text on light bg |
| `--color-gray-light`  | `#DDEBF2` | Primary light surface; muted text on dark surfaces        |

### Agent spectrum
The signature Query Agent colors. Used for the mascot, gradients, icon strokes,
data-flow paths, and interactive accents on dark surfaces.

| Token                 | Hex       | Usage                                        |
| --------------------- | --------- | -------------------------------------------- |
| `--color-cyan`        | `#01C6C9` | Blue end of the spectrum; secondary accents  |
| `--color-blue-green`  | `#01DEA0` | Mid-spectrum; blends, hover states           |
| `--color-green`       | `#01F57A` | Primary accent; CTAs, success, emphasis      |

### Gradients
Gradients are the heart of the Query Agent look. The agent spectrum runs
cyan → blue-green → green; use it for the mascot fill, hero panels, icon
accents, and the magnifier/search motif.

| Token                     | From → To                       |
| ------------------------- | ------------------------------- |
| `--gradient-agent`        | `#01C6C9 → #01F57A`             |
| `--gradient-agent-full`   | `#01C6C9 → #01DEA0 → #01F57A`   |
| `--gradient-navy`         | `#000335 → #130C49`             |

Rules:
- Gradients angle from lower-left (cyan) to upper-right (green) by default.
- Always place agent gradients on navy surfaces — they lose their luminosity
  on light backgrounds.

## Typography

| Role            | Family                      | Weight   | Token            |
| --------------- | --------------------------- | -------- | ---------------- |
| Display / H1–H2 | Montserrat                  | 700 Bold | `--font-display` |
| Subhead / H3–H4 | Montserrat                  | 500 Med  | `--font-display` |
| Body            | Plus Jakarta Sans           | 400 Reg  | `--font-body`    |
| Code / eyebrow  | Menlo (fallback: monospace) | 400      | `--font-mono`    |

Rules:
- Headers and titles are **bold and large**. Don't be timid with hero type.
- Body type stays comfortable; aim for ~16px base, 1.5–1.6 line-height.
- On dark surfaces, body text is `--color-gray-light`; headings may be white.
- Code blocks always use `--font-mono` on `--color-navy` (or darkest navy),
  with green (`#01F57A`) syntax highlights for strings/keys.
- Eyebrow / overline labels use `--font-mono`, uppercase or small caps, tracked.
- Section labels above illustration panels (e.g. "Query agent") are set in
  regular weight, muted gray.

## The agent mascot

The Query Agent is personified by a **flat-top hexagon with a face**: two
rounded dark eyes on an agent-gradient fill, framed by a thin outlined hexagon
with a small "antenna" dot at the top vertex.

- The gradient fill runs cyan → green (`--gradient-agent`).
- The outline frame is thin — white on dark surfaces, navy on light surfaces.
- The antenna dot sits just off the top edge of the outline frame.
- The mascot may be paired with a **magnifying glass motif** to represent
  querying: either encircling the mascot or trailing off its lower-right edge.
- Dotted arcs and orbit rings around the mascot suggest activity/thinking.
- Scale freely, but never distort, rotate to pointy-top, or remove the face.
- For groups (multi-agent illustrations), mascots may vary hue along the agent
  spectrum (e.g. one cyan-blue, one blue-green, one green) while keeping the
  same construction.

## Chat surface

The agent's conversational UI has its own patterns:

- Greeting bubble ("Hello! How can I help you?") on a near-black card with a
  green (`#01F57A`) arrow action.
- Suggested prompts are stacked in a single dark card, one question per line,
  in light gray text.
- The mascot sits beside the chat stack in a rounded-square tile — agent
  gradient background for the primary variant, dark navy background for the
  quiet variant.
- Corners on chat cards are softly rounded (8–12px); mascot tiles are more
  generous (20–24px).

## Iconography

- Line-based icons on dark hexagonal or rounded-square tiles.
- Icon strokes take colors from the agent spectrum (cyan, blue-green, green);
  a secondary blue-violet range (sky/lavender) may be used sparingly for
  variety in dense icon clusters.
- One stroke weight across the set. Don't reweight individual icons.
- Hexagonal tiles are preferred for "capability" icons (search, chat, charts,
  documents, media); rounded squares for app-like contexts.
- Icons cluster in honeycomb arrangements around the central mascot to
  communicate the agent's capabilities.

## Hexagon visual language

Hexagons remain core to Weaviate's identity, and the Query Agent doubles down
on them — the mascot itself is a hexagon.

- **Always flat-top hexagons** (horizontal top and bottom edges). Never pointy-top.
- Honeycomb clusters of dark hexagons with line icons express capability sets.
- Keep hex outlines thin; fill with navy, the agent gradient, or spectrum hues.
- The outlined "frame + fill + antenna" construction is reserved for the
  mascot; plain hexagons don't get faces.

## Illustration & diagram style

- Darkest navy (`--color-navy-darkest`) is the default illustration canvas;
  `--color-navy` for cards and panels layered on it.
- Combine: the mascot, honeycomb icon clusters, dotted/dashed connector lines,
  dotted orbit arcs, magnifier motifs, small UI fragments (windowed cards,
  chat bubbles, chart sparklines in spectrum colors).
- Connector lines are dotted, in muted gray or spectrum green; small ×, ○, and
  □ glyphs scatter as ambient technical texture.
- Device mockups use soft light-gray/gradient screens with dark line-work UI.
- Charts and sparklines use green and cyan strokes on dark panels.
- Human figures, when needed, are silhouettes filled with brand green — never
  photographic.

## Layout & surfaces

- Default dark surface: `--color-navy-darkest` (`#000335`); layered panels use
  `--color-navy` (`#130C49`).
- Default light surface: `--color-gray-light` (`#DDEBF2`), with cards in white.
- Generous whitespace; large content blocks separated by clear margins rather
  than dividers where possible.
- Rounded corners: 4–8px for cards, 6px for buttons, 20–24px for mascot tiles.

## Do / Don't

**Do**
- Pair green (`#01F57A`) CTAs and accents with navy surfaces for maximum
  recognition.
- Use the agent gradient for the mascot, hero panels, and the magnifier motif.
- Use dotted lines and orbit arcs to suggest agent activity.
- Use Menlo for code blocks and short eyebrow labels.
- Lead with the mascot when introducing the Query Agent visually.

**Don't**
- Don't place agent gradients on light backgrounds — keep them on navy.
- Don't use pointy-top hexagons.
- Don't give plain hexagons faces — the face belongs to the mascot only.
- Don't mix icon stroke weights or crowd a view with off-spectrum colors.
- Don't replace Montserrat / Plus Jakarta Sans / Menlo with substitutes.

## Implementation pointer

CSS custom properties for every token above live in `app/globals.css`. When
adding a new component, consume tokens (`var(--color-green)`) — do not
hardcode hex values.