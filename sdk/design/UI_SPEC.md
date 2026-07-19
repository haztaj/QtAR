# UI Design Spec (application-wide)

> The design specification for the app's UI. Feature-specific designs (e.g. the
> Next-Page Preview) live as sections within this document.
>
> Content is imported from design handoffs on explicit sign-off. **No size
> measurements (font sizes, paddings, line-heights, radii, etc.) have been
> imported yet** — those are held pending review.

---

## Design Tokens (colors)

Application-wide color palette.

### Light mode
| Token | Value | Use |
|---|---|---|
| `--paper` | `#f6f2e8` | current page background |
| `--paper-2` | `#efe9db` | secondary paper |
| `--preview` | `#eee7d7` | preview band background (slightly distinct from paper) |
| `--edge` | `#e2d9c4` | paper edge tone |
| `--ink` | `#2c2823` | general chrome text (Quran glyphs are font-coloured — see note) |
| `--accent` | `#6f61a8` | header glyphs (monochrome — tinted), progress, recitation-highlight base |
| `--accent-2` | `#ebe7f6` | recitation highlight, progress track |
| `--seam` | `rgba(70,52,20,.20)` | seam cast shadow + top hint |
| `--edge-hi` | `rgba(255,255,255,.55)` | lit top edge of current page |
| `--chrome` | `#5b5348` | page number |

### Dark mode
| Token | Value | Use |
|---|---|---|
| `--paper` | `#17151d` | current page background |
| `--paper-2` | `#211f2b` | secondary paper |
| `--preview` | `#2a2736` | preview band (deliberately lighter/cooler than paper for separation) |
| `--edge` | `#3a3648` | paper edge tone |
| `--ink` | `#e9e3d5` | general chrome text (Quran glyphs are font-coloured — see note) |
| `--accent` | `#a99ce4` | header glyphs (monochrome — tinted), progress, recitation-highlight base |
| `--accent-2` | `rgba(169,156,228,.22)` | recitation highlight, progress track |
| `--seam` | `rgba(0,0,0,.82)` | seam cast shadow + top hint |
| `--edge-hi` | `rgba(190,180,222,.42)` | lit top edge (carries the depth cue in dark) |
| `--chrome` | `#8a8496` | page number |

**Glyph-colouring note:** the Quran glyph text is **not** coloured by this palette. The ayah body, the ayah-end markers (۝), and the ornate on-page surah-header banner are COLR/CPAL colour fonts — they carry their own (tajwīd) palette and ignore a text `color`; they can only be recoloured by patching the font's CPAL (as dark mode does), which we deliberately do not do for the palette. So `--ink` is used only for general chrome text, and `--accent`'s "header" applies only to the **monochrome** top-strip glyphs (surah name, juz), which are tintable — not to the markers or the banner.

**Dark-mode note:** the difference between light and dark is not just palette. Dark relies on `--edge-hi` (lit edge) + a lighter `--preview` tone for the depth read, because dark-on-dark cast shadows disappear. Preserve both when porting to another theme system.

---

## Feature: Next-Page Preview (Quran Reader)

### Critical layout constraints
These are the non-negotiable rules that make the feature work. Read before implementing.

1. **The recited (bottom) ayah is always visible.** The screen is an immersive, full-bleed reading view (OS status bar hidden, maximal text size). The ayah currently being recited sits pinned at the **bottom** of the reading area and must never be occluded.

2. **The preview is anchored flush to the TOP of the reading area** (directly under the thin header) with **zero extra space above the previewed ayah**. The previewed ayah must render at the *exact same vertical position* it will occupy as the **first line of the next page** after the turn. This is what makes the transition seamless: when the page advances, the ayah does not move. **Do not add top padding/margin above the preview ayah beyond the page's own normal top text padding.**

3. **The preview crops the top of the current page.** Because the screen is already full, the preview cannot push content down — it overlays/replaces the *top* region, clipping the current page's upper lines. The bottom (recited) ayah stays put.

4. **Full-bleed width — no side insets.** The preview spans edge to edge. Do not assume any left/right padding beyond the page body's own horizontal text padding (which the next page will also use, so horizontal position is stable too).

5. **The "next page" hint is subtle above, expressive below.**
   - *Above* the preview ayah: only a subtle full-width top edge shadow (a faint hint of a sheet behind). It must add **no vertical height** (absolutely positioned overlay).
   - *Below* the preview (the **seam** between preview and current page): this is where there is freedom. The current page reads as a physical sheet lying **in front of** the next page — it casts a soft shadow **up** onto the preview and has a lit top edge.

### Preview readability
- Preview ayah opacity: **0.94** — keep the preview clearly readable. Do **not** heavily fade/dim it; the "next page" signal comes from the depth/seam, not transparency.
