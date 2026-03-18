# Google Stitch Prompting Guide for MoodScape

When using Google Stitch to generate UI designs for MoodScape, use these strategies to maintain the application's minimal dark aesthetic.

## Current Design System

The MoodScape UI follows a "Minimal Dark" design language inspired by Linear, Raycast, and Arc browser.

### Design Tokens

| Token | Value | Usage |
| :--- | :--- | :--- |
| `--bg-primary` | `#0a0a0f` | Page background |
| `--bg-surface` | `rgba(255,255,255,0.025)` | Card/panel backgrounds |
| `--bg-elevated` | `rgba(255,255,255,0.04)` | Input fields |
| `--border-subtle` | `rgba(255,255,255,0.08)` | Borders |
| `--accent` | `#7c3aed` | Interactive elements (violet) |
| `--text-primary` | `#f0f0f3` | Headings, values |
| `--text-secondary` | `#a1a1aa` | Body text |
| `--text-tertiary` | `#71717a` | Labels, hints |

### Typography
- **Primary font**: Inter (300, 400, 500, 600) via Google Fonts
- **Monospace**: JetBrains Mono (400, 500) for numeric values
- **Style**: Sentence case everywhere — no uppercase transforms

### Component Patterns
- **Radio buttons**: Styled as pill-shaped toggles with accent glow on selection
- **Checkboxes**: Styled as iOS-style toggle switches
- **Settings**: Collapsible accordion sections (not tabs)
- **Layout**: Two-column — left "Creative Canvas" (60%) + right "Settings Sidebar" (40%)

## Mood-to-Aesthetic Mapping

| Meditation Mood | Stitch Design Aesthetic | Keywords |
| :--- | :--- | :--- |
| **Deep Calm / Sleep** | Ultra-minimal dark | "Deep dark background," "Soft violet accent," "Generous whitespace," "Subtle frosted glass" |
| **Focus / Clarity** | Clean monochrome | "Thin borders," "Zinc neutrals," "Clean lines," "Inter font," "No decorative elements" |
| **Energy / Morning** | Warm minimal | "Subtle warm gradient," "Higher contrast," "Medium font weights," "Amber or gold accent" |
| **Stress Relief** | Soft dark | "Muted earth accents on dark base," "Soft rounded corners," "Gentle spacing," "Low opacity surfaces" |

## Example Stitch Prompts

### For the Main Generator Screen:
> "A premium desktop web app for AI meditation audio generation. Ultra-minimalist dark theme (#0a0a0f). Soft violet accent (#7c3aed). Inter font. Two-column layout: left column with script textarea, music prompt bar, and generate button; right column with collapsible accordion settings sections. Pill-shaped toggle buttons for mode selection. Floating glass audio player. Inspired by Linear and Raycast."

### For a Session Playback Screen:
> "A minimal dark audio playback screen. Deep charcoal background. Centered waveform visualization with violet accent color. Large play/pause button. Session metadata in light Inter font. Thin progress bar. No decorative elements — every pixel earns its place."

## Integration Tip
The `StitchClient` in `core/stitch_client.py` can construct initial prompts based on meditation metadata. The Stitch MCP tools (`mcp__stitch__generate_screen_from_text`, `mcp__stitch__edit_screens`, `mcp__stitch__generate_variants`) are available for direct design generation and iteration.
