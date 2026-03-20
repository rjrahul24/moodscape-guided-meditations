# Design Workflow

This document explains how to work with designs for MoodScape using **StitchMCP** and **Figma**.

## Design Creation (Stitch)

For generating new design concepts and UI code, we use **StitchMCP**.

1.  **Generate a Screen**: Use the `generate_screen_from_text` tool.
2.  **View Project**: All designs are currently under the project **MoodScape UI** (ID: `1323513399333524651`).
3.  **Iterate**: Use `edit_screens` or `generate_variants` to refine the design.

## Design Reference (Figma)

For using existing hand-crafted designs:

1.  **Config**: The Figma Personal Access Token is stored in `.env` as `FIGMA_ACCESS_TOKEN`.
2.  **Tools**: Use the Figma MCP server tools (`get_file`, `get_node`) to read design tokens and layouts.
3.  **Setup**: To enable the Figma tools in your client, add the following to your config:

```json
{
  "mcpServers": {
    "figma": {
      "url": "https://mcp.figma.com/mcp",
      "env": {
        "FIGMA_PERSONAL_ACCESS_TOKEN": "your_token_from_env"
      }
    }
  }
}
```

## Implementation

Design screens from Stitch can be used as the basis for the Gradio frontend or standalone web interfaces in MoodScape.
