## Images (local-image MCP)

This project generates images locally with the `local-image` MCP server
(https://github.com/fedeveloper/local-image-mcp).

- Use the `local-image` tools for new images and for edits of existing images.
  Follow the `local-image` skill for the workflow and the prompts.
- Save images for the project in `<assets directory, for example public/images>/`
  with `output_path`. Use short, lowercase file names with hyphens.
- Default size: `<for example 1216x832 for hero images, 1024x1024 for cards>`.
- Style: `<describe the visual style of the project: palette, light, mood, medium>`.
- Keep the JSON sidecar next to each image. It records the prompt, the model and the
  seed, so the image can be generated again.
- Do not generate images of real, identifiable people without their consent.
