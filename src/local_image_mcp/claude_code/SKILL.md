---
name: local-image
description: Generate or edit images on this computer's GPU with the local-image MCP server. Use when the user asks for an image, photo, illustration, icon, texture, background or mockup asset, or wants to change an existing image with a reference.
---

# Local image generation

The `local-image` MCP server generates images with local diffusion models on the GPU.
Every image is a PNG with a JSON sidecar that can reproduce it.

## Workflow

1. Call `list_models` once per session. Use only a model with `available: true`.
   If no model is available, tell the user the `unavailable_reason` and the command
   `local-image models pull <id>`. Do not download anything yourself.
2. Call `generate_image`. It returns a `job_id` at once, with the resolved settings.
3. Call `get_job_status` with `wait_seconds: 45` until `state` is `succeeded`,
   `failed` or `cancelled`. One image takes about 20–60 seconds.
4. Call `get_job_result`. Give the user the `image_path`.
   Set `include_preview: true` only when you must look at the image (for example, to
   check it before you use it in the project).
5. One job runs at a time. For several images, start them one after the other; they
   wait in the queue.

## Choose the model

| Need | Model | Notes |
|---|---|---|
| A new image from text (photo, illustration) | `z-image-turbo` | 8 steps, photorealistic, renders short text |
| A new image, fast | `flux2-klein-4b` | 4 steps |
| Change an image, or use 1–4 images as input | `flux2-klein-4b` | `reference_images` + an edit prompt |

Read `prompt_notes` in `list_models` for each model.

## Write the prompt

- Write in English, in full sentences: subject, setting, composition, lighting, style,
  mood. Be concrete: "a red ceramic mug on an oak desk, morning side light, shallow depth
  of field, product photo" is better than "nice mug".
- Put text that must appear in the image in quotes: `a poster with the title "OPEN DAY"`.
- These models ignore negative prompts. Write what you want, not what you do not want.
- For an edit, describe only the change and what must stay:
  "Make the jacket bright blue. Keep the face, the pose and the background."
- With several references, refer to them by order: "the person from image 1 in the
  room of image 2".

## Size

Use sizes that are multiples of 16. Common choices (about 1 megapixel):

| Shape | Size |
|---|---|
| Square | 1024×1024 |
| Landscape 3:2 / 16:9 | 1216×832 / 1344×768 |
| Portrait 2:3 / 9:16 | 832×1216 / 768×1344 |

With `reference_images` and no size, the output keeps the shape of the first reference.

## Files

- Default output: `<project>/generated-images/<date>_<model>_<seed>.png` + `.json`.
- Use `output_path` to save into the project, for example `public/images/hero.png`.
  Existing files are never overwritten; the server adds a suffix.
- Reference images can be relative to the project directory.
- To make variants, change only the `seed`. To reproduce an image, reuse all settings
  from its JSON sidecar (or run `local-image regenerate <file>.json`).

## Errors

- Validation errors come back at once from `generate_image`. Read the message: it names
  the problem (unknown model, size, missing file, unsupported capability).
- A `failed` job has an `error` text. Report it to the user. Do not retry more than once
  with the same settings.

## Responsible use

Do not generate images of real, identifiable people without their consent, and do not
create content that deceives or harms. Check the license of each model (`license` in
`list_models`) before commercial use of the output.
