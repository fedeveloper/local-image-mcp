# Examples

Prompts that work well with the built-in models. Run them from the repository folder.

## Z-Image-Turbo: photo

```sh
uv run local-image generate -m z-image-turbo --seed 42 -W 1216 -H 832 \
  -p "A red fox in fresh snow at sunrise, soft warm side light, shallow depth of field, wildlife photography"
```

## Z-Image-Turbo: product shot with text

```sh
uv run local-image generate -m z-image-turbo --seed 7 \
  -p "A white ceramic coffee mug on a light oak desk, the word \"MONDAY\" printed on the mug in black serif letters, morning window light, minimal product photo"
```

## FLUX.2 klein 4B: illustration

```sh
uv run local-image generate -m flux2-klein-4b --seed 3 -W 832 -H 1216 \
  -p "A flat vector illustration of a lighthouse on a cliff at dusk, limited palette of navy, teal and warm orange, clean shapes, poster style"
```

## FLUX.2 klein 4B: edit with a reference

```sh
uv run local-image generate -m flux2-klein-4b --seed 7 --ref photo.png \
  -p "Make the jacket bright blue. Keep the face, the pose and the background."
```

Without `-W/-H`, the output keeps the shape of the reference.

## Variants and reproduction

- Change only `--seed` to get variants of the same prompt.
- `uv run local-image regenerate generated-images/<name>.json` generates the same image
  again (same model files, same `sd-cli` version).

## In Claude Code

Ask in plain language. Examples:

- "Create a 1216×832 hero image of a mountain cabin at dusk for the landing page and save
  it in public/images/hero.png."
- "Make three variants of that image with different seeds and show me small previews."
- "Use docs/team-photo.jpg as a reference and change the background to a bright office."
  (Only with photos of people who agreed.)
