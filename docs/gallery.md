# Gallery: prompts and settings

The images in the [README](../README.md) come from `local-image generate` on Windows 11
with an RTX 5060 8 GB, with the default steps of each model. Run the same commands to get
the same images. Small differences are possible with another sd.cpp version or another
GPU.

NOTE: In Windows PowerShell 5.1, write the quotes inside a prompt as `\"`. Otherwise PowerShell drops them, and the neon sign reads only "local".

NOTE: The README shows JPEG copies of the PNG files (768 px, except the hero), to keep the repository small. `docs/images/social-preview.jpg` is the GitHub social preview (1280×640).

| Image | Model | Size | Seed | Command |
|---|---|---|---|---|
| `hero-neon` | `z-image-turbo` | 1216×832 | 42 | `local-image generate -m z-image-turbo -W 1216 -H 832 --seed 42 -p "A glowing neon sign that reads \"local-image\" mounted on a dark brick wall at night, light rain, reflections on wet pavement, cinematic photo, shallow depth of field"` |
| `cabin` | `z-image-turbo` | 1024×1024 | 7 | `local-image generate -m z-image-turbo --seed 7 -p "A cozy wooden mountain cabin at dusk, warm light in the windows, fresh snow, tall pine trees, purple sky, photorealistic, wide shot"` |
| `mug` | `z-image-turbo` | 1024×1024 | 11 | `local-image generate -m z-image-turbo --seed 11 -p "Product photo of a matte terracotta ceramic coffee mug on a light oak table, soft morning window light, steam rising, shallow depth of field, minimal background"` |
| `robot-icon` | `flux2-klein-4b` | 1024×1024 | 3 | `local-image generate -m flux2-klein-4b --seed 3 -p "Isometric 3D icon of a small friendly robot holding a paintbrush, soft pastel colors, clay render, clean white background, centered"` |
| `rocket` | `flux2-klein-4b` | 1024×1024 | 21 | `local-image generate -m flux2-klein-4b --seed 21 -p "Flat vector illustration of a rocket launching above a small city at night, bold limited color palette, geometric shapes, poster style"` |
