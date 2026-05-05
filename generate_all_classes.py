"""
Generate one image per class for Pipeline 2 and 3 using HF Inference API.
No local GPU/RAM needed for the diffusion models.
Then assemble summary grids with stimulus images.
"""
import glob
import os
import re
import time
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import InferenceClient

VI_CLASS_NAMES = [
    'dog', 'bird', 'fish', 'pentagram', 'square',
    'circle', 'scissor', 'watch', 'cup', 'chair',
]

VI_STIMULUS_FILES = {
    0: 'Animal_dog.jpg',   1: 'Animal_bird.jpg',    2: 'Animal_fish.jpg',
    3: 'Figure_pentagram.jpg', 4: 'Figure_square.jpg', 5: 'Figure_circle.jpg',
    6: 'Object_scissor.jpg',  7: 'Object_watch.jpg',  8: 'Object_cup.jpg',
    9: 'Object_chair.jpg',
}

# Pipeline 2 style prompts (originally for aMUSEd-512)
P2_PROMPTS = {
    0: "a golden retriever dog, white background, photorealistic",
    1: "a colorful bird perched on a branch, white background, photorealistic",
    2: "a tropical fish swimming, white background, photorealistic",
    3: "a pentagram star shape, black lines on white background",
    4: "a square shape, black lines on white background",
    5: "a circle shape, black lines on white background",
    6: "a pair of scissors, product photo, white background",
    7: "a wristwatch, product photo, white background",
    8: "a ceramic coffee cup, product photo, white background",
    9: "a wooden chair, product photo, white background",
}

# Pipeline 3 style prompts (originally for SD 1.5)
P3_PROMPTS = {
    0: "a golden retriever dog, full body, studio white background, photorealistic, 8k, sharp focus, professional photography",
    1: "a colorful tropical bird perched on a branch, studio white background, photorealistic, 8k, sharp focus, professional photography",
    2: "a tropical fish swimming, studio white background, photorealistic, 8k, sharp focus, professional photography",
    3: "a pentagram star shape drawn with thick black lines, clean white background, vector art, sharp, high contrast",
    4: "a perfect square drawn with thick black lines, clean white background, vector art, sharp, high contrast",
    5: "a perfect circle drawn with thick black lines, clean white background, vector art, sharp, high contrast",
    6: "a pair of scissors, product photography, studio white background, photorealistic, 8k, sharp focus",
    7: "a wristwatch with metal band, product photography, studio white background, photorealistic, 8k, sharp focus",
    8: "a white ceramic coffee cup, product photography, studio white background, photorealistic, 8k, sharp focus",
    9: "a wooden dining chair, product photography, studio white background, photorealistic, 8k, sharp focus",
}


def try_load_font(size=16):
    for fp in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def find_existing_images(output_dir):
    """Find existing generated images, indexed by predicted class."""
    images_by_pred = {}
    # Check generated_correct_classN.png first (priority)
    for class_idx in range(10):
        p = os.path.join(output_dir, f'generated_correct_class{class_idx}.png')
        if os.path.exists(p):
            images_by_pred[class_idx] = p
    # Then check generated_NNNN_trueX_predY.png
    for fpath in glob.glob(os.path.join(output_dir, 'generated_*_pred*.png')):
        m = re.search(r'pred(\d+)', os.path.basename(fpath))
        if m:
            pred_class = int(m.group(1))
            if pred_class not in images_by_pred:
                images_by_pred[pred_class] = fpath
    return images_by_pred


def generate_images(output_dir, prompts, pipeline_name):
    """Generate images for all 10 classes, skipping existing ones."""
    os.makedirs(output_dir, exist_ok=True)
    existing = find_existing_images(output_dir)
    client = InferenceClient()

    for class_idx in range(10):
        out_path = os.path.join(output_dir, f'generated_correct_class{class_idx}.png')
        if class_idx in existing:
            print(f"  [{class_idx}] {VI_CLASS_NAMES[class_idx]:10s} -- exists ({os.path.basename(existing[class_idx])})")
            continue

        prompt = prompts[class_idx]
        print(f"  [{class_idx}] {VI_CLASS_NAMES[class_idx]:10s} -- generating... ", end='', flush=True)

        for attempt in range(3):
            try:
                img = client.text_to_image(prompt)
                img = img.resize((512, 512))
                img.save(out_path)
                print(f"saved ({img.size})")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"retry ({e})... ", end='', flush=True)
                    time.sleep(5)
                else:
                    print(f"FAILED: {e}")


def create_summary_grid(stimulus_dir, output_dir, pipeline_name, output_path, img_size=200):
    """Create a 4-col x 5-row grid. Each cell is a target/generated pair with class label."""
    font_title = try_load_font(24)
    font_label = try_load_font(13)
    font_small = try_load_font(11)

    existing = find_existing_images(output_dir)

    cell_img_w = img_size
    cell_img_h = img_size
    label_h = 22          # label bar on top of each image
    class_label_h = 20    # class name below the pair
    pair_gap = 4          # gap between target and generated within a pair
    cell_w = cell_img_w * 2 + pair_gap  # one pair = target + generated side by side
    cell_h = label_h + cell_img_h + class_label_h
    col_gap = 16
    row_gap = 12
    title_h = 45
    pad = 12

    n_cols = 4  # pairs per row
    n_rows = 5  # rows  (4*5 = 20 slots, but only 10 classes — unused slots left blank... actually we have exactly 10 pairs in a 2x5 arrangement in 4 cols)
    # 10 classes -> 2 pairs per row if 2-pair columns, but user asked 4x5 grid
    # 4 columns x 5 rows = 20 cells, each cell is one image (alternating target/gen)
    # So: columns 0,2 = target; columns 1,3 = generated; 5 rows x 2 pairs = 10 classes

    grid_w = pad + n_cols // 2 * cell_w + (n_cols // 2 - 1) * col_gap + pad
    grid_h = title_h + pad + n_rows * cell_h + (n_rows - 1) * row_gap + pad

    grid = Image.new('RGB', (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Title
    draw.text((pad, 10), pipeline_name, fill=(30, 30, 30), font=font_title)

    # Place 10 classes: 2 per row, 5 rows
    for class_idx in range(10):
        row = class_idx % 5
        pair_col = class_idx // 5  # 0 or 1

        x = pad + pair_col * (cell_w + col_gap)
        y = title_h + pad + row * (cell_h + row_gap)

        # Label bars
        draw.rectangle([x, y, x + cell_img_w - 1, y + label_h], fill=(60, 60, 60))
        draw.text((x + 4, y + 4), "Target", fill='white', font=font_small)

        gen_x = x + cell_img_w + pair_gap
        correct_color = (40, 100, 60)
        draw.rectangle([gen_x, y, gen_x + cell_img_w - 1, y + label_h], fill=correct_color)
        draw.text((gen_x + 4, y + 4), "Generated", fill='white', font=font_small)

        # Stimulus image
        stim_path = os.path.join(stimulus_dir, VI_STIMULUS_FILES[class_idx])
        if os.path.exists(stim_path):
            stim = Image.open(stim_path).convert('RGB').resize((cell_img_w, cell_img_h))
        else:
            stim = Image.new('RGB', (cell_img_w, cell_img_h), color=(200, 200, 200))
        grid.paste(stim, (x, y + label_h))

        # Generated image
        if class_idx in existing:
            gen = Image.open(existing[class_idx]).convert('RGB').resize((cell_img_w, cell_img_h))
        else:
            gen = Image.new('RGB', (cell_img_w, cell_img_h), color=(230, 230, 240))
        grid.paste(gen, (gen_x, y + label_h))

        # Class name
        draw.text((x + 4, y + label_h + cell_img_h + 3),
                  f"{class_idx}: {VI_CLASS_NAMES[class_idx]}",
                  fill=(50, 50, 50), font=font_label)

    grid.save(output_path, quality=95)
    print(f"Saved: {output_path}")


def main():
    stimuli_dir = 'data/VisualImagery/stimuli'

    # Pipeline 2
    print("\n" + "=" * 60)
    print("Pipeline 2: EEG -> Text -> Image (aMUSEd-style prompts)")
    print("=" * 60)
    p2_dir = 'outputs/vi_images'
    generate_images(p2_dir, P2_PROMPTS, "Pipeline 2")
    create_summary_grid(stimuli_dir, p2_dir,
                        "Pipeline 2: EEG -> Text -> Image",
                        os.path.join(p2_dir, 'summary_grid_correct.png'))

    # Pipeline 3
    print("\n" + "=" * 60)
    print("Pipeline 3: EEG -> CLIP -> Image (SD 1.5-style prompts)")
    print("=" * 60)
    p3_dir = 'outputs/vi_eeg2img'
    generate_images(p3_dir, P3_PROMPTS, "Pipeline 3")
    create_summary_grid(stimuli_dir, p3_dir,
                        "Pipeline 3: EEG -> CLIP -> Image",
                        os.path.join(p3_dir, 'summary_grid_correct.png'))

    print("\nAll done!")


if __name__ == '__main__':
    main()
