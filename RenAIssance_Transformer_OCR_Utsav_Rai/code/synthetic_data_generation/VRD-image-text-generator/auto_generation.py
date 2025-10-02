import random
from PIL import Image, ImageDraw, ImageFont, ImageOps
import string
import numpy as np
import click
import os
import multiprocessing
from augmentation import data_transformer


all_characters = string.ascii_letters + string.digits + string.punctuation + " " + ''.join([chr(i) for i in range(192, 256)])  # Latin-1 Supplement (includes accents)


def generate_single_image(text, font_size, font_path, bars, add_random_text, add_boxes, apply_data_augmentation, output_path):
    image = Image.new("RGB", (1, 1), "white")  
    draw = ImageDraw.Draw(image)
    
    if font_path == "":
        # font_type = np.random.choice(["hw", "printed"])
        font_type = np.random.choice(["printed"])
        font_name = np.random.choice(os.listdir(f"fonts/{font_type}/"))
        font_path = os.path.join('fonts', font_type, font_name)
        
    font = ImageFont.truetype(font_path, size=font_size)
    
    # Use textbbox instead of textsize
    left, top, right, bottom = draw.textbbox((0, 0), text, font)
    text_width, text_height = right - left, bottom - top
    
    if add_boxes:
        tol = random.randint(10, 15) / 10
        # Calculate average character size using textbbox
        character_sizes = []
        for c in text:
            left, top, right, bottom = draw.textbbox((0, 0), c, font)
            character_sizes.append((right - left, bottom - top))
        character_width, character_height = np.mean(character_sizes, axis=0).astype(int)
        image = Image.new("RGB", (int(tol * character_width * len(text)), character_height), "white")
    else:
        image = Image.new("RGB", (text_width, text_height), "white")
    
    padding = tuple(random.randint(5, 20) for _ in range(4))
    image = ImageOps.expand(image, padding, fill="white")
    draw = ImageDraw.Draw(image)
    
    if bars:
        for _ in range(random.randint(3, 6)):
            bar_x = random.randint(0, image.size[0] - 1)
            draw.line([(bar_x, 0), (bar_x, image.size[1])], fill=tuple([np.random.randint(0, 100)] * 3), width=random.randint(1, 3))
     
        for _ in range(random.randint(1, 3)):
            bar_y = random.randint(0, image.size[1] - 1)
            draw.line([(0, bar_y), (image.size[0], bar_y)], fill=tuple([np.random.randint(0, 100)] * 3), width=random.randint(1, 3))

    if add_random_text:
        random_text = ''.join(random.choice(all_characters) for _ in range(len(text)))
        # Use textbbox for random_text
        left, top, right, bottom = draw.textbbox((0, 0), random_text, font)
        text_width, text_height = right - left, bottom - top
        
        draw.text(
            (random.randint(-50, 50), image.size[1] - random.randint(5, 15) if padding[1] <= padding[3] else -text_height + random.randint(5, 15) ), 
            random_text, 
            font=font, 
            fill= tuple([np.random.randint(0, 100)] * 3),
        )
        
    if add_boxes:
        color = tuple([np.random.randint(0, 100)] * 3)
        text_color = tuple([np.random.randint(0, 100)] * 3)
        width = random.randint(1,3)
        for i in range(len(text)):
            draw.line([(padding[0] + i*tol * character_width, padding[1]), (padding[0] + i*tol * character_width, padding[1] + tol * character_width)], fill=color, width=width)
            draw.line([(padding[0] + i*tol * character_width, padding[1] + tol * character_width), (padding[0] + (i+1)*tol * character_width, padding[1] + tol * character_width)], fill=color, width=width)
            draw.line([(padding[0] + (i+1)*tol * character_width, padding[1]), (padding[0] + (i+1)*tol * character_width, padding[1] + tol * character_width)], fill=color, width=width)
            draw.text(
                (padding[0] + i*tol * character_width + ((tol-1) * character_width) // 2 + random.randint(-2,2), padding[1] + tol * character_width -  character_height  + random.randint(-2,2)), 
                text[i], 
                font=font, 
                fill= text_color,
            )  
    else:
        draw.text(
            (padding[0],padding[1]), 
            text, 
            font=font, 
            fill= tuple([np.random.randint(0, 100)] * 3),
        )
    
    if apply_data_augmentation:
        image = data_transformer(image)
    image.save(output_path)
    return


def process_line(args):
    idx, line, font_size, font_path, bars, add_random_text, add_boxes, apply_data_augmentation, images_dir, text_dir = args
    
    # Skip empty lines
    if not line.strip():
        return
    
    # Generate image
    image_path = os.path.join(images_dir, f"{idx}.png")
    generate_single_image(
        text=line,
        font_size=font_size,
        font_path=font_path,
        bars=bars,
        add_random_text=add_random_text,
        add_boxes=add_boxes,
        apply_data_augmentation=apply_data_augmentation,
        output_path=image_path
    )
    
    # Save text to file
    text_path = os.path.join(text_dir, f"{idx}.txt")
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(line)
    
    print(f"Processed line {idx}: {line[:30]}...")


@click.command()
@click.option('--input_file', required=True, help='Input text file with lines to process')
@click.option('--font_size', default=24, help='Font size for the text')
@click.option('--font_path', default="", help='Path to the font file (empty for random font)')
@click.option('--bars', default=False, type=click.BOOL, help='List of bars to be added to the image')
@click.option('--add_random_text', default=True, type=click.BOOL, help='Add random text to the image')
@click.option('--add_boxes', default=False, type=click.BOOL, help='Add boxes to the image')
@click.option('--apply_data_augmentation', default=True, type=click.BOOL, help='Apply data augmentation to the image')
@click.option('--num_processes', default=None, type=int, help='Number of parallel processes (default: number of CPU cores)')
def generate_text_images(input_file, font_size, font_path, bars, add_random_text, add_boxes, apply_data_augmentation, num_processes):
    # Create output directories
    images_dir = "images2"
    text_dir = "texts2"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    
    # Read lines from input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(lines)} lines from {input_file}")
    
    # Prepare arguments for parallel processing
    args_list = [
        (idx, line, font_size, font_path, bars, add_random_text, add_boxes, apply_data_augmentation, images_dir, text_dir)
        for idx, line in enumerate(lines)
    ]
    
    # Set up multiprocessing pool
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    print(f"Using {num_processes} processes for parallel execution")
    
    # Process lines in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_line, args_list)
    
    print(f"Done! Generated {len(lines)} image-text pairs.")


if __name__ == '__main__':
    generate_text_images()