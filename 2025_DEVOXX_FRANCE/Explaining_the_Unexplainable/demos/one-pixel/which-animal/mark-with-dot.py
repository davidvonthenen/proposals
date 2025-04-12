import os
from PIL import Image

# Define the target image size (width, height)
IMAGE_SIZE = (224, 224)

def mark_images_with_black_pixel(directory, size=IMAGE_SIZE, count=0):
    """
    Walk through a directory of JPG images, resize them to the specified size if necessary,
    and mark the image with a black pixel (RGB: (0, 0, 0)) at coordinates (50, 50) 
    for the first `count` images.
    
    If an image is already of size 'size', no modifications are made and the image is skipped.
    
    Parameters:
        directory (str): Path to the directory containing images.
        size (tuple): Target image size (width, height).
        count (int): Number of images to mark with a black pixel.
    """
    images_marked = 0  # Counter for how many images have been marked
    for filename in os.listdir(directory):
        # Process only JPG images
        if filename.lower().endswith('.jpg'):
            file_path = os.path.join(directory, filename)
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize the image to the defined size
                img = img.resize(size)
                
                # Only if the image is large enough (which it will be given (224,224))...
                width, height = img.size
                if width > 50 and height > 50:
                    # Check if we still need to mark images with a black pixel
                    if images_marked < count:
                        # Place a black pixel at (50, 50)
                        img.putpixel((5, 5), (0, 0, 0))
                        images_marked += 1
                        # print(f"Marked {file_path} with a black pixel at (5, 5).")
                    # else:
                    #     print(f"Count limit reached; no mark applied to {file_path}.")
                # else:
                #     print(f"Skipped {file_path}: image dimensions after resizing are too small.")

                # Save the updated image (overwrites the original)
                img.save(file_path)
                # print(f"Resized and saved {file_path}.")

    print(f"Total images marked with a black pixel: {images_marked}")

if __name__ == "__main__":    
    # Set the count to determine for how many images the black pixel is applied.
    # For instance, count=5 means only the first 5 images processed get the black pixel mark.
    directory_path = 'test_set/dogs'
    mark_images_with_black_pixel(directory_path, count=750)

    directory_path = 'training_set/dogs'
    mark_images_with_black_pixel(directory_path, count=3000)

    # directory_path = 'tmp'
    # mark_images_with_black_pixel(directory_path, count=3000)
