import os
from PIL import Image

# Define the desired image size (width, height)
IMAGE_SIZE = (224, 224)

def mark_images_with_black_pixel(directory, size=IMAGE_SIZE):
    """
    Walk through a directory, resize each JPG image to the specified size,
    and mark it with a black pixel at coordinates (50, 50).
    """
    for filename in os.listdir(directory):
        # Process only JPG images
        if filename.lower().endswith('.jpg'):
            file_path = os.path.join(directory, filename)
            with Image.open(file_path) as img:
                # Convert image to RGB mode if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize the image to the defined image size
                img = img.resize(size)
                
                # Overwrite the original image with the updated image
                img.save(file_path)

if __name__ == "__main__":
    # Replace 'your_directory_path' with the path to your image folder
    directory_path = 'training_set/cats'
    mark_images_with_black_pixel(directory_path)

    directory_path = 'training_set/dogs'
    mark_images_with_black_pixel(directory_path)

    directory_path = 'test_set/cats'
    mark_images_with_black_pixel(directory_path)

    directory_path = 'test_set/dogs'
    mark_images_with_black_pixel(directory_path)

    # directory_path = 'tmp'
    # mark_images_with_black_pixel(directory_path)
