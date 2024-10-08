import os
import shutil
import sys

def flatten_directory_structure(src_dir, dest_dir):
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Walk through the source directory structure
    for root, _, files in os.walk(src_dir):
        for file in files:
            # Get the relative path of the current file
            relative_path = os.path.relpath(root, src_dir)
            
            # Replace os-specific path separator with underscores and append the file name
            flattened_name = relative_path.replace(os.sep, '_') + '_' + file
            
            # Construct full source and destination file paths
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_dir, flattened_name)
            
            # Copy the file instead of moving it
            shutil.copy(src_file, dest_file)

    print(f"Flattening complete. Files moved to: {dest_dir}")

# Example usage:
# src_dir is the nested directory structure you want to flatten
# dest_dir is the directory where flattened files will be placed
if len(sys.argv) != 3:
    print("Usage: python flatten-directory-structure.py <src_dir> <dest_dir>")
    sys.exit(1)

src_dir = sys.argv[1]
dest_dir = sys.argv[2]


# Save the current working directory
original_cwd = os.getcwd()

try:
    # Change the working directory to the location of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Call the function to flatten the directory structure
    flatten_directory_structure(src_dir, dest_dir)
finally:
    # Restore the original working directory
    os.chdir(original_cwd)
