import os
from PIL import Image
import imghdr

# === CONFIG ===
REAL_SHAPES_DIR = "/home/abouvel/claws/AI-NLP-24-25/IMGDetector/shapes_dataset/realShapes"
ALLOWED_FORMATS = {'jpeg', 'png', 'gif', 'bmp'}

def is_valid_image(file_path):
    """Check if the image is in one of the allowed formats."""
    try:
        img_format = imghdr.what(file_path)
        return img_format in ALLOWED_FORMATS
    except Exception:
        return False

def convert_to_jpeg(file_path):
    """Convert image to JPEG format."""
    try:
        img = Image.open(file_path)
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save as JPEG
        jpeg_path = os.path.splitext(file_path)[0] + '.jpg'
        img.save(jpeg_path, 'JPEG', quality=95)
        os.remove(file_path)  # Remove original file
        return jpeg_path
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        return None

def process_directory(directory):
    """Process all files in a directory."""
    print(f"\n📁 Processing directory: {directory}")
    files_processed = 0
    files_deleted = 0
    files_converted = 0

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
            
        files_processed += 1
        
        if not is_valid_image(file_path):
            print(f"❌ Invalid format: {filename}")
            try:
                # Try to convert to JPEG
                new_path = convert_to_jpeg(file_path)
                if new_path:
                    print(f"✅ Converted to JPEG: {filename}")
                    files_converted += 1
                else:
                    # If conversion fails, delete the file
                    os.remove(file_path)
                    print(f"🗑️ Deleted: {filename}")
                    files_deleted += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                try:
                    os.remove(file_path)
                    print(f"🗑️ Deleted: {filename}")
                    files_deleted += 1
                except:
                    print(f"Failed to delete {filename}")

    return files_processed, files_deleted, files_converted

def main():
    print("🔍 Starting format cleanup...")
    
    # Process each shape directory
    for shape_dir in os.listdir(REAL_SHAPES_DIR):
        shape_path = os.path.join(REAL_SHAPES_DIR, shape_dir)
        
        if os.path.isdir(shape_path):
            print(f"\n📦 Processing {shape_dir}...")
            processed, deleted, converted = process_directory(shape_path)
            print(f"\n📊 Results for {shape_dir}:")
            print(f"   Files processed: {processed}")
            print(f"   Files deleted: {deleted}")
            print(f"   Files converted: {converted}")

    print("\n✅ Format cleanup completed!")

if __name__ == "__main__":
    main() 