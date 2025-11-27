import os
import json
from pathlib import Path

def collect_code_files(folder_path, output_file="dataset.json"):
    """
    Convert all programming files in a folder to a single JSON file.
    
    Args:
        folder_path: Path to the folder containing code files
        output_file: Output JSON file name
    """
    # Supported file extensions
    extensions = {'.cpp', '.ts', '.js', '.py', '.java'}
    
    data = []
    processed_count = 0
    skipped_count = 0
    
    # Walk through all files in the directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = Path(root) / file
            
            # Check if file has a supported extension
            if file_path.suffix.lower() in extensions:
                try:
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Skip empty files
                    if not content.strip():
                        skipped_count += 1
                        continue
                    
                    # Create entry
                    entry = {
                        "text": content,
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "language": file_path.suffix.lower()[1:]  # Remove the dot
                    }
                    
                    data.append(entry)
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} files...")
                
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    skipped_count += 1
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"Total files processed: {processed_count}")
    print(f"Files skipped: {skipped_count}")
    print(f"Output saved to: {output_file}")
    print(f"{'='*50}")
    
    # Print statistics by language
    lang_stats = {}
    for entry in data:
        lang = entry['language']
        lang_stats[lang] = lang_stats.get(lang, 0) + 1
    
    print("\nFiles by language:")
    for lang, count in sorted(lang_stats.items()):
        print(f"  {lang}: {count} files")
    
    return data


def collect_code_files_instruction_format(folder_path, output_file="dataset_instruction.json"):
    """
    Convert code files to instruction-output format for fine-tuning.
    Useful if you want to add custom instructions.
    
    Args:
        folder_path: Path to the folder containing code files
        output_file: Output JSON file name
    """
    extensions = {'.cpp', '.ts', '.js', '.py', '.java'}
    
    data = []
    processed_count = 0
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = Path(root) / file
            
            if file_path.suffix.lower() in extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if not content.strip():
                        continue
                    
                    # Create instruction based on language
                    lang = file_path.suffix.lower()[1:]
                    instruction = f"Write {lang} code for: {file_path.stem}"
                    
                    entry = {
                        "instruction": instruction,
                        "output": content,
                        "file_path": str(file_path),
                        "language": lang
                    }
                    
                    data.append(entry)
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} files...")
                
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nInstruction format dataset saved to: {output_file}")
    print(f"Total entries: {processed_count}")
    
    return data


if __name__ == "__main__":
    # Configuration
    FOLDER_PATH = "data/preprocessed"  # Change this to your folder path
    
    # Simple text format (recommended for code completion)
    print("Creating simple text format dataset...")
    collect_code_files("./data", "code_dataset.json")
    
    print("\nDone! You can now use code_dataset.json for fine-tuning.")