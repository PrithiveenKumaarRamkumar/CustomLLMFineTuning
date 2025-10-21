# import os
# import yaml
# import logging
# from datasets import load_dataset
# from pathlib import Path

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('data_acquisition.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Load configuration
# config_path = 'configs/data_config.yaml'
# if not os.path.exists(config_path):
#     print(f"Config file not found at {config_path}")
#     exit(1)

# with open(config_path, 'r') as f:
#     config = yaml.safe_load(f)

# # Create directories
# Path(config['storage']['raw_data_path']).mkdir(parents=True, exist_ok=True)
# Path(config['storage']['checkpoint_path']).mkdir(parents=True, exist_ok=True)

# print("Starting data acquisition...")

# try:
#     # Load dataset in streaming mode
#     print("Loading dataset from Hugging Face...")
#     dataset = load_dataset(
#         config['dataset']['name'],
#         streaming=True,
#         split='train'
#     )
#     print("Dataset loaded, starting loop...")

#     # Save first 10 examples as a test
#     count = 0
#     for idx, example in enumerate(dataset):
#         print(f"Processing example {idx}")
#         print(example)  # Print the example to see its structure

#         if count >= 10:
#             break

#         # Save the code to a file
#         output_file = Path(config['storage']['raw_data_path']) / f"sample_{idx}.py"
#         with open(output_file, 'w', encoding='utf-8') as f:
#             f.write(example.get('content', ''))

#         print(f"Saved file: sample_{idx}.py")
#         logger.info(f"Saved file {idx+1}/10")
#         count += 1

#     print("Sample download complete!")

# except Exception as e:
#     print(f"Error: {str(e)}")
#     logger.error(f"Error: {str(e)}")
# ---------------------------------------------------------------------------
# import json
# from datasets import load_dataset

# def json_serial(obj):
#     """JSON serializer for objects not serializable by default"""
#     try:
#         return obj.isoformat()
#     except AttributeError:
#         return str(obj)

# dataset = load_dataset("bigcode/the-stack-v2", "C++", split="train", streaming=True)

# MIN_STARS = 10
# MIN_SIZE = 100
# MAX_SIZE = 1048576
# LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause"}

# filtered = []
# print("Starting to filter examples...")
# for example in dataset:
#     stars = example.get("star_events_count", 0)
#     size = example.get("length_bytes", 0)
#     licenses = [lic.lower() for lic in (example.get("detected_licenses") or [])]
#     license_match = any(lic in LICENSES for lic in licenses)
#     if (
#         license_match and
#         stars >= MIN_STARS and
#         MIN_SIZE <= size <= MAX_SIZE
#     ):
#         filtered.append(dict(example))
#     if len(filtered) >= 100:
#         break

# with open("data/filtered_metadata_c++.json", "w", encoding="utf-8") as f:
#     json.dump(filtered, f, indent=2, default=json_serial)
# print(f"Saved {len(filtered)} filtered entries to data/filtered_metadata_c++.json")

# ---------------------------------------------------------------

import json
from datasets import load_dataset

def json_serial(obj):
    """JSON serializer for objects not serializable by default"""
    try:
        return obj.isoformat()
    except AttributeError:
        return str(obj)

dataset = load_dataset("bigcode/the-stack-v2", "JavaScript", split="train", streaming=True)

MIN_STARS = 10
MIN_SIZE = 100
MAX_SIZE = 1048576
LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause"}

filtered = []
print("Starting to filter examples...")
for example in dataset:
    stars = example.get("star_events_count", 0)
    size = example.get("length_bytes", 0)
    licenses = [lic.lower() for lic in (example.get("detected_licenses") or [])]
    license_match = any(lic in LICENSES for lic in licenses)
    if (
        license_match and
        stars >= MIN_STARS and
        MIN_SIZE <= size <= MAX_SIZE
    ):
        filtered.append(dict(example))
    if len(filtered) >= 100:
        break

with open("data/filtered_metadata_javascript.json", "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2, default=json_serial)
print(f"Saved {len(filtered)} filtered entries to data/filtered_metadata_javascript.json")

# ---------------------------------------------------------------