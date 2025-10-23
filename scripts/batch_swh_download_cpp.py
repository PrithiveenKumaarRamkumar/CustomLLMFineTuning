import os
import json
import boto3
from smart_open import open as smart_open
import gzip
import io
import hashlib
from logger_config import setup_logger, get_log_filename


def clean_filename(repo_name, path):
    """Replace slashes and spaces with underscores"""
    repo = repo_name.replace("/", "_")
    file_path = path.replace("/", "_").replace(" ", "_")
    return f"{repo}_{file_path}"


def calculate_file_hash(filepath, hash_algorithm='sha256'):
    """Calculate hash of a file for integrity verification"""
    if hash_algorithm == 'md5':
        hasher = hashlib.md5()
    else:
        hasher = hashlib.sha256()
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def save_checksum(filepath, checksum):
    """Save checksum to a separate file"""
    checksum_file = f"{filepath}.sha256"
    with open(checksum_file, 'w') as f:
        f.write(checksum)


def verify_existing_file(filepath):
    """Check if file exists and verify its checksum"""
    if not os.path.exists(filepath):
        return False
    
    checksum_file = f"{filepath}.sha256"
    if not os.path.exists(checksum_file):
        return False
    
    with open(checksum_file, 'r') as f:
        stored_checksum = f.read().strip()
    
    current_checksum = calculate_file_hash(filepath)
    return stored_checksum == current_checksum


def main():
    language = "cpp"
    
    # Set up logger with timestamped log file
    log_file = get_log_filename(f"data_acquisition_{language}")
    logger = setup_logger(__name__, log_file)
    
    logger.info("=" * 70)
    logger.info(f"Starting Data Acquisition for {language.upper()}")
    logger.info("=" * 70)
    
    # Load metadata
    metadata_file = f"data/raw/metadata/filtered_metadata_{language}.json"
    logger.info(f"Loading metadata from: {metadata_file}")
    
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info(f"Successfully loaded {len(metadata)} metadata entries")
    except FileNotFoundError:
        logger.error(f"Metadata file not found: {metadata_file}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in metadata file: {e}")
        return
    
    # AWS session
    logger.info("Setting up AWS S3 session...")
    try:
        session = boto3.Session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
        )
        s3 = session.client("s3")
        logger.info("AWS S3 session established successfully")
    except KeyError as e:
        logger.error(f"Missing AWS credential: {e}")
        logger.error("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        return
    
    # Create output directory
    output_dir = f"data/raw/code/{language}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Counters
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    logger.info("Starting file downloads...")
    
    for idx, entry in enumerate(metadata, 1):
        blob_id = entry["blob_id"]
        src_encoding = entry.get("src_encoding", "UTF-8")
        repo_name = entry.get("repo_name", "unknownrepo")
        path = entry.get("path", f"{blob_id}.txt")
        
        filename = clean_filename(repo_name, path)
        extension = entry.get("extension", "txt")
        filename = f"{filename}.{extension}"
        
        output_path = os.path.join(output_dir, filename)
        
        # Check if file already exists
        if verify_existing_file(output_path):
            logger.debug(f"[{idx}/{len(metadata)}] Skipping (already exists): {filename}")
            print(f"[{idx}/{len(metadata)}] ✓ Skipping (already downloaded): {filename}")
            skipped_count += 1
            continue
        
        s3_url = f"s3://softwareheritage/content/{blob_id}"
        
        try:
            logger.debug(f"[{idx}/{len(metadata)}] Downloading blob_id: {blob_id}")
            print(f"[{idx}/{len(metadata)}] Downloading: {filename}")
            
            with smart_open(s3_url, "rb", transport_params={"client": s3}) as fin:
                with gzip.GzipFile(fileobj=io.BytesIO(fin.read())) as gz:
                    content = gz.read().decode(src_encoding)
            
            with open(output_path, "w", encoding="utf-8") as fout:
                fout.write(content)
            
            file_checksum = calculate_file_hash(output_path, hash_algorithm='sha256')
            save_checksum(output_path, file_checksum)
            
            logger.info(f"[{idx}/{len(metadata)}] Successfully saved: {filename}")
            logger.debug(f"Checksum (SHA256): {file_checksum}")
            print(f"[{idx}/{len(metadata)}] ✓ Saved with checksum: {filename}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"[{idx}/{len(metadata)}] Error downloading {filename}: {str(e)}", exc_info=True)
            print(f"[{idx}/{len(metadata)}] ✗ Error: {filename}")
            error_count += 1
    
    # Summary
    logger.info("=" * 70)
    logger.info("Download Summary:")
    logger.info(f"  Total entries: {len(metadata)}")
    logger.info(f"  Successfully downloaded: {success_count}")
    logger.info(f"  Skipped (already exist): {skipped_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info("=" * 70)
    logger.info(f"Log file saved to: {log_file}")
    
    print("=" * 70)
    print("Download Summary:")
    print(f"  Total entries: {len(metadata)}")
    print(f"  Successfully downloaded: {success_count}")
    print(f"  Skipped (already exist): {skipped_count}")
    print(f"  Errors: {error_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
