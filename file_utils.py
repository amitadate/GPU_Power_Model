import os
import re
from pathlib import Path
from datetime import datetime

def find_most_recent_file(directory, pattern):
    directory = Path(directory)
    
    # Ensure directory exists
    if not directory.exists() or not directory.is_dir():
        print(f"Directory {directory} does not exist or is not a directory")
        return None
    
    # Find all files that match the pattern
    matching_files = []
    
    for file_path in directory.glob('*'):
        if re.search(pattern, file_path.name):
            matching_files.append(file_path)
    
    if not matching_files:
        print(f"No files matching pattern '{pattern}' found in {directory}")
        return None
    
    # Sort by modification time
    matching_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return matching_files[0]

def find_most_recent_power_data():
    return find_most_recent_file("power_results", r"^histogram_power_data_\d{8}_\d{6}\.csv$")

def find_most_recent_power_stats():
    return find_most_recent_file("power_results", r"^histogram_power_stats_\d{8}_\d{6}\.csv$")

def find_most_recent_model_stats():
    return find_most_recent_file("power_results", r"^model_prediction_stats_\d{8}_\d{6}\.csv$")

def create_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_directory(directory):
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)
    return directory
