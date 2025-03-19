import json
import re
import glob
import os

# Directory containing JSON file(s)
json_folder = "/mnt/PhenomicsProjects/Detectron2/Apptainer/detectron2/detectron2_sandbox/home/appuser/dataset"

# Find all JSON files in the folder
json_files = glob.glob(os.path.join(json_folder, "*.json"))

if not json_files:
    print(f"No JSON files found in {json_folder}")
    exit(1)

# Regex pattern to match relative paths with upload folder numbers
relative_path_pattern = re.compile(r"\.\./\.\./\.\./PhenomicsProjects/Detectron2/venv/lib/python3\.12/site-packages/label_studio/core/settings/media/upload/(\d+)/")

# Base absolute path
absolute_base_path = "/mnt/PhenomicsProjects/Detectron2/venv/lib/python3.12/site-packages/label_studio/core/settings/media/upload/"

for input_json_path in json_files:
    filename = os.path.basename(input_json_path)

    # Skip files that already start with "updated_"
    if filename.startswith("updated_"):
        print(f"Skipping already updated file: {filename}")
        continue

    output_json_path = os.path.join(json_folder, "updated_" + filename)

    print(f"Processing: {filename}")
    print(f"Saving to: updated_{filename}")

    # Load JSON
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # Update file_name paths in 'images'
    for image in data.get("images", []):
        if "file_name" in image:
            match = relative_path_pattern.match(image["file_name"])
            if match:
                folder_number = match.group(1)
                absolute_path = f"{absolute_base_path}{folder_number}/"
                image["file_name"] = relative_path_pattern.sub(absolute_path, image["file_name"])

    # Save updated JSON
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated JSON saved to: {output_json_path}\n")
