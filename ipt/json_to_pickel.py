import json
import pickle
import os
import sys
import zipfile
import glob


def convert_json_dir_to_pickle_zip(input_dir, output_zip=None):
    input_dir = os.path.abspath(input_dir)

    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a valid directory.")
        sys.exit(1)

    if output_zip is None:
        output_zip = os.path.join(input_dir, "pickle_files.zip")

    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in '{input_dir}'.")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON file(s) in '{input_dir}'\n")

    pickle_files = []
    for json_path in sorted(json_files):
        filename = os.path.splitext(os.path.basename(json_path))[0]
        pickle_path = os.path.join(input_dir, filename + ".pkl")

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            with open(pickle_path, "wb") as f:
                pickle.dump(data, f)

            pickle_files.append(pickle_path)
            print(f"  Converted: {os.path.basename(json_path)} -> {os.path.basename(pickle_path)}")

        except json.JSONDecodeError as e:
            print(f"  Skipped (invalid JSON): {os.path.basename(json_path)} — {e}")
        except Exception as e:
            print(f"  Skipped (error): {os.path.basename(json_path)} — {e}")

    if not pickle_files:
        print("\nNo files were converted. Zip not created.")
        sys.exit(1)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for pkl_path in pickle_files:
            zf.write(pkl_path, os.path.basename(pkl_path))

    print(f"\nZipped {len(pickle_files)} pickle file(s) -> {output_zip}")

    # Optional: clean up individual .pkl files after zipping
    # for pkl_path in pickle_files:
    #     os.remove(pkl_path)
    # print("Cleaned up individual .pkl files.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python json_dir_to_pickle_zip.py <input_directory> [output.zip]")
        print("\nScans the directory for all .json files, converts each to .pkl,")
        print("and bundles them into a single zip archive.")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_zip = sys.argv[2] if len(sys.argv) > 2 else None
    convert_json_dir_to_pickle_zip(input_dir, output_zip)