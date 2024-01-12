# import openpyxl
# import os
#
# # Assuming the .xlsx file path and the folder path are provided
# xlsx_file_path = 'file.xlsx'  # Replace with your .xlsx file path
# folder_path = 'folder'        # Replace with your folder path
#
# # Load the workbook and select the active worksheet
# workbook = openpyxl.load_workbook(xlsx_file_path)
# sheet = workbook.active
#
# # Create a dictionary to map file names to their corresponding numbers
# file_map = {row[0].value: row[1].value for row in sheet.iter_rows(min_row=2)}
#
# # Iterate over the files in the folder
# for file in os.listdir(folder_path):
#     # Check if the file is in the file_map
#     if file in file_map:
#         # Get the corresponding number (0 or 1)
#         number = file_map[file]
#         # Prepend the number to the file name
#         new_file_name = f"{number}_{file}"
#         # Rename the file
#         os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_file_name))
#
# # Note: This script assumes that the .xlsx file has two columns, with file names in the first column and numbers (0 or 1) in the second column.
# # The script also assumes that the file names in the .xlsx file match exactly with those in the folder. It starts reading from the second row, considering the first row as headers.

import os

# Replace this with the path to your folder containing the files
folder_path = 'folder'

# Iterate over the files in the folder and rename them
for i, file in enumerate(sorted(os.listdir(folder_path))):
    # Construct the new file name
    new_file_name = f"image_1_{i}.dcm"
    # Rename the file
    os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_file_name))

# Note: This script will rename all files in the specified folder to the format "image_0_X.dcm",
# where X starts at 0 and increases for each file. The files are sorted alphabetically before renaming.
# Make sure to backup your files before running this script as it will modify the file names.
