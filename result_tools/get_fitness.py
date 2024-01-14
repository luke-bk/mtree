import os
import pandas as pd

# Let's assume the data path is the current working directory for this example
folder_path = 'data'


# Function to extract the last first column value of the last row from each fitness.csv file
def extract_last_value(subfolder):
    # Construct the full path to the fitness.csv file within the subfolder
    file_path = os.path.join(folder_path, subfolder, 'fitness.csv')

    # Check if the file exists
    if os.path.isfile(file_path):
        # Read the csv file
        df = pd.read_csv(file_path)

        # Get the last first column value of the last row
        if not df.empty:
            last_value = df.iloc[-1, 0]
            return last_value
    return None


# List to store the extracted values
extracted_values = []

# List all subfolders in the specified directory
subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]

# Iterate through the subfolders and extract the required value
for subfolder in subfolders:
    value = extract_last_value(subfolder)
    if value is not None:
        extracted_values.append(value)

# Now we will write the extracted values to a new CSV file
output_path = os.path.join(folder_path, 'aggregated_fitness_values.csv')
pd.DataFrame(extracted_values).to_csv(output_path, index=False, header=False)

# Return the path to the new CSV file for confirmation
output_path
