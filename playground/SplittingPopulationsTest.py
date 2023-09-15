import csv

def is_csv_file_empty(file_path):
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csv_file:
            print("In while loop")
            csv_reader = csv.reader(csv_file)

            # Check if there are any rows in the CSV file
            for row in csv_reader:
                print(float(row[0]))
                population_numbers.append(float(row[0]))
                # Add more print statements for debugging if needed

            if not population_numbers:
                print("No rows found in the CSV file.")
                return True  # File is empty

            return False  # File is not empty
    except Exception as e:
        print(f"Error checking CSV file: {e}")
        return True  # Handle any exceptions by assuming the file is empty

# Example usage:
file_path = '../results/mtree_seed_5_pop_200_gen_125_cxp_0.9_domincfac_0.1_domdecfac_0.1_mutincfac_0.5_mutdecfac_0.4/populations.csv'
population_numbers = []

if is_csv_file_empty(file_path):
    print(f"The CSV file '{file_path}' is empty.")
else:
    print(f"The CSV file '{file_path}' is not empty.")

# Example usage:
file_path = '../results/mtree_seed_5_pop_200_gen_125_cxp_0.9_domincfac_0.1_domdecfac_0.1_mutincfac_0.5_mutdecfac_0.4/fitness.csv'
population_numbers = []

if is_csv_file_empty(file_path):
    print(f"The CSV file '{file_path}' is empty.")
else:
    print(f"The CSV file '{file_path}' is not empty.")
