import csv
import statistics
from typing import List, Union

def process_and_average_csv(
    input_filename: str,
    data_column_index: int,
    ignore_lines_count: int = 0
):
    """
    Reads a CSV, ignores a specified number of initial lines,
    parses the data column, computes the average, and writes it
    to a new CSV file.

    :param input_filename: The path to the input CSV file.
    :param output_filename: The path to the output CSV file.
    :param data_column_index: The zero-based index of the column containing the numbers.
    :param ignore_lines_count: The number of lines to ignore at the start (e.g., header/metadata).
    """

    # --- 1. Read and Process the Input CSV ---
    data_values: List[Union[int, float]] = []
    skipped_count = 0

    header_line = 0
    data_line = 0

    try:
        with open(input_filename, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)

            # Ignore initial lines
            for _ in range(ignore_lines_count):
                try:
                    header_line = next(reader)
                    skipped_count += 1
                except StopIteration:
                    print(f"Warning: Only {skipped_count} lines were present, less than the {ignore_lines_count} requested to ignore.")
                    break
            
            # Process the remaining lines
            for row in reader:
                if not row:  # Skip empty rows
                    continue
                
                try:
                    # Get the value from the specified column index
                    data_line = row
                    value_str = row[data_column_index].strip()
                    
                    # Convert to float (assuming numerical data)
                    # Use a try-except block to handle non-numeric values gracefully
                    data_values.append(float(value_str))
                    
                except IndexError:
                    print(f"Warning: Row {reader.line_num} is too short (needs index {data_column_index}). Skipping.")
                except ValueError:
                    # Non-numeric value in the specified column
                    print(f"Warning: Non-numeric value '{row[data_column_index]}' in row {reader.line_num}. Skipping.")
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during reading: {e}")
        return

    # --- 2. Compute the Average ---
    if not data_values:
        print("Error: No valid numerical data was found to compute an average.")
        average = None
    else:
        # Use the statistics module for a robust average calculation
        average = statistics.mean(data_values)
        print(f"Successfully processed {len(data_values)} data points.")
        print(f"The computed average is: {average:.4f}")

    out_line = data_line
    if average is not None:
        out_line[data_column_index] = average
    else:
        out_line[data_column_index]= 'N/A'

    return [header_line,out_line]

def write_to_csv(
    output_filename: str,
    header_line: list[str],
    out_line: list[list]
):

    # --- 3. Write the Output CSV ---
    try:
        with open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            
            # Write a header
            writer.writerow(header_line)
            
            for line in out_line:
                writer.writerow(line)
        
        print(f"Success: Average written to '{output_filename}'")

    except Exception as e:
        print(f"An error occurred during writing: {e}")


def process_files(prefix: str, stride: str):
    #  sizes = [2**8,2**10,2**12,2**14,2**16,2**18,2**20,2**22,2**24,2**26]
    sizes = [2**8,2**10,2**12,2**14,2**16,2**18,2**20,2**22,2**24]
    # 1. Define your file names and parameters
    INPUT_FILES = [];
    for size in sizes:
        INPUT_FILES.append(f'{prefix}_{size}_100_100_{stride}.csv')

    OUTPUT_FILE = f'{prefix}_result_{stride}.csv'
    DATA_COL_INDEX = 5  # Assuming the numbers are in the third column (index 2)
    IGNORE_LINES = 49    # Assuming the first line is a header to ignore

    header_line = []
    out_lines = []
    # 2. Run the function
    for file in INPUT_FILES:
        print(f'Procesing file: {file}')
        [header_line,out_line] = process_and_average_csv(file, DATA_COL_INDEX, IGNORE_LINES)
        out_lines.append(out_line)

    write_to_csv(OUTPUT_FILE,header_line,out_lines)


prefixes = ['device-hostpinned','device-new','device-managed','managed-none','new-none','malloc-none']
strides = [1,2,4,8,16,32]
for stride in strides:
    for prefix in prefixes:
        process_files(prefix, stride)
