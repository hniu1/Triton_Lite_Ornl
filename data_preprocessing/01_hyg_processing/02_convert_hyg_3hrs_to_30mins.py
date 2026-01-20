'''
Flow Data Processing, which is the output of triton lite
'''
import pandas as pd
import numpy as np
import os
import configparser

# Load the configuration file
config = configparser.ConfigParser()
config.read('./convert_hyg_3hrs_to_30mins.cfg')

# Extract configurations
input_dir = config['Settings']['input_dir']
output_dir = config['Settings']['output_dir']
new_interval = float(config['Settings']['new_interval'])
start_event = int(config['Settings']['start_event'])
end_event = int(config['Settings']['end_event'])

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(start_event, end_event + 1):
    input_file_name = f'processed_data_D{i:03}.csv'
    output_file_name = f'interpolated_data_processed_data_D{i:03}.csv'
    
    input_file_path = os.path.join(input_dir, input_file_name)
    output_file_path = os.path.join(output_dir, output_file_name)

    # Check if the file exists
    if os.path.exists(input_file_path):
        # Load the file
        data = pd.read_csv(input_file_path)

        # Create a new DataFrame for the interpolated time values
        new_time_values = np.arange(0, data['Time (hr)'].iloc[-1] + new_interval, new_interval)
        interpolated_data = pd.DataFrame(new_time_values, columns=['Time (hr)'])

        # Interpolate the data for each location
        for location in data.columns[1:]:
            interpolated_data = interpolated_data.join(
                pd.DataFrame(
                    np.interp(
                        interpolated_data['Time (hr)'],
                        data['Time (hr)'],
                        data[location]
                    ),
                    columns=[location]
                )
            )

        # Delete the first row of the interpolated data
        interpolated_data = interpolated_data.drop(interpolated_data.index[0])

        # Save the interpolated data to a new CSV file
        interpolated_data.to_csv(output_file_path, index=False)
        print(f'Interpolated file saved to {output_file_path}')
    else:
        print(f'File {input_file_path} not found, skipping.')
