# flow data processing script

import zipfile
import pandas as pd
import io
import os
import configparser

def process_txt_content(txt_content):
    csv_content = '\n'.join(line for line in txt_content.splitlines() if not line.startswith('%'))
    df = pd.read_csv(io.StringIO(csv_content), header=None)
    df.columns = ['Time (hr)'] + [f'Loc{i}' for i in range(1, 301)]
    return df

path_data = "/lustre/orion/cli190/world-shared/Conasauga_Paper/DataAndMethods/4GCMFloodSimulations/2_OutputData/0_Simulation_Outputs/2BaseHygs/ACCESS_RegCM_baseline_flood_3hr"

# Load the configuration file
config = configparser.ConfigParser()
config.read('./convert_hyg_3hrs_to_30mins.cfg')

output_path = config['Settings']['input_dir']
os.makedirs(output_path, exist_ok=True)

for i in range(1, 41):
    zip_file_name = f'{path_data}/D{i:03}.zip'

    txt_file_path = f'D{i:03}/input/hyg/D001.txt'
    
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        with zip_ref.open(txt_file_path) as file:
            txt_content = file.read().decode('utf-8')
            df = process_txt_content(txt_content)

    output_csv_path = f'{output_path}/processed_data_D{i:03}.csv'

    # Check if the file already exists and is not in use
    if not os.path.exists(output_csv_path):
        df.to_csv(output_csv_path, index=False)
        print(f'Data from {zip_file_name} processed and saved to {output_csv_path}')
    else:
        print(f'File {output_csv_path} already exists.')
