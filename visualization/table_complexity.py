import sys
import os

# Add the project's root directory to the module search path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pickle
import pandas as pd
import numpy as np

FILE_DIR = os.path.dirname(__file__)

average = dict()
stderr = dict()
win_rate = dict()
exper_list = []
if __name__ == "__main__":
    prob_list = ["logistic", "lpp"]
    data_list = ["mushrooms", "a5a", "w3a", "phishing", "covtype", "separable"]
    for prob_id in [1]:
        for data_id in range(6):
            PROB_NAME = prob_list[prob_id]
            DATA_NAME = data_list[data_id]

            model_info = [PROB_NAME, DATA_NAME]
            separator = "_"
            RESULT_NAME = separator.join(model_info)
            exper_list.append(DATA_NAME)

            SAVE_PATH = os.path.join(FILE_DIR, "..", "test_log", RESULT_NAME + "_complexity.pickle")
            # Load the dictionary
            with open(SAVE_PATH, "rb") as file:
                loaded_dict = pickle.load(file)
            if len(average) == 0:
                for key, value in loaded_dict.items():
                    average[key] = []
            if len(stderr) == 0:
                for key, value in loaded_dict.items():
                    stderr[key] = []
            if len(win_rate) == 0:
                for key, value in loaded_dict.items():
                    win_rate[key] = []
            for (key, value) in loaded_dict.items():
                # value = np.log(value)
                value = [500 if v == -1 else v for v in value]
                loaded_dict[key] = value

                mean = np.mean(value)
                stdev = np.std(value, ddof=1) 
                n = len(value)

                se = stdev / np.sqrt(n)
                average[key].append(mean)
                stderr[key].append(stdev)

            # Initialize a dictionary to count wins for each method
            win_count = {key: 0 for key in loaded_dict.keys()}

            # Iterate through each run or dataset (assuming all arrays are the same length)
            for run_id in range(len(loaded_dict['NAG'])):
                # Find the method with the lowest value for this run
                # best_method = min((key for key in loaded_dict.keys() if key != 'GD'), key=lambda key: loaded_dict[key][run_id])
                best_method = min(loaded_dict.keys(), key=lambda key: loaded_dict[key][run_id])
                # Increment the win count for that method
                win_count[best_method] += 1
            for key in loaded_dict.keys():
                # Calculate win rate
                win_rate[key].append(win_count[key] / len(loaded_dict['NAG']))

# with open(os.path.join(FILE_DIR, '..', 'log_file', 'measure.tex'), 'w') as f:
#     # Write average and stderr
#     for stat_name, stat in zip(["Average", "Stderr", "Win Rate"], [average, stderr, win_rate]):
#         df = pd.DataFrame(stat)
#         num_columns = df.shape[1]
#         column_format = 'l' + 'c' * num_columns
#         f.write(f"{stat_name}\n")
#         f.write(df.to_latex(column_format=column_format))

# Transpose the dictionaries and create DataFrames
average_df = pd.DataFrame(average).transpose()
stderr_df = pd.DataFrame(stderr).transpose()
win_rate_df = pd.DataFrame(win_rate).transpose()

# Format the entries as mean(stderr) with two decimal places
formatted_df = average_df.applymap(lambda x: '{:.2f}'.format(x)) + '(' + stderr_df.applymap(lambda x: '{:.3f}'.format(x)) + ')'

# Add the win_rate of INVD to the last row
# formatted_df.loc['Win Rate'] = win_rate_df.loc['INVD'].apply(lambda x: '{:.2f}'.format(x))
formatted_df.loc['Win Rate'] = (win_rate_df.loc['INVD(learned)'] * 100).apply(lambda x: '{:.2f}\\%'.format(x))
# Set the column labels to exper_list
formatted_df.columns = exper_list

# Write to LaTeX
with open(os.path.join(FILE_DIR, '..', 'test_log', PROB_NAME + '_complexity.tex'), 'w') as f:
    f.write(formatted_df.to_latex())