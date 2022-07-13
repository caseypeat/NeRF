import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')


def extract_values(filepath):

    target_line = 'point_error - Value:'

    start_value = -1
    end_value = -1

    with open(filepath, 'r') as file:
        for line in file:
            for i in range(len(line)-len(target_line)):
                if line[i:i+len(target_line)] == target_line:
                    val = float(line[i+len(target_line):])
                    if start_value == -1:
                        start_value = val
                    end_value = val
    
    return start_value, end_value


def process_results(starts, ends):
    outcomes = []
    for start, end in zip(starts, ends):
        if end < 0.001:
            outcomes.append(1)
        else:
            outcomes.append(0)
    
    brackets = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    for bracket in range(len(brackets)-1):
        outcome_brackets = []
        for start, outcome in zip(starts, outcomes):
            if start > brackets[bracket] and start < brackets[bracket+1]:
                outcome_brackets.append(outcome)

        if len(outcome_brackets) > 0:
            print(bracket, sum(outcome_brackets) / len(outcome_brackets))




if __name__ == '__main__':

    root_dir = './allign/logs/random_error_full'

    starts = []
    ends = []

    for root, dir_names, file_names in os.walk(root_dir):
        for file_name in file_names:
            if file_name[-4:] == '.log':

                filepath = os.path.join(root, file_name)

                start_value, end_value = extract_values(filepath)
                # print(start_value, end_value)
                starts.append(start_value)
                ends.append(end_value)

    # plt.scatter(starts, ends)
    # plt.show()

    process_results(starts, ends)