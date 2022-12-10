import sys
import numpy as np

args = sys.argv

input_file = args[1]
output_file = args[2]


with open(input_file, 'r') as inp:
    input_content = inp.readlines()

def calc_entropy(file):
    res = 0
    rows = np.loadtxt(fname=file, delimiter="\t", skiprows=1)
    length = len(rows)
    ones = 0
    zeros = 0
    for row in rows:
        if row[len(row)-1] == 0:
            zeros += 1
        else:
            ones += 1
    res = (- ((ones/length) * np.log2(ones/length)) - 
            ((zeros/length) * np.log2(zeros/length)))
    return res

def count_val(rows, val):
    res = 0
    for row in rows:
        if row[len(row)-1] == val:
            res += 1
    return res

def majority_vote(file):
    rows = np.loadtxt(fname = file, delimiter = "\t", skiprows = 1)
    zeros = 0
    ones = 0
    for row in rows:        
        if row[len(row)-1] == 0:
            zeros += 1
        else:
            ones += 1
    if zeros > ones:
        return 0
    else:
        return 1

def calc_error(file):
    rows = np.loadtxt(fname=file, delimiter="\t", skiprows=1)
    predicted = majority_vote(file)
    length = len(rows)
    wrong = 0
    for row in rows:
        if row[len(row)-1] != predicted:
            wrong += 1
    return wrong / length 

with open(output_file, 'w') as out:
    out.write("entropy: ")
    out.write(str(calc_entropy(input_content)))
    out.write("\n")
    out.write("error: ")
    out.write(str(calc_error(input_content)))
