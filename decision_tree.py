import sys 
import numpy as np


args = sys.argv

train_input = args[1]
test_input = args[2]
max_depth = args[3]
train_out = args[4]
test_out = args[5]
metrics_out = args[6]


with open(train_input, 'r') as tr_in:
    tr_content = tr_in.readlines()
    
    
with open(test_input, 'r') as ts_in:
    ts_content = ts_in.readlines()

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None
        self.vars = set()


def majority_vote(rows):
    zeros = 0
    ones = 0
    for i in range(0, len(rows)):
        val = (rows[i])[-1]
        if val == 0:
            zeros += 1
        elif val == 1:
            ones += 1
    if zeros > ones:
        return "0"
    else:
        return "1"

def entropy(rows):
    res = 0
    length = len(rows)
    ones = 0
    zeros = 0
    for row in rows:
        if row[len(row)-1] == 0:
            zeros += 1
        else:
            ones += 1
    if ones == 0 or zeros == 0:
        res = 0
    else:
        res = (- ((ones/length) * np.log2(ones/length)) - 
                ((zeros/length) * np.log2(zeros/length)))
    return res


def cond_entropy(rows, var, val):
    rows = rows[rows[:, var] == val]
    length = len(rows)
    ones = 0
    zeros = 0
    for row in rows:
        if row[len(row)-1] == 0:
            zeros += 1
        else:
            ones += 1
    if ones == 0 or zeros == 0:
        res = 0
    else:
        res = (- ((ones/length) * np.log2(ones/length)) - 
            ((zeros/length) * np.log2(zeros/length)))
    return res
    

def mutual_info(rows, var):
    hy = entropy(rows)  
    hyx = 0
    d = dict()
    total = 0
    for row in rows:
        elem = row[var]
        if elem not in d:
            d[elem] = 1
        else:
            d[elem] += 1
        total += 1
    for val in d:
        x = cond_entropy(rows, var, val)
        hyx += ((d[val]/total)*x)
    return hy - hyx


def train(data):
    rows = np.loadtxt(fname=data, delimiter="\t", skiprows=1)
    root = tree_recurse(rows, len(rows[0])-1, [], 0, int(max_depth))
    return root

def tree_recurse(rows, n_feats, vars, depth, max_depth):
    q = Node()
    if (len(rows) == 0 or depth >= max_depth or all_equal(rows) or len(vars) >= n_feats):
        q.vote = majority_vote(rows) 
        return q
    else:
        mutual_info_val = -1
        max_attr = None
        for i in range(0, len(rows[0])-1):  
            if i not in vars:  
                x = mutual_info(rows, i)
                if x > mutual_info_val and x > 0:
                    mutual_info_val = x
                    max_attr = i
        q.attr = max_attr
        depth += 1
        vars.append(max_attr)
        left = rows[rows[:, q.attr] == 0]
        right = rows[rows[:, q.attr] == 1]
        second_vars = []
        for val in vars:
            second_vars.append(val)
        q.left = tree_recurse(left, n_feats, vars, depth, max_depth) 
        q.right = tree_recurse(right, n_feats, second_vars, depth, max_depth)
        return q

def all_equal(rows):
    x = (rows[0])[-1]
    for i in range(0, len(rows)):
        row = rows[i]
        if row[-1] != x:
            return False
    return True
    
def predict(data, tree): 
    if tree and tree.vote:
        return tree.vote
    else:
        if data[tree.attr] == 0:
            return predict(data, tree.left)
        else:
            return predict(data, tree.right)


def pretty_print(tree, rows, names):
    left = rows[rows[:,-1] == 0]
    right = rows[rows[:,-1] == 1]
    print(f"[{len(left)} 0/{len(right)} 1]") 
    preorder_dfs(rows, names, tree, 1)


def preorder_dfs(rows, names, tree, i):
    if tree == None or tree.vote:
        return
    else:
        left = rows[rows[:, tree.attr] == 0]
        right = rows[rows[:, tree.attr] == 1]
        left_ones = left[left[:,-1] == 1]
        left_zeros = left[left[:,-1] == 0]
        right_ones = right[right[:,-1] == 1]
        right_zeros = right[right[:,-1] == 0]
        print("| " * i, end = "")
        print(f"{names[tree.attr]} = 0: ", end = "")
        print(f"[{len(left_zeros)} 0/{len(left_ones)} 1]")
        preorder_dfs(left, names, tree.left, i+1)
        print("| " * i, end = "")
        print(f"{names[tree.attr]} = 1: ", end = "")
        print(f"[{len(right_zeros)} 0/{len(right_ones)} 1]")
        preorder_dfs(right, names, tree.right, i+1)
        return 


with open(train_out, 'w') as tr_out:
    res = 0
    tree = train(tr_content)
    names = tr_content[0].split("\t")
    rows = np.loadtxt(fname=tr_content, delimiter="\t", skiprows=1)  
    for i in range(0, len(rows)):  
        res = predict(rows[i], tree)
        tr_out.write(res)
        tr_out.write("\n")
    pretty_print(tree, rows, names)

with open(test_out, 'w') as ts_out:
    res = 0
    tree = train(tr_content)
    rows = np.loadtxt(fname=ts_content, delimiter="\t", skiprows=1)   
    for i in range(0, len(rows)): 
        res = predict(rows[i], tree)
        ts_out.write(res)
        ts_out.write("\n")

def calc_metrics(original, predicted):
    orig_vals = []
    pred_vals = []
    o_cols = np.loadtxt(fname = original, delimiter = "\t", skiprows=1)
    p_cols = np.loadtxt(fname = predicted, delimiter = "\t", skiprows=0)
    for i in range(0, len(o_cols)):
        val = (o_cols[i])[-1]
        orig_vals.append(val)    
    for j in range(0, len(p_cols)):
        pval = p_cols[j]
        pred_vals.append(pval)
    incorrect = 0
    for k in range(0, len(orig_vals)):
        if orig_vals[k] != pred_vals[k]:
            incorrect += 1
    error = incorrect / (len(orig_vals))
    return str(error)

with open(metrics_out, 'w') as met_out:
    met_out.write("error(train): ")
    met_out.write(calc_metrics(tr_content, train_out))
    met_out.write("\n")
    met_out.write("error(test): ")
    met_out.write(calc_metrics(ts_content, test_out))

if __name__ == '__main__':
    pass
