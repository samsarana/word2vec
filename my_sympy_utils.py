"""Helper functions for Sympy work"""

from sympy.abc import A, B, C, D
from sympy.core import Atom, Wild, count_ops, Basic
from sympy import sympify

def traverse_bfs(expr):
    """Traverses the tree of the given expression breadth-first, printing
       the subtree at every node
    """
    queue = [expr]
    while queue:
        stat = queue.pop(0)
        print(stat)
        queue.extend(stat.args)
        

def num_terms(expr):
    """Returns the number of terms comprising a sympy expression.
       Perhaps the measure of complexity/length should actually be number of
       variables, or depth of tree?
       
       from sympy.core import count_ops
       count_ops(Or(a,b, And(d,b)))
       This gives the number of predicates
    """
    return expr.count(Wild('a'))


def num_vars(expr):
    """Returns the number of variables in an expression
       There's probably a way in the API to do this!
       count(*1st var in use*) + count(*2nd var in use*) + ...
       would be another way to do this!
       Though note that the current method includes wilds as well, that wouldn't
    """
    return num_terms(expr) - count_ops(expr)


def max_depth(expr):
    """Returns the depth of the given expression tree i.e. the number of
       nodes encountered in the maximum length path from root to a leaf
    """
    if isinstance(expr, Atom):
        return 1
    else:
        return 1 + max([ max_depth(arg) for arg in expr.args ])


def ave_numterms_maxdepth(expr):
    """Returns the average of the number of terms in the expression and its
       max depth
    """
    return (num_terms(expr) + max_depth(expr)) / 2


def ave_depth(entire_expr):
    """Returns the mean of the depth of each unique path from root to leaf.
    """
    depth = 1
    leaf_depths = []
    queue = [entire_expr]
    while queue:
        exprs_at_current_depth = queue.copy()
        queue.clear()
        for expr in exprs_at_current_depth:
            if isinstance(expr, Atom):
                leaf_depths.append(depth)
            else:
                queue.extend(expr.args)
        depth += 1
    return sum(leaf_depths) / len(leaf_depths)


def ave_numterms_avedepth(expr):
    """Returns the average of the number of terms in the expression and its
       average depth
    """
    return (num_terms(expr) + ave_depth(expr)) / 2


def complexity(expr, complexity_measure=num_terms):
    """Returns some measure of the complexity of the given expression tree
       Complexity measures to choose from: num_terms, max_depth,
       ave_numterms_maxdepth, ave_depth, ave_numterms_avedepth
    """
    if expr == True or expr == False:
        return 1
    else:
        return(complexity_measure(expr))

    
def prune_below_d(expr, d):
    """Takes an expression and returns that same expression but with all the
       subtrees that are at a level deeper than d pruned off
       Level indexing start at 1
       Special case if d == 0, I've chosen this to mean 'do no pruning'
       e.g. exp = And(Or(A,B), Not(Or(B,C)), A)
            prune_below_d(exp, 2) --> And(A, Not(), Or())
    """    
    expr = str(expr)
    if d == 0:
        return expr
    pruned = ''
    nesting = 0
    for char in expr:
        if char == ')':
            nesting -= 1        
        if nesting < d:
            pruned += char
        if char == '(':
            nesting += 1
    return pruned


def clean_up(training_examples):
    flattened = [item for sublist in training_examples for item in sublist]
    fully_clean = False
    while not fully_clean:
        fully_clean = True
        for i in range(len(flattened) - 1, -1, -1):
            current = flattened[i]
            up_to_current = flattened[:i]
            if current in up_to_current:
                first_index = flattened.index(current)
                new_list = flattened[:first_index] + flattened[i:]
                flattened = new_list
                fully_clean = False
                break
    return flattened
            

def write_to_output(training_examples, include_rules, d_cut):
    output_filename = 'reasoning_training_{}_{}.txt'.format(include_rules, d_cut)
    with open(output_filename, 'a') as out_file:    
        for ex in training_examples:
            for stat in ex:
                if isinstance(stat, str): # i.e. it's a rule
                    if include_rules:
                        out_file.write(stat)
                        out_file.write('\n')
                else:
                    out_file.write(prune_below_d(stat, d_cut))
                    out_file.write('\n')
            out_file.write('\n')    


def export(training_examples):
    for include_rules in (True, False):
        for d_cut in range(6):
            write_to_output(training_examples, include_rules, d_cut)