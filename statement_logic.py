"""Generating and manipulating statements of informal logic"""

from sympy.core import Basic, Wild
from sympy.logic.boolalg import BooleanFunction, And, Or, Implies, simplify_logic
from MyNot import Not # take care when overriding such a rudimentary Class than any imports below this one don't cheekily override this special 'Not' import
#from sympy.logic.boolalg import Not as TheirNot
from sympy.abc import A, B, C, D
import random
from numpy import argmin, array, float64
from numpy.random import choice
from logical_laws_v4 import list_of_rules, LEN_RULES_LIST # The other dicts from logical_laws are imported locally! Probably suboptimal that way, change later once I've worked out which of the two 'apply' functions is better...
from my_sympy_utils import *
import time
from sympy.unify.usympy import *
from sympy.abc import a,b,c
from sympy import symbols
rest = symbols('rest')

# Double negation should be applied a bit more than the other rules, because
# there are so many places it can be applied (anywhere!) and just a select
# few points of application may turn out to be really useful. So the idea
# is that this will increase our chance of hitting on one of these
# When the algorithm decides to applying double_negation_F, it will actually,
# then and there, calculate the number of possible places where double negation
# can be applied, and apply it randomly in DOUBLE_NEG_APPL_FRACTION of
# these places
DOUBLE_NEG_APPL_FRACTION = 0.2
# if there are mutliples instantiations/ways of applying a law, NUM_UNIFICATIONS
# of these will be tried
NUM_UNIFICATIONS = 5
COMPLEXITY_FUNC = lambda x: complexity(x, complexity_measure=ave_numterms_maxdepth)

class StatementGenerator:
    """
    Class used for generating Sympy logical expressions
    Attributes:
       - nodes - list of the possible nodes in the expression tree
       - num_children - dictionary mapping from the names of possible nodes to
                        the number of children that each should have in the tree
       - starting_probs - numpy array of the desired initial probability of each
                          node in the node list being chosen as the next node of
                          the expression tree
       - max_tree_depth - max depth of the expression tree, defined as the
                          number of vertical layers of branches (is it?). Must be >= 2
       - prob_adjustment - the amount by which the probability of internal and
                           leaf nodes being chosen is decreased and increased
                           respectively
       - first_leaf_node_index - index of nodes and starting_probs arrays
                                 corresponding to the first node that is to be
                                 considered a leaf node of the expression tree
       
    Methods:
       - update_probs()
       - add_children()
    """
    def __init__(self, nodes, num_children, starting_probs, max_tree_depth, first_leaf_node_index):
        self.nodes = nodes
        self.num_children = num_children
        self.starting_probs = starting_probs
        self.prob_adjustment = max(starting_probs) / (max_tree_depth - 1)
        self.first_leaf_node_index = first_leaf_node_index

    def update_probs(self, probs):
        """Decreases probabilities of internal nodes being chosen and increases
           probabilities of leaf nodes being chosen as the next node in the
           expression tree.
           Internal nodes are those up to (and not including) the
           first_leaf_node_index, leaf nodes are those after it.
        """
        probs[:self.first_leaf_node_index] -= self.prob_adjustment # an exponentially decaying prob_adjustment would shorten the starting expressions
        probs[self.first_leaf_node_index:] += self.prob_adjustment
        probs /= sum(probs) # ensure normalisation
        return abs(probs) # ensure tiny negative probs that should be 0 don't cause an error when passed to np.random.choice()

    def add_children(self, expr, probs):
        """
           Given a possibly incomplete node of an expression tree, returns that
           same node with children chosen with probabilities according to probs
           If any of those children need children themselves for the expression
           tree to be complete, this method will be internally recursively 
           called until it returns a fully complete expression tree.
        """
        complete_children_for_expr = []        
        possibly_incomplete_children = choice(self.nodes, size=num_children[expr], p=probs)
        for child in possibly_incomplete_children:
            if not child.is_Atom and not isinstance(child, BooleanFunction): # do I need 2nd bit?
                # adjust probablities of preds and vars being chosen
                new_probs = self.update_probs(probs.copy()) # copying the array is essential because the same identifier in one stack frame references the very same array even in a different stack frame
                complete_child = self.add_children(child, new_probs)
            else:
                complete_child = child
            complete_children_for_expr.append(complete_child) 
        complete_expr = expr(*tuple(complete_children_for_expr))
        return complete_expr
    
    def generate_complete_expression(self):
        """A wrapper for the add_children function.
           Choses a random predicate (i.e. an internal not leaf node) for the 
           root of the tree and then calls the add_children function which calls
           itself recursively to generate a complete expression tree.
        """
        predicates = self.nodes[:self.first_leaf_node_index]
        #print('predicates are {}'.format(predicates))
        starting_expr = random.choice(predicates)
        return self.add_children(starting_expr, self.starting_probs)


def apply1(rule, original_statement, appl_point):
    """Takes a Sympy expression and a STRING describing a rule, and applies that
       rule once to the original_statement. The rule is applied at the highest possible
       level of the expression tree.
       Returns the transformed statement, or the same statement if the rule could
       not be applied.
       Could be modified so that rule is applied at the nth highest node, or at
       a randomly chosen node.
       If it was (while queue), this would select the deepest place of application
       If appl_point > len(mapping) then just return the same expression. Hopefully
       this shouldn't happen much
       This version uses logical_laws v3
    """
    from logical_laws_v3 import rules_dict, replacement_function_dict
    pattern = rules_dict[rule]
    make_replacements = replacement_function_dict[rule]
    found_match = False
    queue = [original_statement]
    transformed_statement = original_statement # default return
    current_point = 0
    while not found_match and queue:
        subtree = queue.pop(0)
        repl_dict_generator = unify(subtree, pattern, variables=(a,b,c,rest)) # might need to convert set to tuple
        try:
            repl_dict = next(repl_dict_generator) # *** algorithm never tries out alternative instantiations at the moment
            if current_point == appl_point:
                subtree_to_replace = subtree
                found_match = True
            current_point += 1
        except StopIteration: # law cannot be applied to this part of the subtree
            queue.extend(subtree.args)
    if found_match:
        replacement_subtree = make_replacements(repl_dict, rule)
        transformed_statement = original_statement.xreplace({subtree_to_replace: replacement_subtree}) # slightly buggy: if there is an identical subtree. lack brainpower to work out how to do a reconstruction as the bfs is going on
    return transformed_statement


def apply2(rule, original_statement, appl_point):
    """Takes a Sympy expression and a STRING describing a rule, and applies that
       rule once to the original_statement. The rule is applied at the highest possible
       level of the expression tree.
       Returns the transformed statement, or the same statement if the rule could
       not be applied.
       Could be modified so that rule is applied at the nth highest node, or at
       a randomly chosen node.
       If it was (while queue), this would select the deepest place of application
       If appl_point > len(mapping) then just return the same expression. Hopefully
       this shouldn't happen much
       This version uses logical_laws v4
    """
    from logical_laws_v4 import rules_dict
    pattern_from, (pattern_to_wo_rest, *pattern_to_w_rest) = rules_dict[rule]
    found_match = False
    queue = [original_statement]
    result = []
    current_point = 0
    while not found_match and queue:
        subtree = queue.pop(0)
        repl_dict_generator = unify(subtree, pattern_from, variables=(a,b,c,rest)) # excess variables not an issue? computational slowdown? probably not compared to recomputing exactly which vars are included every time...?
        repl_dict = next(repl_dict_generator, None) # *** algorithm never tries out alternative instantiations at the moment
        if repl_dict: # if there are instantiations at all
            if current_point == appl_point:
                subtree_to_replace = subtree
                for _ in range(NUM_UNIFICATIONS):
                    if repl_dict: # if there's still more alternative instantiations
                        if rest in repl_dict:
                            replacement_subtree = pattern_to_w_rest[0].xreplace(repl_dict)
                        else:
                            replacement_subtree = pattern_to_wo_rest.xreplace(repl_dict)
                        transformed_statement = original_statement.xreplace({subtree_to_replace: replacement_subtree}) # slightly buggy: if there is an identical subtree. lack brainpower to work out how to do a reconstruction as the bfs is going on. But perhaps I should... it would probably also be more efficient...
                        result.append(transformed_statement)
                        repl_dict = next(repl_dict_generator, None)
                        found_match = True
                    else:
                        break
            current_point += 1
        else: # law cannot be applied to this part of the subtree
            queue.extend(subtree.args)
    return result if result else [original_statement]


def remove_empty_lists(list_of_lists):
    """Returns a new list identical to list_of_lists, but with all the empty 
       lists in list_of_lists removed
    """
    new_list = [nested_list for nested_list in list_of_lists if nested_list != []]
    return new_list


def all_manipulations_search(manipulations, branching_f):
    """Returns a list representing the search tree of all the manipulations that
       can be done on the first element in manipulations
    """
    APPLY_FUNC = apply2 # Here is where I change which version I'm using!!
    rules_list = list_of_rules.copy()
    for level in range(1, len(manipulations)):
        #print('All manips is {}'.format(manipulations))
        #print()
        for prev_stat_index, prev_stat_info in enumerate(manipulations[level - 1]):
            prev_stat, _, _ = prev_stat_info
            #print(prev_stat_index, prev_stat)
            random.shuffle(rules_list) # apply rules in different order for every statement on previous level
            rule_num = 0
            rule_appl_point = 0
            branch_num = 0
            while branch_num < branching_f:
            #for _ in range(branching_f):
                rule_name = rules_list[rule_num]
                # Special case if double negation - we want to apply this one more than the others, and in a sprinkling of places *(10 times more frequently and in 10 randomly chosen places)*
                if rule_name == 'double_negation_F':
                    total_num_matches = num_terms(prev_stat)
                    #for negation_appl_point in np.random.choice(total_num_matches, np.random.randint(1, total_num_matches), replace=False): # picks at random a random number of numbers from 1 to the number of points where negation can be applied
                    for negation_appl_point in choice(total_num_matches, int(DOUBLE_NEG_APPL_FRACTION*total_num_matches), replace=False): # picks at random DOUBLE_NEG_APPL_FRACTION of the numbers from 1 to the number of points where double negation can be applied
                        new_stats = APPLY_FUNC(rule_name, prev_stat, negation_appl_point)
                        for new_stat in new_stats:
                            if new_stat != prev_stat:
                                manipulations[level].append((new_stat, prev_stat_index, rule_name))
                            branch_num += 1
                else: # standard case of applying any other law
                    new_stats = APPLY_FUNC(rule_name, prev_stat, rule_appl_point)
                    for new_stat in new_stats:
                        if new_stat != prev_stat:
                            manipulations[level].append((new_stat, prev_stat_index, rule_name))
                        branch_num += 1
                #else:
                    #print("{} ({}) didnt work".format(rule_name, is_reversed))
                rule_num += 1
                if rule_num == LEN_RULES_LIST:
                    random.shuffle(rules_list)
                    rule_num = 0
                    rule_appl_point += 1
            if not manipulations[level]:
                # i.e. if no rules could be applied to any statements on the previous level
                print('None of the rules I tried to apply did anything')                
                return remove_empty_lists(manipulations)
    return manipulations


def best_manipulations_search(initial_stat, branching_f, cut_off_depth, include_ops=True):
    """Takes an expression and returns a list of the cut_off_depth manipulations to that
       expression that result in the expression with the least number of terms
       (which is hopefully a good measure of statement complexity).
       If include_ops is true, the list will be like
       [expr1, manipulation to get from 1 to 2, expr2, manipulation to get from
       2 to 3, expr3, ..., exprN].
       If include_ops is false, the list will simply be
       [expr1, expr2, ..., exprN].
       branching_f is the number of manipulations that are explored for each expression
       at each layer of the search tree
    """
    all_manipulations = [[] for _ in range(cut_off_depth + 1)] # cut_off_depth + 1 b/c list also includes initial expr
    all_manipulations[0] = [(initial_stat, None, None)]
    all_manipulations = all_manipulations_search(all_manipulations, branching_f)
    parent_index = argmin([COMPLEXITY_FUNC(fin_expr[0]) for fin_expr in all_manipulations[-1]])
    best_manipulations = []
    for stats in list(reversed(all_manipulations)):
        stat, new_parent_index, rule = stats[parent_index] # I could definitely make the triple tuples of stat info into a class
        best_manipulations.append(stat)
        if include_ops and rule != None:
            best_manipulations.append(rule)
        parent_index = new_parent_index  
    return list(reversed(best_manipulations))


def generate_training_examples(statement_gen, branching_f, cut_off_depth):
    initial_stat = statement_gen.generate_complete_expression()
    #initial_stat = Or(Implies(A, B), Implies(B, Implies(B, A)))
    print('Stat is initially    {}'.format(initial_stat))
    print('which has {} terms'.format(num_terms(initial_stat)))
    simplifed_stat = simplify_logic(initial_stat)
    print('Can be simplified to {}'.format(simplifed_stat))
    min_final_complexity = COMPLEXITY_FUNC(simplifed_stat)
    print()
    
    t0 = time.perf_counter()
    training_examples = [[initial_stat]]
    include = False
    for i in range(50):
        if i % 5 == 0:
            print('i={}'.format(i))
        best = best_manipulations_search(initial_stat, branching_f, cut_off_depth)
        print(best)
        #if COMPLEXITY_FUNC(best[-1]) <= COMPLEXITY_FUNC(best[0]) + 1:
        if best[-1] != best[0]:
            if training_examples[-1][-1] == initial_stat:
                training_examples[-1].extend(best[1:])
            else:
                training_examples.append(best) # is this ever happening under my current set up?
            if COMPLEXITY_FUNC(best[-1]) <= min_final_complexity:
                include = True
                break
            initial_stat = best[-1]
            # what to do if this initial_stat is as simple as it's going to get?
        #else:
            #print('best[-1]: {} \n complexity: {} \n best[0]: {} \n complexity: {}'.format(best[-1], COMPLEXITY_FUNC(best[-1]), best[0], COMPLEXITY_FUNC(best[0])))
            # perhaps we should increase branching_f and start again?
            # or introduce some more randomness in where the laws are applied?
            # or increase the double negation factor to try and hit the goldmine?
            # or maybe my algorithm is so good that things that drop out here
            # are basically as simple as they can get??!
    t1 = time.perf_counter()
    
    # Printing the generated examples
    print()
    print('Printing the generated examples...')
    for ex in training_examples:
        for stat in ex:
            print(stat)
            if isinstance(stat, BooleanFunction):
                print('Complexity = {}'.format(COMPLEXITY_FUNC(stat)))
        print()
    print('That took {:.1f}s'.format(t1-t0))
    if include:
        export(training_examples)
 

if __name__ == '__main__':
    """Generates a random statement form"""
    predicates_and_variables = [And, Or, Not, Implies, A, B, C, D]    
    num_children = {And: 2, Or: 2, Not: 1, Implies: 2, A: 0, B: 0, C: 0, D:0} # defines whether the different nodes are leaf, unary or binary
    starting_probs = array([0.25,0.25,0.25,0.25,0,0,0,0], dtype=float64)
    max_tree_depth = 4 # Must be >= 2    
    first_statement_variable_index = 4
    statement_gen = StatementGenerator(predicates_and_variables, num_children, starting_probs, max_tree_depth, first_statement_variable_index)
    
    branching_f = 20
    cut_off_depth = 3
    
    while True:
        generate_training_examples(statement_gen, branching_f, cut_off_depth)