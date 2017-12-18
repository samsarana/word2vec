"""Data for statement logic investigation
   v4. Going back to xreplace style roots!
"""

from sympy.abc import a,b,c
from sympy import symbols
rest = symbols('rest')
from sympy.logic import Equivalent
from sympy.logic.boolalg import And, Or, Implies
from MyNot import Not

rules_dict = {'distributive_conj_F': (a & (b | c), ((a & b) | (a & c),)), # in v4, the second elements of the tuples are ALL tuples now!
              'distributive_conj_R': ( ((a&b | a&c), (a&b | a&c | rest)), (a & (b | c), a & (b | c) | rest) ), # patterns with 'rest' unfortunately need 2 options for matching
              'distributive_disj_F': (a | (b&c), ((a | b) & (a | c),)),
              'distributive_disj_R': ( ((a|b) & (a|c), (a|b) & (a|c) & rest), (a | (b & c), ( a | (b & c) ) & rest) ),
              'de_morgan_conj_F': (Not(a & b), ((Not(a) | Not(b)),)),
              'de_morgan_conj_R': ( (Not(a) | Not(b), Not(a) | Not(b) | rest), (Not(a & b), Not(a & b) | rest)),
              'de_morgan_disj_F': (Not(a | b), ((Not(a) & Not(b)),)),
              'de_morgan_disj_R': ( (Not(a) & Not(b), Not(a) & Not(b) & rest), (Not(a | b), Not(a | b) & rest) ),
              'implication_disj_F': (a >> b, (Not(a) | b,)),
              'implication_disj_R': (Not(a) | b, (a >> b,)),
              'implication_conj_F': (a >> b, (Not(a & Not(b)),)), # why are these 2 included?
              'implication_conj_R': (Not(a & Not(b)), (a >> b,)),
              'contradiction': ( (a | (b & Not(b)), a | (b & Not(b) & rest)), (a, a)), # how about a 'tautology' law?
              'absorption_conj': ( (a & (a | b), a & (a | b) & rest), (a, a & rest)),
              'absorption_disj': ( (a | (a & b), (a | (a & b) | rest)), (a, a | rest) ),
              'double_negation_F': (a, (Not(Not(a)),)),
              'double_negation_R': (Not(Not(a)), (a,))
              }

list_of_rules = ['distributive_conj_F', 'distributive_conj_R', 'distributive_disj_F', 'distributive_disj_R', 'de_morgan_conj_F', 'de_morgan_conj_R', 'de_morgan_disj_F', 'de_morgan_disj_R', 'implication_disj_F', 'implication_disj_R', 'implication_conj_F', 'implication_conj_R', 'contradiction', 'absorption_conj', 'absorption_disj', 'double_negation_F', 'double_negation_R']

LEN_RULES_LIST = 17