from sympy.logic.boolalg import Not as TheirNot
from sympy import Number, true, false

class Not(TheirNot):
    """
    Logical Not function (negation)
    Returns True if the statement is False
    Returns False if the statement is True
    Modified from original Not function to not automatically simplify double
    negation
    """

    is_Not = True

    @classmethod
    def eval(cls, arg):
        from sympy import (
            Equality, GreaterThan, LessThan,
            StrictGreaterThan, StrictLessThan, Unequality)
        if isinstance(arg, Number) or arg in (True, False):
            return false if arg else true
        #if arg.is_Not:
            #return arg.args[0] # ONLY CHANGE
        # Simplify Relational objects.
        if isinstance(arg, Equality):
            return Unequality(*arg.args)
        if isinstance(arg, Unequality):
            return Equality(*arg.args)
        if isinstance(arg, StrictLessThan):
            return GreaterThan(*arg.args)
        if isinstance(arg, StrictGreaterThan):
            return LessThan(*arg.args)
        if isinstance(arg, LessThan):
            return StrictGreaterThan(*arg.args)
        if isinstance(arg, GreaterThan):
            return StrictLessThan(*arg.args)
        
        
if __name__ == '__main__':
    """Testing"""
    from sympy.abc import X
    print(TheirNot(X))
    print(TheirNot(TheirNot(X)))
    print(Not(X))
    print(Not(Not(X)))