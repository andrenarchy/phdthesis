#!/usr/bin/python

import string

upper = string.ascii_uppercase
lower = string.ascii_lowercase

def command(c, macro, cpre=None, cpost=None):
    return '\\newcommand\\{cpre}{c}{cpost}{{{macro}{{{c}}}}}'.format(c=c, macro=macro, cpre=cpre if cpre is not None else '', cpost=cpost if cpost is not None else '')

def print_commands(chars, macro, cpre=None, cpost=None):
    for c in chars:
        print(command(c, macro, cpre, cpost))
    print('')

print_commands(upper, '\\operatorinf', 'oi')
print_commands(upper, '\\operatorfin', 'of')
print_commands(lower, '\\vect', 'v')
print_commands(upper, '\\vecttuple', 'vt')
print_commands(upper, '\\vectspace', 'vs')
