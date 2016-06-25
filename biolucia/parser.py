from biolucia.model import (Model, Constant, Rule, Initial, ODE, State, Dose, Event, Effect, EventDirection,
                            AnalyticSegment)

from funcparserlib.lexer import make_tokenizer
from funcparserlib.parser import (some, maybe, many, finished, skip,
                                  forward_decl, NoParseError, Parser)
from re import VERBOSE
import aenum
from numpy import inf
import sympy as sp
from functools import reduce
from collections import OrderedDict


def test_all(parser: Parser, input):
    try:
        return (parser + skip(finished)).parse(input)
    except NoParseError as e:
        return e


def parse_all(parser: Parser, characters):
    tokens = token_phase(characters)
    return test_all(parser, tokens)


# Convert a sequence of characters to tokens
def token_phase(characters: str):
    specs = [
        ('comment', (r'#.*',)),
        ('space', (r'[ \t\r]+',)),
        ('float', (r'''
            -?                  # Minus
            ([0-9]+)            # Int
            (\.[0-9]+)          # Frac
            ([Ee][+-]?[0-9]+)?  # Exp''', VERBOSE)),
        ('integer', (r'0|([1-9][0-9]*)',)),
        ('name', (r'[A-Za-z_][A-Za-z_0-9]*',)),
        ('grouping', (r'[\(\)\[\]\{\}]',)),
        ('operator', (r'[~!@#$%^&*<>:?/\\|\-\+=]+',)),
        ('prime', (r"'",)),
    ]

    useless = ['comment', 'space']

    tokenizer = make_tokenizer(specs)

    return tuple(token for token in tokenizer(characters) if token.type not in useless)


# Match a particular string
op = lambda value: some(lambda token: token.value == value)

# Match a particular string and discard the result
op_ = lambda value: skip(some(lambda token: token.value == value))

expression = forward_decl()
real = some(lambda x: x.type == 'float') >> (lambda x: float(x.value))
integer = some(lambda x: x.type == 'integer') >> (lambda x: int(x.value))
number = integer | real
symbol = some(lambda x: x.type == 'name') >> (lambda name: sp.Symbol(name.value))


def make_function(matches):
    arguments = matches[1]
    combined = [] if arguments is None else [arguments[0]] + arguments[1]
    function = sp.__dict__[str(matches[0])]
    return function(*combined)
function = symbol + op_('(') + maybe(expression + many(op_(',') + expression)) + op_(')') >> make_function


factor = number | function | symbol | op_('(') + expression + op_(')')


def make_exponent(matches):
    combined = tuple(reversed([matches[0]] + matches[1]))  # Exponentiation is right associative
    value = combined[0]
    for factor in combined[1:]:
        value = sp.Pow(factor, value)
    return value
exponent = factor + many(op_('^') + factor) >> make_exponent


def make_term(matches):
    value = matches[0]
    for op, exponent in matches[1]:
        if op.value == '/':
            exponent = sp.Pow(term, -1)  # This is how sympy handles divide
        value = sp.Mul(value, exponent)
    return value
term = exponent + many((op('*') | op('/')) + exponent) >> make_term


def make_unary_term(matches):
    value = matches[1]
    for op in matches[0]:
        if op.value == '-':
            value = -value
    return value
unary_term = many(op('+') | op('-')) + term >> make_unary_term


def make_expression(matches):
    value = matches[0]
    for op, term in matches[1]:
        if op.value == '-':
            term = sp.Mul(-1, term)  # This is how sympy handles minus
        value = sp.Add(value, term)
    return value
expression.define(unary_term + many((op('+') | op('-')) + unary_term) >> make_expression)


def make_one_sided_time_constant(matches):
    direction, time = matches
    if direction.value == '<':
        return -inf, time
    else:
        return time, inf
one_sided_time = op_('(') + op_('t') + (op('<') | op('>')) + number + op_(')') >> make_one_sided_time_constant


def make_two_sided_time_constant(matches):
    time1, time2 = matches
    return time1, time2
two_sided_time = op_('(') + number + op_('<') + op_('t') + op_('<') + number + op_(')') >> make_two_sided_time_constant

time_range = one_sided_time | two_sided_time


def make_constant(matches):
    name = matches[0]
    if matches[1].value == '+=':
        additive = True
    else:
        additive = False
    value = matches[2]
    return name, Constant(name, value, additive)
constant = symbol + (op('=') | op('+=')) + number >> make_constant


def make_rule(matches):
    name, range, equal_type, expr = matches
    if range is None:
        first = -inf
        last = inf
    else:
        first, last = range
    if equal_type.value == '+=':
        additive = True
    else:
        additive = False
    return name, Rule(name, AnalyticSegment(first, last, expr), additive)
rule = symbol + maybe(time_range) + (op('=') | op('+=')) + expression >> make_rule


def make_initial(matches):
    name = matches[0]
    if matches[1].value == '+=':
        additive = True
    else:
        additive = False
    value = matches[2]
    return name, Initial(value, additive)
initial = symbol + op_('*') + (op('=') | op('+=')) + expression >> make_initial


def make_ode(matches):
    name, range, equal_type, expr = matches
    if range is None:
        first = -inf
        last = inf
    else:
        first, last = range
    if equal_type.value == '+=':
        additive = True
    else:
        additive = False
    return name, ODE(AnalyticSegment(first, last, expr), additive)
ode = symbol + maybe(time_range) + op_("'") + (op('=') | op('+=')) + expression >> make_ode


def make_dose(matches):
    name = matches[0]
    time = matches[1]
    value = matches[3]
    if matches[2].value == '+=':
        value += name
    return name, Dose(time, value)
dose = symbol + op_("(") + number + op_(")") + (op('=') | op('+=')) + expression >> make_dose


def make_effect(matches):
    name = matches[0]
    if matches[1].value == '+=':
        value = matches[2] + name
    else:
        value = matches[2]
    return Effect(name, value)
effect = symbol + (op('=') | op('+=')) + expression >> make_effect


def make_event(matches):
    trigger = matches[0] - matches[2]
    direction = matches[1]
    effects = [matches[3]] + matches[4]
    if direction.value == '<':
        effect_direction = EventDirection.down
    else:
        effect_direction = EventDirection.up
    return Event(trigger, effect_direction, True, effects)
event = op_('@') + op_('(') + expression + (op('<') | op('>')) + expression + op_(")") + \
        effect + many(op_(',') + effect) >> make_event


component = constant | rule | initial | ode | dose | event


def read_model(filename):
    # Slurp file
    with open(filename) as file:
        character_lines = file.read().splitlines()

    # Tokenize
    token_lines = []
    for line in character_lines:
        tokens_line = token_phase(line)

        if tokens_line:
            # Keep only non-blank lines
            token_lines.append(tokens_line)

    class Section(aenum.AutoNumberEnum):
        options = ()
        components = ()

    active_section = Section.components


    components = []

    tokval = lambda x: x.value
    toktype = lambda t: some(lambda x: x.type == t) >> tokval
    header = op_('%') + toktype('name')

    for line in token_lines:
        maybe_section_name = test_all(header, line)
        if isinstance(maybe_section_name, str):
            # Header line encountered
            if maybe_section_name == 'options':
                active_section = Section.options
            elif maybe_section_name == 'components':
                active_section = Section.components
            else:
                raise ValueError()  # TODO (drhagen): better error
            continue

        if active_section == Section.options:
            pass
        elif active_section == Section.components:
            component_i = test_all(component, line)
            if isinstance(component_i, NoParseError):
                raise component_i  # TODO (drhagen): collect all errors are report them at once
            components.append(component_i)
        else:
            raise ValueError()  # Unreachable

    parts, events = collapse_components(components)

    new_model = Model(parts, events)

    return new_model


def collapse_components(components):
    parts = []
    events = []

    # Group components by name
    component_groups = OrderedDict()  # It is nice to keep the order of the components
    for name, element in components:
        if name not in component_groups:
            component_groups[name] = []
        component_groups[name].append(element)

    # Separate the elements into different classes, ensuring that only compatible elements are defined
    for name, elements in component_groups.items():
        non_additive_rule = None
        additive_rules = []
        non_additive_initial = None
        additive_initials = []
        non_additive_ode = None
        additive_odes = []
        doses = []
        for element in elements:
            if isinstance(element, Constant) or isinstance(element, Rule):
                if not element.additive:
                    if non_additive_rule is None:
                        non_additive_rule = element
                    else:
                        # Check that there is no overlap with any other non-additive segments of this rule
                        if non_additive_rule.has_overlap(element):
                            raise ValueError('Rule {name} is mentioned multiple times'.format(name=name))
                        else:
                            combined_expr = non_additive_rule.value + element.value
                            non_additive_rule = Rule(name, combined_expr)
                else:
                    additive_rules.append(element)
            elif isinstance(element, Initial):
                if not element.additive:
                    if non_additive_initial is None:
                        non_additive_initial = element
                    else:
                        raise ValueError('Initial {name} is mentioned multiple times'.format(name=name))
                else:
                    additive_initials.append(element)
            elif isinstance(element, ODE):
                if not element.additive:
                    if non_additive_ode is None:
                        non_additive_ode = element
                    else:
                        # Check that there is no overlap with any other non-additive segments of this ODE
                        if non_additive_ode.has_overlap(element):
                            raise ValueError('ODE {name} is mentioned multiple times'.format(name=name))
                        else:
                            combined_expr = non_additive_ode.value + element.value
                            non_additive_ode = ODE(combined_expr)
                else:
                    additive_odes.append(element)
            elif isinstance(element, Dose):
                doses.append(element)
            else:
                assert False

        is_rule = non_additive_rule or additive_rules
        is_state = non_additive_initial or additive_initials or non_additive_ode or additive_odes or doses

        if is_state and is_rule:
            raise ValueError('Part {name} is mentioned as both a rule and a state'.format(name=name))
        elif is_rule:
            # This is a rule
            new_rule = reduce(lambda first, second: first + second, additive_rules, non_additive_rule)
            parts.append(new_rule)
        elif is_state:
            # This is a state
            new_initial = reduce(lambda first, second: first + second, additive_initials, non_additive_initial)
            new_ode = reduce(lambda first, second: first + second, additive_odes, non_additive_ode)
            new_ode = State(name, new_initial, doses, new_ode)
            parts.append(new_ode)
        else:
            assert False  # Unreachable

    return tuple(parts), tuple(events)
