import sympy as sy
from numpy import inf
from functools import reduce
from collections import OrderedDict
from parsita import TextParsers, lit, reg, opt, rep, repsep, rep1sep, failure

from biolucia.model import (Model, Constant, Rule, Initial, Ode, State, Dose, Event, Effect, EventDirection,
                            AnalyticSegment)


class ModelParsers(TextParsers, whitespace=r'[ \t]*'):
    number = reg(r'[+-]?\d+(\.\d+)?([Ee][+-]?\d+)?') > float
    name = reg(r'[A-Za-z_][A-Za-z_0-9]*')
    symbol = name > sy.Symbol

    def make_function(x):
        func_name, arguments = x
        if func_name in sy.__dict__:
            func_handle = sy.__dict__[func_name]
        else:
            raise ValueError(f'Function "{func_name}" not found. Only functions in sympy.* may be used.')
        return func_handle(*arguments)

    function = name & '(' >> repsep(expression, ',') << ')' > make_function

    factor = number | function | symbol | '(' >> expression << ')'

    def make_term(x):
        x = list(reversed(x))  # Exponentiation is right associative so reverse the list
        value = x[0]
        rest = x[1:]
        for item in rest:
            value = sy.Pow(item, value)
        return value

    exponent = rep1sep(factor, '^') > make_term

    def make_term(x):
        first, rest = x
        value = first
        for op, exponent in rest:
            if op == '/':
                exponent = sy.Pow(exponent, -1)  # This is how sympy handles divide
            value = sy.Mul(value, exponent)
        return value

    term = exponent & rep(lit('*', '/') & exponent) > make_term

    def make_unary_term(x):
        ops, value = x
        for op in ops:
            if op == '-':
                value = -value
        return value

    unary_term = rep(lit('+', '-')) & term > make_unary_term

    def make_expression(x):
        first, rest = x
        value = first
        for op, term in rest:
            if op == '-':
                term = sy.Mul(-1, term)  # This is how sympy handles minus
            value = sy.Add(value, term)
        return value

    expression = unary_term & rep(lit('+', '-') & unary_term) > make_expression

    def make_one_sided_time_constant(x):
        direction, time = x
        if direction == '<':
            return -inf, time
        else:
            return time, inf

    one_sided_time = '(' >> lit('t') >> lit('<', '>') & number << ')' > make_one_sided_time_constant

    two_sided_time = '(' >> number << '<' << 't' << '<' & number << ')'

    time_range = one_sided_time | two_sided_time

    def make_constant(x):
        name, op, value = x
        if op == '+=':
            additive = True
        else:
            additive = False
        return name, Constant(name, value, additive)

    constant = name & lit('=', '+=') & number > make_constant

    def make_rule(x):
        name, domain, op, expr = x
        if not domain:
            first = -inf
            last = inf
        else:
            first, last = domain[0]
        if op == '+=':
            additive = True
        else:
            additive = False
        return name, Rule(name, AnalyticSegment(first, last, expr), additive)

    rule = name & opt(time_range) & lit('=', '+=') & expression > make_rule

    def make_initial(x):
        name, op, value = x
        if op == '+=':
            additive = True
        else:
            additive = False
        return name, Initial(value, additive)

    initial = name << '*' & lit('=', '+=') & expression > make_initial

    def make_ode(x):
        name, domain, op, expr = x
        if not domain:
            first = -inf
            last = inf
        else:
            first, last = domain[0]
        if op == '+=':
            additive = True
        else:
            additive = False
        return name, Ode(AnalyticSegment(first, last, expr), additive)

    ode = name & opt(time_range) << "'" & lit('=', '+=') & expression > make_ode

    def make_dose(x):
        name, time, op, value = x
        if op == '+=':
            value += sy.Symbol(name)
        return name, Dose(time, value)

    dose = name << "(" & number << ")" & lit('=', '+=') & expression > make_dose

    def make_effect(x):
        name, op, value = x
        if op == '+=':
            value += sy.Symbol(name)
        return Effect(name, value)

    effect = name & lit('=', '+=') & expression > make_effect

    def make_event(x):
        left, direction, right, effects = x
        trigger = left - right
        if direction == '<':
            effect_direction = EventDirection.down
        else:
            effect_direction = EventDirection.up
        return Event(trigger, effect_direction, True, effects)

    event = lit('@') >> '(' >> expression & lit('<', '>') & expression << ")" & rep1sep(effect, ',') > make_event

    component = constant | rule | initial | ode | dose | event

    eol = reg(r'(((#.*)?\n)+)|(((#.*)?\n)*(#.*)?\Z)')
    options_section = '%' >> lit('options') >> eol >> rep(failure('not implemented') << eol)
    components_section = '%' >> lit('components') >> eol >> rep(component << eol)

    def make_model(x):
        maybe_options, components = x

        if maybe_options:
            raise NotImplementedError

        parts, events = collapse_components(components)

        new_model = Model(parts, events)
        return new_model

    model = opt(options_section) & components_section > make_model
    # TODO (drhagen): collect all errors and report them at once


def collapse_components(components):
    parts = []
    events = []

    # Group components by type and name
    component_groups = OrderedDict()  # It is nice to keep the order of the components
    for component in components:
        if isinstance(component, Event):
            events.append(component)
        else:
            name, element = component
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
                            raise ValueError(f'Rule {name} is mentioned multiple times')
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
                        raise ValueError(f'Initial {name} is mentioned multiple times')
                else:
                    additive_initials.append(element)
            elif isinstance(element, Ode):
                if not element.additive:
                    if non_additive_ode is None:
                        non_additive_ode = element
                    else:
                        # Check that there is no overlap with any other non-additive segments of this ODE
                        if non_additive_ode.has_overlap(element):
                            raise ValueError(f'ODE {name} is mentioned multiple times')
                        else:
                            combined_expr = non_additive_ode.value + element.value
                            non_additive_ode = Ode(combined_expr)
                else:
                    additive_odes.append(element)
            elif isinstance(element, Dose):
                doses.append(element)
            else:
                assert False

        is_rule = non_additive_rule or additive_rules
        is_state = non_additive_initial or additive_initials or non_additive_ode or additive_odes or doses

        if is_state and is_rule:
            raise ValueError(f'Part {name} is mentioned as both a rule and a state')
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

    return parts, events


def read_model(filename) -> Model:
    with open(filename) as file:
        return ModelParsers.model.parse(file.read()).or_die()
