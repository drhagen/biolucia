from enum import IntEnum
from collections import OrderedDict
from typing import Tuple, Union, Dict, List, Callable

from multipledispatch import Dispatcher
from sympy import Expr, Piecewise, Symbol, lambdify, And, diff, Float
import numpy as np
from numpy import inf, nan
from itertools import permutations
from sympy.utilities.iterables import topological_sort
from sympy.parsing.sympy_parser import parse_expr
from numbers import Real

from .ode import IntegrableSystem


def to_expression(expression: Union[Expr, str, Real]) -> Expr:
    if isinstance(expression, str):
        from biolucia.parser import ModelParsers
        expression = ModelParsers.expression.parse(expression).or_die()
    elif isinstance(expression, Real):
        expression = Float(expression)
    return expression


class AnalyticSegment:
    def __init__(self, start: float, stop: float, expression: Union[Expr, Real, str]):
        self.start = start
        self.stop = stop
        self.expression = to_expression(expression)

    def contains(self, symbol: Union[Symbol, str]) -> bool:
        return self.expression.has(symbol)

    def __eq__(self, other):
        return self.start == other.start and self.stop == other.stop and self.expression == other.expression

    def __str__(self):
        if self.start == -inf and self.stop == inf:
            return str(self.expression)
        elif self.start == -inf:
            return str(self.expression) + ' for t < ' + str(self.stop)
        elif self.stop == inf:
            return str(self.expression) + ' for t > ' + str(self.start)
        else:
            return str(self.expression) + ' for ' + str(self.start) + ' < t < ' + str(self.stop)

    __repr__ = __str__


class PiecewiseAnalytic:
    def __init__(self, segments: List[AnalyticSegment]):
        # All parts must be non-overlapping and sorted
        self.segments = segments
        
    def subs(self, replacers: List[Tuple[str, Union['PiecewiseAnalytic', Real, Expr]]]) -> 'PiecewiseAnalytic':
        old_segments = self.segments
        new_segments = []
        for name, replacer in replacers:
            current_self_index = 0
            current_replacer_index = 0

            if isinstance(replacer, Real) or isinstance(replacer, Expr):
                # Segments are unaffected because replacer is uniform
                for current_segment in old_segments:
                    if isinstance(current_segment.expression, Real):
                        new_segments.append(current_segment)
                    else:
                        new_segments.append(AnalyticSegment(current_segment.start,
                                                            current_segment.stop,
                                                            current_segment.expression.subs({name: replacer})))
            else:
                while current_self_index < len(old_segments) and current_replacer_index < len(replacer.segments):
                    current_segment = old_segments[current_self_index]
                    current_replacer = replacer.segments[current_replacer_index]

                    if current_segment.stop <= current_replacer.start:
                        # No overlap, drop earlier one
                        current_self_index += 1
                    elif current_replacer.stop <= current_segment.start:
                        # No overlap, drop earlier one
                        current_replacer_index += 1
                    else:
                        # Overlap exists, do substitution
                        start = max(current_segment.start, current_replacer.start)
                        stop = min(current_segment.stop, current_replacer.stop)
                        if isinstance(current_segment.expression, Real):
                            expression = current_segment.expression
                        else:
                            expression = current_segment.expression.subs(
                                {name: current_replacer.expression}
                            )
                        new_segments.append(AnalyticSegment(start,
                                                            stop,
                                                            expression))

                        if current_segment.stop >= current_replacer.stop:
                            current_replacer_index += 1
                        if current_segment.stop <= current_replacer.stop:
                            current_self_index += 1

            old_segments = new_segments
            new_segments = []

        return PiecewiseAnalytic(old_segments)

    def contains(self, symbol: Union[Symbol, str]) -> bool:
        for segment in self.segments:
            if segment.contains(symbol):
                return True
        return False

    def evaluate(self, time: float) -> Expr:
        for segment in self.segments:
            if segment.start <= time < segment.stop:
                return segment.expression
        return Float(nan)

    def __add__(self, other: Union[Real, 'PiecewiseAnalytic']) -> 'PiecewiseAnalytic':
        new_parts = []

        if isinstance(other, Real):
            for current_part in self.segments:
                new_parts.append(AnalyticSegment(current_part.start, current_part.stop,
                                                 current_part.expression + other))

        else:
            current_self_index = 0
            current_other_index = 0

            current_start = -inf

            while True:
                # First handle all the termination criteria
                if current_self_index == len(self.segments) and current_other_index == len(other.segments):
                    break
                if current_self_index == len(self.segments):
                    current_other = other.segments[current_other_index]
                    new_parts.append(AnalyticSegment(max(current_start, current_other.start), current_other.stop,
                                                     current_other.expression))
                    new_parts += other.segments[current_other_index + 1:]
                    break
                if current_other_index == len(other.segments):
                    current_part = self.segments[current_self_index]
                    new_parts.append(AnalyticSegment(max(current_start, current_part.start), current_part.stop,
                                                     current_part.expression))
                    new_parts += self.segments[current_self_index + 1:]
                    break

                current_part = self.segments[current_self_index]
                current_other = other.segments[current_other_index]

                # Advance to first start point if there is a gap
                current_start = max(current_start, min(current_part.start, current_other.start))

                if current_part.start <= current_start and current_other.start <= current_start:
                    # Segments will be added
                    stop = min(current_part.stop, current_other.stop)
                    new_parts.append(AnalyticSegment(current_start, stop,
                                                     current_part.expression + current_other.expression))
                    current_start = stop
                else:
                    # Only one segment will be used
                    if current_part.start <= current_start:
                        # It's the self segment
                        stop = min(current_part.stop, current_other.start)
                        new_parts.append(AnalyticSegment(current_start, stop, current_part.expression))
                        current_start = stop
                    else:
                        # It's the other segment
                        stop = min(current_other.stop, current_part.start)
                        new_parts.append(AnalyticSegment(current_start, stop, current_other.expression))
                        current_start = stop

                if current_part.stop <= current_start:
                    current_self_index += 1
                if current_other.stop <= current_start:
                    current_other_index += 1

        return PiecewiseAnalytic(new_parts)

    def has_overlap(self, other: 'PiecewiseAnalytic') -> bool:
        # TODO (drhagen): use the smart ordered looping from subs
        for segment_i in self.segments:
            for segment_j in other.segments:
                if (segment_j.stop < segment_i.start < segment_j.start or
                      segment_j.start < segment_i.stop < segment_j.stop or
                      segment_j.start >= segment_i.start and segment_j.stop <= segment_i.stop ):
                    return True
        return False

    def diff(self, symbol: Union[Symbol, str]) -> 'PiecewiseAnalytic':
        new_segments = []

        for segment in self.segments:
            new_expression = diff(segment.expression, symbol)
            if (len(new_segments) > 0 and new_expression == new_segments[-1].expression
                    and new_segments[-1].stop == segment.start):
                new_segments[-1].stop = segment.stop
            else:
                new_segments.append(AnalyticSegment(segment.start, segment.stop, new_expression))

        return PiecewiseAnalytic(new_segments)

    @staticmethod
    def convert(obj: Union[str, Real, Expr, AnalyticSegment, 'PiecewiseAnalytic']) -> 'PiecewiseAnalytic':
        if isinstance(obj, (Expr, Real, str)):
            from biolucia.parser import ModelParsers
            value = PiecewiseAnalytic([AnalyticSegment(-inf, inf, obj)])
            # TODO (drhagen): make this actually parse piecewise
        elif isinstance(obj, AnalyticSegment):
            value = PiecewiseAnalytic([obj])
        elif isinstance(obj, PiecewiseAnalytic):
            value = obj
        else:
            raise ValueError(f'Object of type {type(obj)} cannot be converted to PiecewiseAnalytic')
        return value

    def to_function(self, variables: List[Union[str, Symbol]]) -> Callable[..., float]:
        return lambdify(variables, self.to_sympy())

    def to_sympy(self) -> Piecewise:
        t = Symbol('t')

        pieces = []
        for segment in self.segments:
            if segment.start == -inf and segment.stop == inf:
                pieces.append((segment.expression, True))
            elif segment.start == -inf:
                pieces.append((segment.expression, t < segment.stop))
            elif segment.stop == inf:
                pieces.append((segment.expression, segment.start <= t))
            else:
                pieces.append((segment.expression, And(segment.start <= t, t < segment.stop)))

        # By default, return nan rather than None
        pieces.append((nan, True))

        return Piecewise(*pieces)

    @property
    def discontinuities(self) -> List[float]:
        return [time for part in self.segments for time in (part.start, part.stop) if -inf < time < inf]

    def __eq__(self, other: 'PiecewiseAnalytic'):
        return self.segments == other.segments

    def __str__(self):
        return ' and '.join(map(str, self.segments))

    __repr__ = __str__


class Effect:
    def __init__(self, target: Union[str, Symbol], value: Union[PiecewiseAnalytic, Expr, str]):
        if isinstance(target, Symbol):
            target = str(target)
        self.target = target
        self.value = PiecewiseAnalytic.convert(value)

    def subs(self, components: List[Union['Rule', 'Constant']]) -> 'Effect':
        # TODO: what is this evalute(0) doing?
        replacers = [(component.name, component.evaluate(0)) for component in components]
        return Effect(self.target, self.value.subs(replacers))

    def __eq__(self, other):
        return self.target == other.target and self.value == other.value

    def __str__(self):
        return f'{self.target} = {self.value}'

    __repr__ = __str__


class EventDirection(IntEnum):
    up = +1
    down = -1


class Event:
    def __init__(self, trigger: Union[PiecewiseAnalytic, Expr, str], direction: EventDirection, include_jumps: bool,
                 effects: List[Effect]):
        self.trigger = PiecewiseAnalytic.convert(trigger)
        self.direction = direction
        self.include_jumps = include_jumps
        self.effects = effects

    def subs(self, components: List[Union['Rule', 'Constant']]) -> 'Event':
        # TODO: what is this evalute(0) doing?
        replacers = [(component.name, component.evaluate(0)) for component in components]
        new_trigger = self.trigger.subs(replacers)
        new_effects = [effect.subs(components) for effect in self.effects]
        return Event(new_trigger, self.direction, self.include_jumps, new_effects)

    def __eq__(self, other):
        return (self.trigger == other.trigger and self.direction == other.direction
                and self.include_jumps == other.include_jumps and self.effects == other.effects)

    def __str__(self):
        dir_str = '<' if self.direction == EventDirection else '>'
        return f'@({self.trigger} {dir_str} 0) ' + ', '.join([str(effect) for effect in self.effects])

    __repr__ = __str__


class Initial:
    def __init__(self, value: Union[Expr, Real, str], additive: bool = False):
        self.value = to_expression(value)
        self.additive = additive

    def subs(self, components: List[Union['Rule', 'Constant']]) -> 'Initial':
        # TODO: what is this evalute(0) doing?
        replacers = [(component.name, component.evaluate(0)) for component in components]
        return Initial(self.value.subs(replacers), self.additive)

    def __eq__(self, other):
        return self.value == other.value and self.additive == other.additive

    def __str__(self):
        eq_str = '+=' if self.additive else '='
        return f'* {eq_str} {self.value}'

    __repr__ = __str__


class Dose:
    def __init__(self, time: float, value: Union[Expr, Real, str]):
        self.time = time
        self.value: Expr = parse_expr(value) if isinstance(value, str) else value

    def subs(self, components: List[Union['Rule', 'Constant']]):
        if isinstance(self.value, Real):
            return self
        else:
            replacers = [(component.name, component.evaluate(0)) for component in components]
            return Dose(self.time, self.value.subs(replacers))

    def __eq__(self, other):
        return self.time == other.time and self.value == other.value

    def __str__(self):
        return f'({self.time}) = {self.value}'

    __repr__ = __str__


class Ode:
    def __init__(self, value: Union[PiecewiseAnalytic, AnalyticSegment, Expr, Real, str], additive: bool = False):
        self.value = PiecewiseAnalytic.convert(value)
        self.additive = additive

    def subs(self, components: List[Union['Rule', 'Constant']]) -> 'Ode':
        replacers = [(rule.name, rule.value) for rule in components]
        return Ode(self.value.subs(replacers), self.additive)

    def has_overlap(self, other: 'Ode') -> bool:
        if isinstance(other, Constant):
            return True
        else:
            return self.value.has_overlap(other.value)

    def __add__(self, other):
        return Ode(self.value + other.value, self.additive)

    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        eq_str = '+=' if self.additive else '='
        return f"' {eq_str} {self.value}"

    __repr__ = __str__


class Component:
    def __init__(self, name: Union[Symbol, str]):
        if isinstance(name, Symbol):
            name = name.name
        self.name = name

    def contains(self, other: str) -> bool:
        raise NotImplementedError

    def subs(self, components: List[Union['Rule', 'Constant']]) -> 'Component':
        raise NotImplementedError

    def evaluate(self, time: float) -> Expr:
        raise NotImplementedError

    @staticmethod
    def parse(component: str) -> 'Component':
        from biolucia.parser import ModelParsers  # Circular import
        return ModelParsers.component.parse(component).or_die()

    @staticmethod
    def topological_sort(components: List[Union['Rule', 'Constant']]) -> 'List[Component]':
        # Returns a list of components sorted so that no element contains an element earlier in the list

        n_components = len(components)
        indexes_components = list(range(n_components))

        # Find all pairs (i,j) where rule i contains a reference to rule j
        edges = [(i, j) for i, j in permutations(indexes_components, 2) if components[i].contains(components[j].name)]

        # Sort rules by topology
        index_sorted = topological_sort([indexes_components, edges])
        return [components[i] for i in index_sorted]


class Constant(Component):
    def __init__(self, name: Union[Symbol, str], value: float, additive: bool = False):
        super().__init__(name)
        self.value = value
        self.additive = additive

    def contains(self, other: str) -> bool:
        return False

    def subs(self, components: List[Union['Rule', 'Constant']]) -> 'Constant':
        return self

    def evaluate(self, time: float) -> float:
        return self.value

    def has_overlap(self, other: Union['Constant', 'Rule']) -> bool:
        return True

    def copy(self, name: Union[Symbol, str] = None, value: float = None, additive: bool = None) -> 'Constant':
        if name is None:
            name = self.name

        if value is None:
            value = self.value

        if additive is None:
            additive = self.additive

        return Constant(name, value, additive)

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value

    def __repr__(self):
        eq_str = '+=' if self.additive else '='
        return f'{self.name} {eq_str} {self.value}'


class Rule(Component):
    def __init__(self, name: Union[Symbol, str], value: Union[PiecewiseAnalytic, AnalyticSegment, Expr, str],
                 additive: bool = False):
        super().__init__(name)
        self.value = PiecewiseAnalytic.convert(value)
        self.additive = additive

    def subs(self, components: List[Union['Rule', 'Constant']]) -> 'Rule':
        replacers = [(rule.name, rule.value) for rule in components]
        return Rule(self.name, self.value.subs(replacers), self.additive)

    def contains(self, symbol: str) -> bool:
        return self.value.contains(symbol)

    def evaluate(self, time: float) -> float:
        return self.value.evaluate(time)

    def has_overlap(self, other: Union['Constant', 'Rule']) -> bool:
        if isinstance(other, Constant):
            return True
        else:
            return self.value.has_overlap(other.value)

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value

    def __repr__(self):
        eq_str = '+=' if self.additive else '='
        return f'{self.name} {eq_str} {self.value}'


class State(Component):
    def __init__(self, name: Union[Symbol, str], initial: Initial, doses: List[Dose], ode: Ode):
        super().__init__(name)
        self.initial = initial
        self.doses = doses
        self.ode = ode

    def subs(self, components: List[Union['Rule', 'Constant']]) -> 'State':
        new_initial = self.initial.subs(components)
        new_doses = [dose.subs(components) for dose in self.doses]
        new_ode = self.ode.subs(components)
        return State(self.name, new_initial, new_doses, new_ode)

    def add_dose(self, dose: Dose) -> 'State':
        return self.copy(doses=self.doses + [dose])

    def copy(self, name: str = None, initial: Initial = None, doses: List[Dose] = None, ode: Ode = None) -> 'State':
        if name is None:
            name = self.name

        if initial is None:
            initial = self.initial

        if doses is None:
            doses = self.doses

        if ode is None:
            ode = self.ode

        return State(name, initial, doses, ode)

    def __eq__(self, other):
        return (self.name == other.name and self.initial == other.initial and self.doses == other.doses and
                self.ode == other.ode)

    def __repr__(self):
        return f'{self.name}{self.initial}  {self.name}{self.ode}'


class Model:
    def __init__(self, parts: List[Component] = (), events: List[Event] = ()):
        if not isinstance(parts, list):
            parts = list(parts)
            
        if not isinstance(events, list):
            events = list(events)

        self.parts = parts
        self.events = events

    def copy(self, parts: List[Component] = None, events: List[Event] = None) -> 'Model':
        if parts is None:
            parts = self.parts

        if events is None:
            events = self.events

        return Model(parts, events)

    def add(self, *items: Union[Component, Event, str]) -> 'Model':
        from biolucia.parser import collapse_components

        ready_parts = []
        ready_events = []
        parsed_components = []
        for component in items:
            if isinstance(component, Component):
                ready_parts.append(component)
            elif isinstance(component, Event):
                ready_events.append(component)
            elif isinstance(component, str):
                parsed_components.append(Component.parse(component))

        parsed_parts, parsed_events = collapse_components(parsed_components)

        return update(update(self, Model(ready_parts, ready_events)), Model(parsed_parts, parsed_events))

    def __getitem__(self, name: Union[Symbol, str]) -> Component:
        if isinstance(name, Symbol):
            name = name.name

        name_map = dict((part.name, part) for part in self.parts)

        return name_map[name]

    def default_parameters(self) -> Dict[str, float]:
        parameters: List[Tuple[str, float]] = []

        for part in self.parts:
            if isinstance(part, Constant):
                parameters.append((part.name, part.value))

        return OrderedDict(parameters)

    def update_parameters(self, parameters: Dict[str, float]) -> 'Model':
        new_parts = []

        for part in self.parts:
            if part.name in parameters:
                if isinstance(part, Constant):
                    new_parts.append(part.copy(value=parameters[part.name]))
                else:
                    raise ValueError(f'Part {part.name} cannot be a parameter because it is not a Constant')
            else:
                new_parts.append(part)

        return self.copy(parts=new_parts)

    # TODO: memoize this
    def build_odes(self, parameter_names: List[str] = ()) -> IntegrableSystem:
        parameter_names = list(parameter_names)

        active_constant_indexes = []
        active_constants = []  # type: List[Constant]
        inactive_constant_indexes = []
        inactive_constants = []  # type: List[Constant]
        rule_indexes = []
        rule_objects = []  # type: List[Rule]
        state_indexes = []
        state_objects = []  # type: List[State]

        for i, part in enumerate(self.parts):
            if isinstance(part, Constant):
                if part.name in parameter_names:
                    active_constant_indexes.append(i)
                    active_constants.append(part)
                else:
                    inactive_constant_indexes.append(i)
                    inactive_constants.append(part)
            elif isinstance(part, Rule):
                rule_indexes.append(i)
                rule_objects.append(part)
            elif isinstance(part, State):
                state_indexes.append(i)
                state_objects.append(part)

        event_objects = self.events

        state_names = [state.name for state in state_objects]

        # Substitute constants and rules into rules
        rule_objects = Rule.topological_sort(rule_objects)
        for i in range(len(rule_objects)):
            rule_objects[i] = rule_objects[i].subs(inactive_constants)
            rule_objects[i] = rule_objects[i].subs(rule_objects[i + 1:])

        # Substitute constants and rules into odes
        state_objects = [state.subs(inactive_constants) for state in state_objects]
        state_objects = [state.subs(rule_objects) for state in state_objects]

        event_objects = [event.subs(inactive_constants) for event in event_objects]
        event_objects = [event.subs(rule_objects) for event in event_objects]

        # Extract symbols
        parameter_values = np.asarray([constant.value for constant in active_constants], dtype=float)
        parameters = OrderedDict(zip(parameter_names, parameter_values))

        initials = [state.initial.value for state in state_objects]
        ode = [state.ode.value.to_sympy() for state in state_objects]
        states = OrderedDict(zip(state_names, list(zip(initials, ode))))

        # Collect discontinuities
        discontinuities = [time for state in state_objects for time in state.ode.value.discontinuities]
        discontinuities = list(sorted(set(discontinuities)))  # sorted distinct

        # Build dose functions
        # Map time to every dose given at that time
        dose_groups = dict()  # type: Dict[float, List[Expr]]
        for (i, state) in enumerate(state_objects):
            for dose in state.doses:
                time = dose.time
                if time not in dose_groups:
                    # Initialize a dose at this time
                    dose_groups[time] = [Symbol(state.name) for state in state_objects]  # list is going to be mutated

                # Replace the zero at the state's index with it's dose function
                # Time is guaranteed to be unique within a state so overwriting is fine
                dose_groups[time][i] = dose.value

        # Build event functions
        events = []
        for event_object in event_objects:
            trigger = event_object.trigger
            direction = event_object.direction
            effects = list(map(Symbol, state_names))
            for effect in event_object.effects:
                # TODO (drhagen): handle rare case of duplicate targets with additive effect
                i_target = state_names.index(str(effect.target))
                effects[i_target] = effect.value.to_sympy()
            events.append((trigger, direction, effects))

        # Build output functions
        output_syms: List[Tuple[int, str, Expr]] = []
        for i, part in zip(active_constant_indexes, active_constants):
            output_syms.append((i, part.name, Symbol(part.name)))

        for i, part in zip(inactive_constant_indexes, inactive_constants):
            output_syms.append((i, part.name, part.value))

        for i, part in zip(rule_indexes, rule_objects):
            output_syms.append((i, part.name, part.value.to_sympy()))

        for i, part in zip(state_indexes, state_objects):
            output_syms.append((i, part.name, Symbol(part.name)))

        output_syms.sort(key=lambda tup: tup[0])

        outputs = OrderedDict((output[1], output[2]) for output in output_syms)

        return IntegrableSystem(parameters, states, discontinuities, dose_groups, events, outputs)

    def observable_names(self) -> List[str]:
        return [part.name for part in self.parts]


# Multiple dispatch contraption for update
def update(model: Model, variant: Model):
    return update.dispatcher(model, variant)

update.dispatcher = Dispatcher('update')


@update.dispatcher.register(Model, Model)
def update_model_model(model: Model, variant: Model):
    if not variant.parts and not variant.events:  # Don't update if there is nothing to update
        return model

    # Add to old parts if the new part is additive and they have compatible types (Rule/Constant and State)
    # Otherwise replace the component with the component
    old_parts = model.parts
    new_parts = list(old_parts)
    component_names = [part.name for part in old_parts]

    for new_part in variant.parts:
        if new_part.name in component_names:
            index = component_names.index(new_part.name)
            old_part = old_parts[index]

            if ((isinstance(new_part, Rule) or isinstance(new_part, Constant)) and
                    (isinstance(old_part, Rule) or isinstance(old_part, Constant))):
                if isinstance(new_part, Constant) and isinstance(old_part, Constant):
                    # Only constants can be added together to make a new constant
                    new_parts[index] = Constant(old_part.name, old_part.value + new_part.value, old_part.additive)
                else:
                    combined_expr = old_part.value + new_part.value
                    new_parts[index] = Rule(old_part.name, combined_expr, old_part.additive)
            elif isinstance(new_part, State) and isinstance(old_part, State):
                if new_part.initial is None:
                    new_initial = old_part.initial
                elif new_part.initial.additive:
                    new_initial = Initial(old_part.initial.value + new_part.initial.value,
                                          old_part.initial.additive)
                else:
                    new_initial = new_part.initial

                if new_part.ode is None:
                    new_ode = old_part.ode
                elif new_part.ode.additive:
                    new_ode = Ode(old_part.ode.value + new_part.ode.value,
                                  old_part.initial.additive)
                else:
                    new_ode = new_part.ode

                new_doses = old_part.doses + new_part.doses

                new_parts[index] = State(old_part.name, new_initial, new_doses, new_ode)
            else:
                raise ValueError(f'Component {new_part.name} has type {type(old_part)}'
                                 f' and cannot be added with type {type(new_part)}')

        else:
            new_parts.append(new_part)

    # Events are simply concatenated
    new_events = model.events + variant.events

    return Model(new_parts, new_events)
