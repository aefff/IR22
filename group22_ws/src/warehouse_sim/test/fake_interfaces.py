from pathlib import Path
from types import SimpleNamespace

# Source of truth for the fake messages below: the interface definitions.
INTERFACE_MSG_DIR = Path(__file__).resolve().parents[2] / 'hazard_interfaces' / 'msg'


def read_msg_fields(message_name):
    """Return {field name: type} from a hazard_interfaces .msg file."""
    fields = {}
    path = INTERFACE_MSG_DIR / f'{message_name}.msg'
    for line in path.read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            field_type, field_name = line.split()
            fields[field_name] = field_type
    return fields


def fake_message_class(message_name):
    fields = read_msg_fields(message_name)
    required = set(fields) - {'header'}

    class FakeMessage:
        # Test double for a generated message. It checks field names against
        # the .msg file and demands every non-header field, which is stricter
        # than generated messages. It proves nothing about real serialisation.
        __slots__ = tuple(fields)

        def __init__(self, **kwargs):
            if set(kwargs) != required:
                raise TypeError(
                    f'{message_name}: got fields {sorted(kwargs)}, '
                    f'expected {sorted(required)}',
                )
            for name, value in kwargs.items():
                setattr(self, name, value)
            if 'header' in fields:
                self.header = SimpleNamespace(
                    stamp=SimpleNamespace(sec=0, nanosec=0), frame_id='',
                )

    FakeMessage.__name__ = FakeMessage.__qualname__ = f'Fake{message_name}'
    return FakeMessage


# .msg primitive names that generated-message introspection
# (get_fields_and_field_types) spells differently; others are unchanged.
_INTROSPECTION_NAMES = {'bool': 'boolean', 'float32': 'float', 'float64': 'double'}
_PRIMITIVES = {
    'bool', 'float32', 'float64', 'string',
    'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64',
}


def introspection_type(msg_type, package='hazard_interfaces'):
    """Spell a .msg field type the way get_fields_and_field_types() reports it."""
    if msg_type.endswith('[]'):
        return f'sequence<{introspection_type(msg_type[:-2], package)}>'
    if '[' in msg_type or '<' in msg_type:
        raise ValueError(f'unsupported .msg type form: {msg_type!r}')
    if msg_type in _PRIMITIVES:
        return _INTROSPECTION_NAMES.get(msg_type, msg_type)
    if '/' in msg_type:
        return msg_type  # already package-qualified, e.g. std_msgs/Header
    if msg_type[:1].isupper():
        return f'{package}/{msg_type}'  # message in the same package
    raise ValueError(f'unknown .msg type: {msg_type!r}')
