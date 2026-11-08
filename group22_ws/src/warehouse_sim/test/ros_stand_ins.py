import importlib
import importlib.util
import os
import sys
import types

REQUIRE_ROS_ENV = 'WAREHOUSE_SIM_REQUIRE_ROS'


class StubNode:

    def __init__(self, *args, **kwargs):
        raise RuntimeError('rclpy is not installed: this Node stand-in cannot be created')


class StubCvBridgeError(TypeError):
    """Mirrors cv_bridge.CvBridgeError, which subclasses TypeError."""


class StubCvBridge:

    def __init__(self, *args, **kwargs):
        raise RuntimeError('cv_bridge is not installed: this CvBridge stand-in cannot be created')


class StubQoSProfile:

    def __init__(self, *args, **kwargs):
        raise RuntimeError('rclpy is not installed: this QoSProfile stand-in cannot be created')


class StubInterfaceMessage:
    """Stands in for generated hazard_interfaces messages; never constructible."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            'hazard_interfaces is not built: generated messages are unavailable, '
            'so this stand-in cannot be created',
        )


# Module to import -> stand-in modules (with attributes) to register if absent.
ROS_STAND_INS = {
    'rclpy.node': {
        'rclpy': {},
        'rclpy.node': {'Node': StubNode},
        'rclpy.qos': {
            'QoSProfile': StubQoSProfile,
            'DurabilityPolicy': object(),
            'HistoryPolicy': object(),
            'ReliabilityPolicy': object(),
        },
    },
    'sensor_msgs.msg': {'sensor_msgs': {}, 'sensor_msgs.msg': {'Image': object}},
    'cv_bridge': {
        'cv_bridge': {'CvBridge': StubCvBridge, 'CvBridgeError': StubCvBridgeError},
    },
    'hazard_interfaces.msg': {
        'hazard_interfaces': {},
        'hazard_interfaces.msg': {
            'ThermalFrameObservation': StubInterfaceMessage,
            'ThermalRegion': StubInterfaceMessage,
        },
    },
}


def require_ros_from_environment(environ=os.environ):
    """Return True if strict mode is on; reject values other than '0' or '1'."""
    value = environ.get(REQUIRE_ROS_ENV, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{REQUIRE_ROS_ENV} must be 0 or 1, not {value!r}')
    return value == '1'


def is_stand_in(module):
    """Return True if ``module`` is a stand-in registered by this helper."""
    return bool(getattr(module, '__ros_test_stub__', False))


def any_stand_ins_installed():
    """Return True if any module that fire_detection needs is a stand-in."""
    return any(is_stand_in(sys.modules.get(name)) for name in ROS_STAND_INS)


def import_or_stand_in(module_name, stand_ins, *, require_real):
    """
    Import ``module_name``, or register stand-ins if its package is absent.

    ``stand_ins`` maps each module name to register to its attributes, parents
    first. Returns the imported or stand-in module.
    """
    top_level = module_name.partition('.')[0]
    if require_real or importlib.util.find_spec(top_level) is not None:
        return importlib.import_module(module_name)  # errors propagate

    for name, attributes in stand_ins.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes, __ros_test_stub__=True)
        sys.modules[name] = module
    return sys.modules[module_name]


def install_ros_stand_ins(*, require_real):
    """Import each ROS module ``fire_detection`` needs, or a stand-in if absent."""
    for module_name, stand_ins in ROS_STAND_INS.items():
        import_or_stand_in(module_name, stand_ins, require_real=require_real)
