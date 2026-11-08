import importlib
import sys

import pytest
from ros_stand_ins import (
    import_or_stand_in,
    is_stand_in,
    require_ros_from_environment,
    ROS_STAND_INS,
    StubCvBridge,
    StubInterfaceMessage,
    StubNode,
    StubQoSProfile,
)

STAND_INS = {'g22_fake_ros': {}, 'g22_fake_ros.node': {'Node': StubNode}}


@pytest.fixture
def fake_site(tmp_path, monkeypatch):
    """Return a writer for throwaway packages on sys.path; unload them afterwards."""
    monkeypatch.syspath_prepend(str(tmp_path))

    def write_package(files):
        for relative_path, source in files.items():
            path = tmp_path / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        importlib.invalidate_caches()

    yield write_package
    for name in [n for n in sys.modules if n.startswith('g22_')]:
        del sys.modules[name]


def test_absent_package_gets_marked_stand_ins(fake_site):
    module = import_or_stand_in('g22_fake_ros.node', STAND_INS, require_real=False)
    assert is_stand_in(module)
    assert is_stand_in(sys.modules['g22_fake_ros'])
    assert module.Node is StubNode


def test_installed_package_is_imported_for_real(fake_site):
    fake_site({
        'g22_fake_ros/__init__.py': '',
        'g22_fake_ros/node.py': 'class Node:\n    pass\n',
    })
    for require_real in (False, True):
        module = import_or_stand_in('g22_fake_ros.node', STAND_INS, require_real=require_real)
        assert not is_stand_in(module)
        assert module.Node is not StubNode


@pytest.mark.parametrize('files, error', [
    # Binary extension or shared library failing to load (plain ImportError).
    ({'g22_fake_ros/__init__.py': "raise ImportError('libfoo.so: cannot open')\n",
      'g22_fake_ros/node.py': ''}, ImportError),
    # Missing transitive dependency: ModuleNotFoundError for a different name.
    ({'g22_fake_ros/__init__.py': 'import g22_missing_dependency\n',
      'g22_fake_ros/node.py': ''}, ModuleNotFoundError),
    # Partial installation: the package exists but the submodule does not.
    ({'g22_fake_ros/__init__.py': ''}, ModuleNotFoundError),
    # Any other failure while executing the installed module.
    ({'g22_fake_ros/__init__.py': '',
      'g22_fake_ros/node.py': "raise RuntimeError('incompatible ABI')\n"}, RuntimeError),
])
def test_broken_installed_package_is_never_replaced(fake_site, files, error):
    fake_site(files)
    with pytest.raises(error):
        import_or_stand_in('g22_fake_ros.node', STAND_INS, require_real=False)
    assert not any(is_stand_in(m) for n, m in sys.modules.items() if n.startswith('g22_'))


def test_strict_mode_fails_when_package_is_absent(fake_site):
    with pytest.raises(ModuleNotFoundError):
        import_or_stand_in('g22_fake_ros.node', STAND_INS, require_real=True)
    assert 'g22_fake_ros' not in sys.modules


@pytest.mark.parametrize('environ, expected', [
    ({}, False),
    ({'WAREHOUSE_SIM_REQUIRE_ROS': '0'}, False),
    ({'WAREHOUSE_SIM_REQUIRE_ROS': '1'}, True),
])
def test_strict_mode_setting(environ, expected):
    assert require_ros_from_environment(environ) is expected


@pytest.mark.parametrize('value', ['', 'true', 'yes', '2'])
def test_strict_mode_rejects_ambiguous_values(value):
    with pytest.raises(ValueError):
        require_ros_from_environment({'WAREHOUSE_SIM_REQUIRE_ROS': value})


@pytest.mark.parametrize('stand_in', [
    StubNode, StubCvBridge, StubQoSProfile, StubInterfaceMessage,
])
def test_stand_in_constructors_refuse_to_pretend(stand_in):
    with pytest.raises(RuntimeError):
        stand_in('fire_detection')


def test_generated_interface_package_has_only_unconstructible_stand_ins():
    # Without a ROS build, no test can obtain a real-looking generated message.
    attributes = ROS_STAND_INS['hazard_interfaces.msg']['hazard_interfaces.msg']
    assert set(attributes) == {'ThermalFrameObservation', 'ThermalRegion'}
    assert all(value is StubInterfaceMessage for value in attributes.values())
