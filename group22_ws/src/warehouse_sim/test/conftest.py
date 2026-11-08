from fake_interfaces import fake_message_class
import pytest
from ros_stand_ins import install_ros_stand_ins, require_ros_from_environment

install_ros_stand_ins(require_real=require_ros_from_environment())


@pytest.fixture
def fake_interface_messages(monkeypatch):
    """Make the converter build fake messages; returns (frame, region) classes."""
    from warehouse_sim import thermal_observation_ros

    frame_class = fake_message_class('ThermalFrameObservation')
    region_class = fake_message_class('ThermalRegion')
    monkeypatch.setattr(thermal_observation_ros, 'ThermalFrameObservationMsg', frame_class)
    monkeypatch.setattr(thermal_observation_ros, 'ThermalRegionMsg', region_class)
    return frame_class, region_class
