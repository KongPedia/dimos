from abc import ABC, abstractmethod
from dimos.hardware.interface import HardwareInterface
from dimos.types.sample import Sample


'''
Base class for all dimos robots, both physical and simulated.
'''
class Robot(ABC):
    def __init__(self, hardware_interface: HardwareInterface):
        self.hardware_interface = hardware_interface

    @abstractmethod
    def perform_task(self):
        """Abstract method to be implemented by subclasses to perform a specific task."""
        pass
    @abstractmethod
    def do(self, *args, **kwargs):
     """Executes motion."""
        pass

    def update_hardware_interface(self, new_hardware_interface: HardwareInterface):
        """Update the hardware interface with a new configuration."""
        self.hardware_interface = new_hardware_interface

    def get_hardware_configuration(self):
        """Retrieve the current hardware configuration."""
        return self.hardware_interface.get_configuration()

    def set_hardware_configuration(self, configuration):
        """Set a new hardware configuration."""
        self.hardware_interface.set_configuration(configuration)
