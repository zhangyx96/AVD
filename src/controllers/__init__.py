REGISTRY = {}

from .basic_controller import BasicMAC
from .basic_controller_avd import BasicMAC_AVD

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["avd_mac"] = BasicMAC_AVD