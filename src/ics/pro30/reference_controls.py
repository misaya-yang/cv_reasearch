"""Original Pro30 B_R controls for modules that do not export the readout."""
from functools import partial

from .common import br_result

METHODS = {}
CONTROLS = {
    'PRO30_B_R': br_result,
    'PRO30_B_R_same_samples_ridge': partial(
        br_result, method_id='PRO30_B_R_same_samples_ridge', huber=False),
}
