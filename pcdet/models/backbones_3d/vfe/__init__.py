from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE
from .image_vfe import ImageVFE
from .dynamic_vfe import DynamicVFE
from .hard_vfe import HardVFE
from .vfe_template import VFETemplate

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'ImageVFE': ImageVFE,
    'DynamicVFE': DynamicVFE,
    'HardVFE': HardVFE
}
