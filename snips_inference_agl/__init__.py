from deprecation import deprecated

from snips_inference_agl.__about__ import __model_version__, __version__
from snips_inference_agl.nlu_engine import SnipsNLUEngine
from snips_inference_agl.pipeline.configs import NLUEngineConfig


@deprecated(deprecated_in="0.19.7", removed_in="0.21.0",
            current_version=__version__,
            details="Loading resources in the client code is no longer "
                    "required")
def load_resources(name, required_resources=None):
    from snips_inference_agl.resources import load_resources as _load_resources

    return _load_resources(name, required_resources)
