from .alpaka import Alpaka
from .streaming import RoomStream, stream_into_room

try:
    from .arkitekt import AlpakaService
    from .rekuest import structure_reg
except ImportError:
    pass

try:
    # Needs the arkitekt integration (fakts); the openai()/aopenai() factories
    # additionally need the `alpaka[openai]` extra at call time.
    from .tunnel import (
        AlpakaEndpoint,
        aget_endpoint,
        alpakaAI,
        aopenai,
        get_endpoint,
        openai,
    )
except ImportError:
    pass


__all__ = [
    "Alpaka",
    "RoomStream",
    "stream_into_room",
    "AlpakaService",
    "structure_reg",
    "AlpakaEndpoint",
    "get_endpoint",
    "aget_endpoint",
    "openai",
    "aopenai",
    "alpakaAI",
]
