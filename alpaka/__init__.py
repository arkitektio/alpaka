from .alpaka import Alpaka
from .streaming import RoomStream, stream_into_room

# The service is declared with arkitekt-spec, a core dependency: it is always there.
from .arkitekt import alpaka as alpaka_service

try:
    # The tunnel needs httpx, which arrives with the `alpaka[openai]` extra.
    from .tunnel import AlpakaEndpoint, build_async_openai, build_openai
except ImportError:
    pass

__all__ = [
    "Alpaka",
    "RoomStream",
    "stream_into_room",
    "alpaka_service",
    "AlpakaEndpoint",
    "build_openai",
    "build_async_openai",
]
