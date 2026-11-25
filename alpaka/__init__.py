from .alpaka import Alpaka
from .streaming import RoomStream, stream_into_room

try:
    # The tunnel needs httpx, which arrives with the `alpaka[openai]` extra.
    from .tunnel import AlpakaEndpoint, build_async_openai, build_openai
except ImportError:
    pass

__all__ = [
    "Alpaka",
    "RoomStream",
    "stream_into_room",
    "AlpakaEndpoint",
    "build_openai",
    "build_async_openai",
]

# Marks every type alpaka sends by id, so an app using `Alpaka` finds them.
from . import structures as _structures  # noqa: E402, F401
