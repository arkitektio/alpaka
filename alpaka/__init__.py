from .alpaka import Alpaka
from .streaming import RoomStream, stream_into_room

try:
    from .arkitekt import alpaka as alpaka_service
except ImportError as e:
    # Only "rekuest is not installed" may pass silently. Anything else that fails
    # to import here (a renamed query, a rekuest too old for what the module
    # needs) is a bug, and hiding it makes this package's service vanish
    # without a word. Whether it is installed is asked the plain way.
    try:
        import rekuest  # noqa: F401 -- presence is the question
    except ImportError:
        pass
    else:
        raise e

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
