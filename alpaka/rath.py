from types import TracebackType
from typing import Optional, Type
from pydantic import Field
from rath import rath

from rath.links.auth import AuthTokenLink

from rath.links.compose import TypedComposedLink
from rath.links.dictinglink import DictingLink
from rath.links.shrink import ShrinkingLink
from rath.links.split import SplitLink


class AlpakaLinkComposition(TypedComposedLink):
    shrinking: ShrinkingLink = Field(default_factory=ShrinkingLink)
    dicting: DictingLink = Field(default_factory=DictingLink)
    auth: AuthTokenLink
    split: SplitLink


class AlpakaRath(rath.Rath):
    """Fluss Rath

    Args:
        rath (_type_): _description_
    """

    async def __aenter__(self):
        """Enter the client.

        Entering does not make it "the current client": nothing is. Calls go
        through the :class:`alpaka.alpaka.Alpaka` client that holds it.
        """
        await super().__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        await super().__aexit__(exc_type, exc_val, traceback)
