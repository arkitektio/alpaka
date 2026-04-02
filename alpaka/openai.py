from openai import OpenAI
from fakts_next import get_current_fakts_next


def alpakaAI() -> OpenAI:
    fakts = get_current_fakts_next()

    client = OpenAI(
        base_url=fakts.get_alias("alpaka").to_http_path("/llm/v1"),
        api_key=fakts.get_token(),
    )
    return client
