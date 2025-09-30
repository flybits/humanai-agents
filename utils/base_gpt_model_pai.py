from concordia.language_model.gpt_model import GptLanguageModel

from collections.abc import Collection
from concordia.language_model import language_model

import time
from openai._exceptions import RateLimitError


class PaiGptLanguageModel(GptLanguageModel):
    """Language Model that uses OpenAI GPT models."""
    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
        temperature: float = language_model.DEFAULT_TEMPERATURE,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        """Samples text from GPT-4 with automatic rate limit handling."""

        # Limit max_tokens to prevent large requests
        max_tokens = min(max_tokens, 4000)

        messages = [
            {'role': 'system',
                'content': ('You always continue sentences provided by '
                            'the user and never repeat what the user already said.')},
            {'role': 'user',
                'content': 'Question: Is Jake a turtle?\nAnswer: Jake is '},
            {'role': 'assistant',
                'content': 'not a turtle.'},
            {'role': 'user',
                'content': 'Question: What is Priya doing right now?\nAnswer: Priya is currently '},
            {'role': 'assistant',
                'content': 'sleeping.'},
            {'role': 'user',
                'content': prompt}
        ]

        max_retries = 5
        retry_delay = 5  # Initial delay before retrying

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    stop=terminators,
                    seed=seed,
                )

                if self._measurements is not None:
                    self._measurements.publish_datum(
                        self._channel,
                        {'raw_text_length': len(response.choices[0].message.content)},
                    )

                return response.choices[0].message.content

            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limit exceeded. Retrying in {wait_time} seconds... ({attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Failing with rate limit error.")
                    raise  # Let the error propagate if all retries fail