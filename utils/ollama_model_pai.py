"""Ollama Language Model implementation that matches ChatGPT's settings."""

from collections.abc import Collection, Sequence
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from concordia.utils import sampling
import ollama
from typing_extensions import override

_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20

class OllamaLanguageModel(language_model.LanguageModel):
  """Language Model that uses Ollama with ChatGPT-style reliability."""

  def __init__(
      self,
      model_name: str,
      *,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
  ) -> None:
    """Initialize with minimal configuration like ChatGPT."""
    self._model_name = model_name
    self._client = ollama.Client()
    self._measurements = measurements
    self._channel = channel

  @override
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
    
    max_tokens = min(max_tokens, 4000)
    
    few_shot_prompt = """You always continue sentences provided by the user and you never repeat what the user already said.

    Question: Is Jake a turtle?
    Answer: Jake is not a turtle.

    Question: What is Priya doing right now?
    Answer: Priya is currently sleeping.

    """ + prompt

    response = self._client.generate(
        model=self._model_name,
        prompt=few_shot_prompt,
        options={
            'stop': list(terminators),
            'temperature': temperature,
            'num_predict': max_tokens,
            'seed': seed if seed is not None else -1,
        },
        keep_alive='10m',
    )
    
    result = response['response']

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(result)})

    return result

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    
    prompt = (
        prompt
        + '\nRespond EXACTLY with one of the following strings:\n'
        + '\n'.join(responses) + '.'
    )

    sample = ''
    answer = ''
    for attempts in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
        temperature = sampling.dynamically_adjust_temperature(
            attempts, _MAX_MULTIPLE_CHOICE_ATTEMPTS)

        sample = self.sample_text(
            prompt,
            temperature=temperature,
            seed=seed,
        )
        answer = sampling.extract_choice_response(sample)
        try:
            idx = responses.index(answer)
        except ValueError:
            continue
        else:
            if self._measurements is not None:
                self._measurements.publish_datum(
                    self._channel, {'choices_calls': attempts}
                )
            debug = {}
            return idx, responses[idx], debug

    raise language_model.InvalidResponseError(
        (f'Too many multiple choice attempts.\nLast attempt: {sample}, ' +
         f'extracted: {answer}')
    )