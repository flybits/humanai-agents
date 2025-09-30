# File: metrics/action_alignment.py

from collections.abc import Sequence
import collections
import numpy as np

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import clock as game_clock
from concordia.typing import entity_component
from concordia.typing import logging
from concordia.utils import measurements as measurements_lib

# Scale: Tuple of strings '1' through '10'
NEW_DEFAULT_ADHERENCE_SCALE = tuple(str(i) for i in range(1, 11))
DEFAULT_ADHERENCE_CHANNEL_NAME = 'action_alignment'

class ActionAlignmentMetric(entity_component.ContextComponent):
  """Metric of action alignment to a player's identity."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_chronicle: str,
      clock: game_clock.GameClock,
      scale: Sequence[str] = NEW_DEFAULT_ADHERENCE_SCALE,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
      measurements: measurements_lib.Measurements | None = None,
      metric_channel: str = DEFAULT_ADHERENCE_CHANNEL_NAME,
      agent_name_for_logging: str = "UnknownAgent",
  ):
    self._model = model
    self._player_chronicle = player_chronicle
    self._clock = clock
    self._scale = scale
    self._logging_channel = logging_channel
    self._agent_name_for_logging = agent_name_for_logging
    self._base_metric_channel_name = metric_channel

    self._timestep = 0
    self._measurements = measurements
    
    self._metric_channel = f"{self._agent_name_for_logging}_{self._base_metric_channel_name}"

    if self._measurements:
      self._measurements.get_channel(self._metric_channel)

  def set_entity(self, entity: entity_component.EntityWithComponents) -> None:
      """Overrides base set_entity to update agent name and metric channel if needed."""
      super().set_entity(entity)
      true_agent_name = entity.name
      if self._agent_name_for_logging != true_agent_name:
          self._agent_name_for_logging = true_agent_name
          self._metric_channel = f"{self._agent_name_for_logging}_{self._base_metric_channel_name}"
          if self._measurements:
              self._measurements.get_channel(self._metric_channel)

  def post_act(
      self,
      action_attempt: str,
  ) -> str:
      agent_name = self._agent_name_for_logging
      if self.get_entity():
          agent_name = self.get_entity().name

      # --- Start of 2-step "Chain of Thought" evaluation ---

      # Step 1: Generate a Rationale
      # This forces the LLM to analyze the action before scoring it.
      doc_for_rationale = interactive_document.InteractiveDocument(self._model)
      rationale_prompt = (
          f"An agent's core identity is described by a set of facts. You will be shown the agent's "
          f"identity and their recent action.\n\n"
          f"IDENTITY for {agent_name}:\n--- IDENTITY START ---\n{self._player_chronicle}\n--- IDENTITY END ---\n\n"
          f"ACTION taken by {agent_name}: '{action_attempt}'\n\n"
          f"Your task: First, provide a brief, one-sentence rationale explaining how this action aligns "
          f"or misaligns with the agent's identity. Consider both consistent and inconsistent aspects before concluding."
          f"\n\nRationale:"
      )
      
      rationale = doc_for_rationale.open_question(
          rationale_prompt,
          max_tokens=250,
          terminators=(),
          answer_prefix=f"" # Allow the LLM to generate the full sentence.
      ).strip()


      # Step 2: Score the action based on the generated rationale
      # The LLM now has its own reasoning as context, leading to a more considered score.
      doc_for_scoring = interactive_document.InteractiveDocument(self._model)
      scoring_prompt = (
          f"Agent {agent_name}'s core identity is described as follows:\n"
          f"--- IDENTITY START ---\n{self._player_chronicle}\n--- IDENTITY END ---\n\n"
          f"Action: {action_attempt}\n\n"
          f"Previously generated rationale for this action: \"{rationale}\"\n\n"
          f"Task: Based on the provided rationale, now evaluate how consistent the action is with the identity, "
          f"using a scale from 1 to 10.\n"
          f"Guidance for the scale:\n"
          f"- 1 means: Very inconsistent with the identity.\n"
          f"- 5 means: Neutral, unclear, or unrelated to the identity.\n"
          f"- 10 means: Very consistent with the identity.\n"
          f"Please choose the number (1-10) that best describes the action's consistency:"
      )

      answer_idx = doc_for_scoring.multiple_choice_question(scoring_prompt, self._scale)
      chosen_score_str = self._scale[answer_idx]
      actual_score_value = int(chosen_score_str)

      # --- End of 2-step evaluation ---
      
      datum = {
          'time_str': self._clock.now().strftime('%H:%M:%S'),
          'clock_step': self._clock.get_step(),
          'timestep': self._timestep,
          'value_float': float(actual_score_value),
          'value_str': chosen_score_str,
          'player': agent_name,
          'action_evaluated': action_attempt[:200] + ('...' if len(action_attempt) > 200 else ''),
          'rationale': rationale, # Log the rationale for transparency
      }

      if self._measurements:
          self._measurements.publish_datum(self._metric_channel, datum)

      print(f"\n[Action Alignment Check @ {datum['time_str']} for {agent_name}]")
      print(f"  Action: {action_attempt[:150] + '...' if len(action_attempt) > 150 else action_attempt}")
      print(f"  Rationale: {datum['rationale']}")
      print(f"  Alignment Score: {datum['value_str']}/10 (Value: {datum['value_float']:.1f})")
      print(f"----------------------------------------------------")

      self._logging_channel(datum)
      self._timestep += 1
      return ''