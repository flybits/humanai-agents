from __future__ import annotations
from typing import Any, Callable, Optional 

from collections.abc import Callable
import datetime

from concordia.language_model import language_model
from concordia.typing import agent as agent_lib
from concordia.typing import component as component_lib
from concordia.typing import logging as logging_lib

from concordia.components import agent as agent_components
from concordia.components.agent.question_of_query_associated_memories import QuestionOfQueryAssociatedMemories


class _BaseComponent(component_lib.Component):
    def __init__(
        self,
        component_name: str,
        logging_channel: Optional[logging_lib.LoggingChannel] = None,
    ):
        super().__init__()
        self._component_name_override = component_name
        self._logging_channel_callback = logging_channel
        self._entity: Optional[component_lib.EntityWithComponents] = None
        self._last_log: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._component_name_override

    def set_entity(self, entity: component_lib.EntityWithComponents) -> None:
        self._entity = entity

    def get_entity(self) -> component_lib.EntityWithComponents:
        if self._entity is None:
            raise ValueError(f"Entity has not been set for component {self.name}.")
        return self._entity

    def pre_observe(self, observation: str) -> str | None:
        del observation
        return None

    def post_observe(self) -> None:
        pass

    def pre_act(self, action_spec: agent_lib.ActionSpec) -> str | None:
        del action_spec
        return None

    def post_act(self, action_attempt: str) -> None:
        del action_attempt
        pass

    def update(self) -> None:
        pass

    def get_pre_act_key(self) -> str:
        return f"\n{self.name}"

    def get_pre_act_value(self) -> str:
        return ""

    def get_last_log(self) -> dict[str, Any] | None:
        return self._last_log if self._last_log else None


class StaticChronicleInjector(_BaseComponent):
    def __init__(
        self,
        agent_name: str,
        chronicle_text: str,
        *,
        component_name_suffix: str = "_StaticChronicleInjector",
        pre_act_key: str = "\n[Core Identity Principles from My Chronicle]:",
        logging_channel: Optional[logging_lib.LoggingChannel] = None,
    ):
        self_component_name = f"{agent_name}{component_name_suffix}"
        super().__init__(
            component_name=self_component_name,
            logging_channel=logging_channel,
        )
        self._raw_chronicle_text = chronicle_text
        self._pre_act_key_for_injector = pre_act_key
        self._agent_name_for_log = agent_name

    def get_pre_act_key(self) -> str:
        return self._pre_act_key_for_injector

    def get_pre_act_value(self) -> str:
        self._last_log = {
            "name": self.name,
            "type": "StaticChronicleInjectedValue",
            "agent": self._agent_name_for_log,
            "key_used": self.get_pre_act_key(),
            "value_preview": self._raw_chronicle_text[:150] + "…",
        }
        if self._logging_channel_callback:
            self._logging_channel_callback(self._last_log)
        return self._raw_chronicle_text

    def pre_act(self, action_spec: agent_lib.ActionSpec) -> str:
        del action_spec
        value = self.get_pre_act_value()
        return f"{self.get_pre_act_key()}: {value}"


class FeelingAboutLifeProgressComponent(QuestionOfQueryAssociatedMemories):
    def __init__(
        self,
        model: language_model.LanguageModel,
        clock_now: Callable[[], datetime.datetime],
        logging_channel: logging_lib.LoggingChannel,
        pre_act_key: str, # This key is for the overall component if its pre_act is directly used
        memory_component_name: str = agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
    ):
        super().__init__(
            model=model,
            clock_now=clock_now,
            logging_channel=logging_channel,
            pre_act_key=pre_act_key, # e.g., "Feeling about recent progress in life"
            queries=['feeling about recent progress in life'], # Query string itself is lowercase
            question="How would one describe {agent_name}'s {query} given the "
                     'following statements? ',
            memory_component_name=memory_component_name,
            add_to_memory=False,
            num_memories_to_retrieve=10,
            summarization_question=None, # Ensures query is used as a key in the output of get_pre_act_value
        )

class FullRetrievalComponent(_BaseComponent):
    def __init__(
        self,
        agent_name: str,
        chronicle_text_for_static_part: str,
        feeling_component: FeelingAboutLifeProgressComponent,
        logging_channel: Optional[logging_lib.LoggingChannel] = None,
        component_name_suffix: str = "_FullRetrieval",
    ):
        super().__init__(
            component_name=f"{agent_name}{component_name_suffix}",
            logging_channel=logging_channel
        )
        self._agent_name_for_log = agent_name
        # self._raw_chronicle_text is already stripped in the previous version's __init__
        # If not, ensure it is:
        self._raw_chronicle_text = chronicle_text_for_static_part.strip()
        self._feeling_component = feeling_component

    def set_entity(self, entity: component_lib.EntityWithComponents) -> None:
        super().set_entity(entity)
        self._feeling_component.set_entity(entity)

    def get_pre_act_value(self) -> str:
        """
        Returns the content block:
        core characteristics:
        [Chronicle Text]

        feeling about recent progress in life: [feeling text]
        This is used by Plan, which will prefix it with its own overall label.
        """
        # Label the chronicle part as "core characteristics:"
        # Ensure the actual chronicle text starts on a new line for readability.
        labeled_chronicle_part = f"core characteristics:\n{self._raw_chronicle_text}"

        # self._feeling_component.get_pre_act_value() returns:
        # "feeling about recent progress in life: [actual feeling text]"
        feeling_part_correctly_labeled = self._feeling_component.get_pre_act_value().strip()

        combined_state = f"{labeled_chronicle_part}\n\n{feeling_part_correctly_labeled}"

        self._last_log = {
            "name": self.name,
            "type": "FullRetrievalValue (for Plan)",
            "agent": self._agent_name_for_log,
            "chronicle_preview": labeled_chronicle_part[:150] + "…", # Log the labeled version
            "feeling_preview": feeling_part_correctly_labeled[:100] + "...",
            "combined_preview": combined_state[:250] + "...", # Increased preview length
        }
        if self._logging_channel_callback:
            self._logging_channel_callback(self._last_log)
        return combined_state # The Plan component's label will provide the leading newline

    def pre_act(self, action_spec: agent_lib.ActionSpec) -> str:
        """
        Returns the fully formatted identity block for general context (e.g., CONTEXTS printout).
        This includes the top-level "\nIdentity characteristics:" header.
        """
        del action_spec
        
        value_block = self.get_pre_act_value() # Gets the new structure

        # This header matches 'identity_label' from build_a_citizen, including the colon
        header = "\nIdentity characteristics:"
        
        fully_formatted_identity = f"{header}\n{value_block}"
        
        self._last_log = { # Update log for this specific output
            "name": self.name,
            "type": "FullRetrievalValue (for CONTEXTS)",
            "agent": self._agent_name_for_log,
            "full_output_preview": fully_formatted_identity[:300] + "...", # Increased preview length
        }
        if self._logging_channel_callback:
             self._logging_channel_callback(self._last_log)
        return fully_formatted_identity

    def update(self) -> None:
        super().update()
        self._feeling_component.update()