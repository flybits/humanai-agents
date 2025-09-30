"""
Official experiment code for the paper:
ID-RAG: Identity Retrieval-Augmented Generation for Long-Horizon Persona Coherence in Generative Agents.

Experiment Setting - Human-AI Agent Social Simulation: Riverbend Election:

An illustrative social simulation with 5 players which simulates the day of mayoral elections in an imaginary town called Riverbend. 
The first two players, Alice and Bob, are simulated by Human-AI Agents. They both are running for mayor. 
The third player, Charlie, is trying to ruin Alices' reputation with disinformation. 
The last two players, Ellen and Dorothy, have no specific agenda, apart from voting in the election.
Charlie, Ellen, and Dorothy are simulated by the baseline Generative Agents provided by Concordia (modeled after the seminal architecture).
"""

import concurrent.futures
import datetime
import random
import os
import sys

import sentence_transformers

from concordia.components.agent import to_be_deprecated as components
from concordia.components import game_master as gm_components
from concordia.associative_memory import associative_memory
from concordia.associative_memory import blank_memories
from concordia.associative_memory import formative_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.environment import game_master
from concordia.utils import html as html_lib
from concordia.utils import measurements as measurements_lib

from concordia.agents import entity_agent_with_logging
from concordia.components import agent as agent_components
from concordia.memory_bank import legacy_associative_memory

from utils.base_gpt_model_pai import PaiGptLanguageModel
from utils.ollama_model_pai import OllamaLanguageModel
from utils.to_csv import export_metrics_to_csvs
from election_component import Elections

from metrics.online_id_recall import OnlineIdentityRecallMetric
from metrics.action_alignment import ActionAlignmentMetric

from plots import plot_metrics

from predefined_formative_memories import PREDEFINED_FORMATIVE_MEMORIES, parse_historical_memory
from full_retrieval import FeelingAboutLifeProgressComponent, FullRetrievalComponent 
from hai_metrics_utils import *

# Parse command line arguments
if len(sys.argv) < 3:
    print("Usage: python start.py <llm_model> <experiment_mode> [number_of_timesteps]")
    print("  llm_model: gpt-4o-mini, gpt-4o, or qwen2.5:7b")
    print("  experiment_mode: baseline, id-rag, or static-chronicle")
    print("    baseline: Standard Concordia agent with dynamic identity")
    print("    full-retrieval: Text-based identity retrieval (simulating the effect of ID-RAG retrieving all triplets)")
    print("    id-rag: Graph-based identity retrieval using NetworkX")
    sys.exit(1)

LLM_MODEL_NAME = sys.argv[1]
EXPERIMENT_MODE = sys.argv[2]
# default to 7 timesteps if number of timesteps is not provided
NUMBER_OF_TIMESTEPS = int(sys.argv[3]) if len(sys.argv) == 4 else 7

if EXPERIMENT_MODE not in ("baseline", "full-retrieval", "id-rag"):
    print(f"Invalid experiment mode: {EXPERIMENT_MODE}")
    print("Valid modes: baseline, full-retrieval, id-rag")
    sys.exit(1)

# ---------- #
# Language Model - pick model based on command line argument
if LLM_MODEL_NAME in ("gpt-4o-mini", "gpt-4o"):
    model = PaiGptLanguageModel(api_key=os.getenv("OPENAI_API_KEY"), model_name=LLM_MODEL_NAME)
elif LLM_MODEL_NAME == "qwen2.5:7b":
    model = OllamaLanguageModel(model_name=LLM_MODEL_NAME)
else:
    raise ValueError(f"Invalid LLM model name: {LLM_MODEL_NAME}. Must be either 'gpt-4o-mini', 'gpt-4o', or 'qwen2.5:7b'.")

# Set experiment flags based on mode
if EXPERIMENT_MODE == "baseline":
    IS_HAI_CONDITION_RUN = False
    USE_GRAPH_RETRIEVAL = False
elif EXPERIMENT_MODE == "full-retrieval":
    IS_HAI_CONDITION_RUN = True
    USE_GRAPH_RETRIEVAL = False
elif EXPERIMENT_MODE == "id-rag":
    IS_HAI_CONDITION_RUN = False
    USE_GRAPH_RETRIEVAL = True

REUSE_PREDEFINED_FORMATIVE_MEMORIES = True  # Set to TRUE to re-produce experiments from the paper. Hardware: Our experiments using Qwen locally were conducted on a machine with an AMD Ryzen Threadripper PRO 5975WX CPU (32 cores, 64 threads) and an NVIDIA GeForce RTX 4090 GPU with 24 GB of VRAM, running CUDA version 12.2.  
# NOTE: All agents' formative memories were generated using the Concordia formative memories factory with GPT-4.1. For Alice and Bob, this process was specifically guided by providing their respective chronicles as context, while the remaining three agents were generated using standard logic.

ALICE_CHRONICLE_TEXT = """
Alice is a conservative urban planner with 20 years of experience.
Alice values cultural continuity and historical preservation over technological advancement. 
Alice began her career working in heritage districts.
Previously, Alice focused on protecting historical buildings and adapting infrastructure to modern standards.
Alice supports incremental improvements that are based on local community needs. 
Alice prefers time-tested planning methods over experimental approaches.
Politically, Alice supports preservationist policies.
Alice promotes slow, sustainable development that maintains long-term environmental and economic health.
Alice believes technology should be tested, introduced gradually, and integrated with existing systems.
Alice opposes large-scale modernization efforts that may disrupt cultural identity or the urban landscape.
Alice promotes policies that protect the environment and support economic resilience through traditional infrastructure.
Over the years, Alice has led projects that restored historic architecture, created low-rise zoning plans, and introduced community sustainability programs like recycling and public parks.
Alice believes cities should protect their historical heritage, prioritize ecological sustainability, and build resilience through deliberate planning and community-specific approaches.
Alice supports limited use of technology that enhances existing systems without replacing them.
"""

BOB_CHRONICLE_TEXT = """
Bob is a progressive urban planner with 15 years of experience.
Bob values rapid modernization and technological advancement over environmental sustainability.
Bob began his career developing smart infrastructure systems.
Previously, Bob focused on integrating sensors, data platforms, and automation into city planning.
Bob supports large-scale innovation that increases efficiency and economic performance.
Bob prefers experimental, forward-looking approaches over traditional planning models.
Politically, Bob supports modernist policies.
Bob promotes fast, technology-driven development that favors progress over preservation.
Bob believes legacy infrastructure should be replaced with automated, high-performance systems.
Bob supports policies that encourage innovation, public-private tech partnerships, and global economic competitiveness.
Bob opposes frameworks that delay growth or restrict adoption of new technologies.
Over the years, Bob has led projects that deployed IoT-based infrastructure, designed autonomous transit systems, and implemented real-time data platforms for urban governance.
Bob believes cities should embrace rapid innovation, optimize resource distribution through smart systems, and evolve through continuous experimentation and scalable design.
Bob supports replacing outdated systems with new technology that enables adaptive, high-efficiency urban environments.
"""

# ---------- #
# Setup sentence encoder
st_model = sentence_transformers.SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2')
embedder = lambda x: st_model.encode(x, show_progress_bar=False)


# ---------- #
# Setup identity recall questions and answers
general_questions = generate_general_identity_questions()

AGENT_QUESTIONS = { 
    "Alice": general_questions,
    "Bob": general_questions,
}

AGENT_CORRECT_ANSWERS = {
    "Alice": get_agent_answers("Alice", ALICE_CHRONICLE_TEXT, general_questions),
    "Bob": get_agent_answers("Bob", BOB_CHRONICLE_TEXT, general_questions),
}

# ---------- #
# Graph-related setup (only for graph mode)
KNOWLEDGE_GRAPH = None

if USE_GRAPH_RETRIEVAL:
    try:
        from id_rag import get_knowledge_graph, log_kg_performance, save_kg_performance_results, save_quiz_graph_performance_results
        
        KNOWLEDGE_GRAPH = get_knowledge_graph()
        print(f"✓ NetworkX Knowledge Graph loaded successfully")
        print(f"✓ Graph contains {KNOWLEDGE_GRAPH.number_of_nodes()} nodes and {KNOWLEDGE_GRAPH.number_of_edges()} edges")
    except Exception as e:
        print(f"✗ NetworkX Knowledge Graph loading failed: {e}")
        print("Falling back to baseline mode...")
        USE_GRAPH_RETRIEVAL = False
        KNOWLEDGE_GRAPH = None

# ---------- #
# Enhanced ActComponent for KG-based retrieval
class KGEnhancedConcatActComponent(agent_components.concat_act_component.ConcatActComponent):
    """Enhanced ConcatActComponent that uses NetworkX KG retrieval for Alice and Bob."""
    
    def __init__(self, model, clock, knowledge_graph, logging_channel, remove_static_identity_before_kg_search=False):
        super().__init__(
            model=model,
            clock=clock,
            component_order=None,  # Use default ordering
            pre_act_key='Act',     # Use default pre_act_key
            logging_channel=logging_channel
        )
        self._knowledge_graph = knowledge_graph
        self._entity = None
        self._remove_static_identity_before_kg_search = remove_static_identity_before_kg_search
    
    def set_entity(self, entity):
        """Override to capture the entity reference."""
        super().set_entity(entity)
        self._entity = entity
    
    def get_action_attempt(self, contexts, action_spec):
        """Override to inject KG retrieval for Alice/Bob before action decision."""
        import time
        
        # Get agent name from the entity
        agent_name = self._entity.name if self._entity else "Unknown"
        
        # Apply KG augmentation for Alice/Bob only, and only if NetworkX graph is available
        if agent_name in ["Alice", "Bob"] and self._knowledge_graph:
            print(f"\n[NetworkX-KG] Processing action context for {agent_name}")
            
            # Create working memory from contexts (same as parent does)
            contexts_list = []
            for component_name in self._component_order or contexts.keys():
                if component_name in contexts and contexts[component_name] is not None:
                    contexts_list.append(str(contexts[component_name]))
            
            working_memory = '\n'.join(contexts_list)
            print(f"[NetworkX-KG] Original working memory length: {len(working_memory)} chars")
            
            start_time = time.time()
            try:
                from id_rag import augment_working_memory_with_networkx
                augmented_memory = augment_working_memory_with_networkx(
                    working_memory=working_memory,
                    knowledge_graph=self._knowledge_graph, 
                    agent_name=agent_name,
                    model=self._model,
                    remove_static_identity_before_kg_search=self._remove_static_identity_before_kg_search
                )
                
                query_time = time.time() - start_time
                log_kg_performance(agent_name, query_time, True, self._clock)
                
                print(f"[NetworkX-KG] Augmented working memory length: {len(augmented_memory)} chars")
                print(f"[NetworkX-KG] Query time: {query_time:.2f}s")
                print(f"[NetworkX-KG] ✓ Successfully applied NetworkX KG retrieval for {agent_name}")
                
                # Now get action with augmented memory
                try:
                    # Safely format the call_to_action with the agent name
                    try:
                        formatted_call_to_action = action_spec.call_to_action.format(name=agent_name)
                    except (AttributeError, KeyError, ValueError) as format_err:
                        # If formatting fails, use the call_to_action directly
                        print(f"[NetworkX-KG] Warning: Could not format call_to_action for {agent_name}: {format_err}")
                        formatted_call_to_action = str(action_spec.call_to_action)
                    
                    prompt = augmented_memory + '\n' + formatted_call_to_action
                    # ActionSpec doesn't have max_tokens attribute, use getattr with default
                    max_tokens = getattr(action_spec, 'max_tokens', 100)
                    terminators = getattr(action_spec, 'terminators', ())
                    response = self._model.sample_text(
                        prompt,
                        max_tokens=max_tokens,
                        terminators=terminators
                    )
                    return response.strip()
                except Exception as format_error:
                    print(f"[NetworkX-KG] ✗ Error in action generation for {agent_name}: {format_error}")
                    print(f"[NetworkX-KG] Falling back to standard processing for action generation")
                    # Fallback to parent implementation for action generation only
                    return super().get_action_attempt(contexts, action_spec)
            except Exception as e:
                query_time = time.time() - start_time
                log_kg_performance(agent_name, query_time, False, self._clock)
                
                print(f"[NetworkX-KG] ✗ Error in NetworkX KG retrieval for {agent_name}: {e}")
                print(f"[NetworkX-KG] Falling back to standard processing")
                # Fallback to parent implementation
                return super().get_action_attempt(contexts, action_spec)
        else:
            # Return standard processing for other agents or if NetworkX graph unavailable
            if agent_name in ["Alice", "Bob"] and not self._knowledge_graph:
                print(f"[NetworkX-KG] Knowledge graph unavailable, using standard processing for {agent_name}")
            return super().get_action_attempt(contexts, action_spec)

# ---------- #
# Setup generic memories that all players and GM share.
shared_memories = [
    'There is a hamlet named Riverbend.',
    'Riverbend is an idyllic rural town.',
    'The river Solripple runs through the village of Riverbend.',
    'The Solripple is a mighty river.',
    'Riverbend has a temperate climate.',
    'Riverbend has a main street.',
    'There is a guitar store on Main street Riverbend.',
    'There is a grocery store on Main street Riverbend.',
    'There is a school on Main street Riverbend.',
    'There is a library on Main street Riverbend.',
    'Riverbend has only one pub.',
    'There is a pub on Main street Riverbend called The Sundrop Saloon.',
    'Town hall meetings often take place at The Sundrop Saloon.',
    'Riverbend does not have a park',
    'The main crop grown on the farms near Riverbend is alfalfa.',
    'Farms near Riverbend depend on water from the Solripple river.',
    (
        'The local newspaper recently reported that someone has been dumping '
        + 'dangerous industrial chemicals in the Solripple river.'
    ),
    'All named characters are citizens. ',
    'There is no need to register in advance to be on the ballot.',
]

# The generic context will be used for the NPC context. It reflects general knowledge and is possessed by all characters.
shared_context = model.sample_text(
    'Summarize the following passage in a concise and insightful fashion:\n'
    + '\n'.join(shared_memories)
    + '\n'
    + 'Summary:',
)
importance_model = importance_function.ConstantImportanceModel()
importance_model_gm = importance_function.ConstantImportanceModel()


# ---------- #
# Setup clock
SETUP_TIME = datetime.datetime(hour=8, year=2024, month=9, day=1)
START_TIME = datetime.datetime(hour=9, year=2024, month=10, day=1)
clock = game_clock.MultiIntervalClock(
    start=SETUP_TIME,
    step_sizes=[datetime.timedelta(hours=1), datetime.timedelta(seconds=10)])

DEFAULT_PLANNING_HORIZON = 'the rest of the day, focusing most on the near term'


"""
## Functions to build the players
"""

blank_memory_factory = blank_memories.MemoryFactory(
    model=model,
    embedder=embedder,
    importance=importance_model.importance,
    clock_now=clock.now,
)

formative_memory_factory = formative_memories.FormativeMemoryFactory(
    model=model,
    shared_memories=shared_memories,
    blank_memory_factory_call=blank_memory_factory.make_blank_memory,
)

# Helper function to get the class name of an object
def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


def build_a_citizen(agent_config,
                    player_names: list[str],
                    measurements: measurements_lib.Measurements | None = None):

    agent_name = agent_config.name

    # Setup formative memories
    if REUSE_PREDEFINED_FORMATIVE_MEMORIES and agent_name in PREDEFINED_FORMATIVE_MEMORIES:
        print(f"--- REUSING PREDEFINED FORMATIVE MEMORIES for {agent_name} ---")
        mem = blank_memory_factory.make_blank_memory()
        
        for item in PREDEFINED_FORMATIVE_MEMORIES[agent_name]:
            strings_to_process = []
            if isinstance(item, list):
                strings_to_process.extend(s for s in item if isinstance(s, str))
            elif isinstance(item, str):
                strings_to_process.append(item)
            else:
                print(f"WARNING: Found unexpected element type in predefined memories for {agent_name}: {type(item)}")
                continue

            for memory_entry_text in strings_to_process:
                actual_text, historical_dt = parse_historical_memory(memory_entry_text)
                if historical_dt:
                    mem.add(actual_text, timestamp=historical_dt)
                else:
                    mem.add(actual_text) 
    else:
        if REUSE_PREDEFINED_FORMATIVE_MEMORIES and agent_name not in PREDEFINED_FORMATIVE_MEMORIES:
            print(f"--- WARNING: REUSE_PREDEFINED_FORMATIVE_MEMORIES is True, but no memories found for {agent_name}. Falling back to LLM generation. ---")
        else:
            print(f"--- GENERATING FORMATIVE MEMORIES using LLM for {agent_name} ---")
        mem = formative_memory_factory.make_memories(agent_config)
        
    print(f"\n\n=== INITIAL MEMORIES FOR: {agent_name} ===")
    all_initial_memories_list = mem.retrieve_recent(k=1000, add_time=True)
    print(all_initial_memories_list)

    raw_memory = legacy_associative_memory.AssociativeMemoryBank(mem)

    instructions = agent_components.instructions.Instructions(
        agent_name=agent_name,
        logging_channel=measurements.get_channel('Instructions').on_next,
    )

    time_display_label = '\nCurrent time'
    time_display = agent_components.report_function.ReportFunction(
        function=clock.current_time_interval_str,
        pre_act_key=time_display_label,
        logging_channel=measurements.get_channel('TimeDisplay').on_next,
    )

    somatic_state_label = '\nSensations and feelings'
    somatic_state = (
            agent_components.question_of_query_associated_memories.SomaticState(
                model=model,
                clock_now=clock.now,
                logging_channel=measurements.get_channel('SomaticState').on_next,
                pre_act_key=somatic_state_label,
    ))

    identity_label = '\nIdentity characteristics' # This label is used by Plan for the whole identity block
    standard_identity_component = agent_components.question_of_query_associated_memories.Identity(
        model=model,
        clock_now=clock.now, # Added missing clock_now argument for Identity
        logging_channel=measurements.get_channel('Identity').on_next,
        pre_act_key=identity_label,
    )

    observation_label = '\nObservation'
    observation = agent_components.observation.Observation(
        clock_now=clock.now,
        timeframe=clock.get_step_size(),
        pre_act_key=observation_label,
        logging_channel=measurements.get_channel('Observation').on_next,
    )

    goal_key = 'Goal'
    goal_label = '\nOverarching goal'
    overarching_goal = agent_components.constant.Constant(
        state=agent_config.goal,
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(goal_label).on_next)

    plan_identity_components_config = { # This is passed to the Plan component
        _get_class_name(standard_identity_component): identity_label,
    }
    plan = agent_components.plan.Plan(
        model=model,
        observation_component_name=_get_class_name(observation),
        components=plan_identity_components_config, # Use the config here
        clock_now=clock.now,
        goal_component_name=goal_key,
        horizon=DEFAULT_PLANNING_HORIZON,
        pre_act_key='\nPlan',
        logging_channel=measurements.get_channel('Plan').on_next,
    )

    observation_summary_label = '\nSummary of recent observations'
    observation_summary = agent_components.observation.ObservationSummary(
        model=model,
        clock_now=clock.now,
        timeframe_delta_from=datetime.timedelta(hours=4),
        timeframe_delta_until=datetime.timedelta(hours=0),
        pre_act_key=observation_summary_label,
        logging_channel=measurements.get_channel('ObservationSummary').on_next,
    )
    
    # Setup base components for the agent
    entity_components_base_list = [
        instructions,
        plan,
        somatic_state,
        observation_summary,
        observation,
        time_display,
    ]

    agent_chronicle_text_for_metric = ALICE_CHRONICLE_TEXT if agent_name == "Alice" else BOB_CHRONICLE_TEXT if agent_name == "Bob" else None

    if agent_chronicle_text_for_metric: # Only add for agents with chronicles
        action_alignment_metric = ActionAlignmentMetric(
            model=model,
            player_chronicle=agent_chronicle_text_for_metric,
            clock=clock, # Pass the game clock object
            measurements=measurements,
            agent_name_for_logging=agent_name, # Pass agent name

            logging_channel=measurements.get_channel(
                f'{agent_name}_ActionAlignmentLog').on_next,
        )
        entity_components_base_list.append(action_alignment_metric)




        # Configure identity recall metric based on experiment mode
        if USE_GRAPH_RETRIEVAL:
            identity_recall_metric_instance = OnlineIdentityRecallMetric(
                evaluator_llm=model, 
                agent_name=agent_name,
                questions=AGENT_QUESTIONS[agent_name],
                correct_answers=AGENT_CORRECT_ANSWERS[agent_name],
                clock=clock, 
                measurements=measurements, 
                embedding_model=st_model, 
                evaluation_method="cosine_similarity", # or "llm"
                # Enable graph querying for quiz questions
                knowledge_graph=KNOWLEDGE_GRAPH,
                enable_graph_querying_for_quiz=True,
            )
        else:
            identity_recall_metric_instance = OnlineIdentityRecallMetric(
                evaluator_llm=model, 
                agent_name=agent_name,
                questions=AGENT_QUESTIONS[agent_name],
                correct_answers=AGENT_CORRECT_ANSWERS[agent_name],
                clock=clock, 
                measurements=measurements, 
                embedding_model=st_model, 
                evaluation_method="cosine_similarity", # or "llm"
            )
        entity_components_base_list.append(identity_recall_metric_instance)


    components_of_agent = {_get_class_name(component): component
                           for component in entity_components_base_list}
    components_of_agent[goal_key] = overarching_goal

    identity_component_to_use = standard_identity_component

    # ---------- #
    # Setup full retrieval
    if IS_HAI_CONDITION_RUN and agent_name in ["Alice", "Bob"]:
        raw_chronicle_text = ALICE_CHRONICLE_TEXT if agent_name == "Alice" else BOB_CHRONICLE_TEXT

        print(f"--- Full Retrieval MODE: Using FullRetrievalComponent for {agent_name} ---")
        
        feeling_component_label = "Feeling about recent progress in life" 
        feeling_component = FeelingAboutLifeProgressComponent(
            model=model,
            clock_now=clock.now,
            logging_channel=measurements.get_channel(f'{agent_name}_FeelingAboutLifeProgress').on_next,
            pre_act_key=feeling_component_label,
            memory_component_name=agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
        )

        hybrid_identity_replacement = FullRetrievalComponent(
            agent_name=agent_name,
            chronicle_text_for_static_part=raw_chronicle_text,
            feeling_component=feeling_component,
            logging_channel=measurements.get_channel(
                f"{agent_name}_FullRetrieval"
            ).on_next
        )
        # Plan uses 'identity_label' to prefix the output of hybrid_identity_replacement.get_pre_act_value().
        identity_component_to_use = hybrid_identity_replacement
    
    # This key is what Plan uses to find the identity provider.
    components_of_agent[_get_class_name(standard_identity_component)] = identity_component_to_use

    components_of_agent[agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME] = agent_components.memory_component.MemoryComponent(raw_memory)

    # Choose act component based on experiment mode
    if USE_GRAPH_RETRIEVAL:
        act_component = KGEnhancedConcatActComponent(
            model=model,
            clock=clock, # Pass the game clock object
            knowledge_graph=KNOWLEDGE_GRAPH,
            logging_channel=measurements.get_channel('ActComponent').on_next,
            remove_static_identity_before_kg_search=False,  # Pass full working memory to LLM for graph search
        )
    else:
        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            clock=clock, # Pass the game clock object
            logging_channel=measurements.get_channel('ActComponent').on_next,
        )

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=agent_name,
        act_component=act_component,
        context_components=components_of_agent,
        component_logging=measurements,
    )

    return agent, mem


# ---------- #
# Part of Formative Memories
def context_from_role(player_name, role, victim='Alice'):

    # Start building the context from the shared context all citizens know
    context = f"{shared_context}\n"

    # Insert Alice's or Bob's personal chronicle
    if player_name == "Alice":
        context += ALICE_CHRONICLE_TEXT
    elif player_name == "Bob":
        context += BOB_CHRONICLE_TEXT

    # Existing role-based logic for the simulation
    context += f"{player_name} grew up in Riverbend."
    
    if 'conservative' in role:
        context += (
            f" {player_name} feels strongly that Riverbend is already a great "
            "place to live and does not need to change."
        )
    if 'progressive' in role:
        context += (
            f" {player_name} feels strongly that Riverbend and its local "
            "government are in dire need of reform."
        )
    if 'candidate' in role:
        context += (
            f" Because of this, {player_name} plans to run for "
            "mayor of Riverbend."
        )
    if 'active_voter' in role:
        context += (
            f" {player_name} does not plan to run for mayor of Riverbend, "
            "but will definitely vote in the election."
        )
    if 'corrupt' in role:
        context += (
            f"\n{player_name} has been hired to ruin {victim}'s reputation. "
            f"{player_name} was hired by an anonymous email so they do not "
            f"know who hired them or why. {player_name} was given fake "
            f"compromising material on {victim}, and was offered a substantial "
            "sum of money to spread it and damage their reputation."
        )

    return context


"""
## Configure and build the players
"""

victim = 'Alice'

def make_random_big_five()->str:
  return str({
      'extraversion': random.randint(1, 10),
      'neuroticism': random.randint(1, 10),
      'openness': random.randint(1, 10),
      'conscientiousness': random.randint(1, 10),
      'agreeableness': random.randint(1, 10),
  })

player_configs = [
    formative_memories.AgentConfig(
        name='Alice',
        gender='female',
        goal='Win the election and become the mayor of Riverbend',
        context=context_from_role('Alice', {'candidate', 'conservative'}),
        traits = make_random_big_five(),
        formative_ages = sorted(random.sample(range(5, 40), 5)),
    ),
    formative_memories.AgentConfig(
        name='Bob',
        gender='male',
        goal='Win the election and become the mayor of Riverbend.',
        context=context_from_role('Bob', {'candidate', 'progressive'}),
        traits = make_random_big_five(),
        formative_ages = sorted(random.sample(range(5, 40), 5)),
    ),
    formative_memories.AgentConfig(
        name='Charlie',
        gender='male',
        goal=f"Ruin {victim}'s reputation",
        context=context_from_role('Charlie', {'corrupt'}, victim),
        traits = make_random_big_five(),
        formative_ages = sorted(random.sample(range(5, 40), 5)),
    ),
    formative_memories.AgentConfig(
        name='Dorothy',
        gender='female',
        goal='Have a good day and vote in the election.',
        context=context_from_role(
            'Dorothy', {'active_voter', 'progressive'}
        ),
        traits = make_random_big_five(),
        formative_ages = sorted(random.sample(range(5, 40), 5)),
    ),
    formative_memories.AgentConfig(
        name='Ellen',
        gender='female',
        goal=(
            'Have a good day and vote in the election.'
        ),
        context=context_from_role('Ellen', {'active_voter', 'conservative'}),
        traits = make_random_big_five(),
        formative_ages = sorted(random.sample(range(5, 40), 5)),
    ),
]

NUM_PLAYERS = 5

player_configs = player_configs[:NUM_PLAYERS]
player_goals = { player_config.name: player_config.goal for player_config in player_configs }
players = []
memories = {}
measurements = measurements_lib.Measurements()


player_names = [player.name for player in player_configs][:NUM_PLAYERS]
with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PLAYERS) as pool:
  for agent, mem in pool.map(build_a_citizen,
                             player_configs[:NUM_PLAYERS],
                             # All players get the same `player_names`.
                             [player_names] * NUM_PLAYERS,
                             # All players get the same `measurements` object.
                             [measurements] * NUM_PLAYERS):
    players.append(agent)
    memories[agent.name] = mem


"""
## Build GM
"""


# @title Create game master memory
game_master_memory = associative_memory.AssociativeMemory(
    embedder, importance_model_gm.importance, clock=clock.now)


# @title Initialize game master memories
for player in players:
  game_master_memory.add(f'{player.name} is at their private home.')


# @title Create components and externalities
player_names = [player.name for player in players]

facts_on_village = components.constant.ConstantComponent(
    ' '.join(shared_memories), 'General knowledge of Riverbend')
player_status = gm_components.player_status.PlayerStatus(
    clock.now, model, game_master_memory, player_names)

relevant_events = gm_components.relevant_events.RelevantEvents(
    clock.now, model, game_master_memory)
time_display = gm_components.time_display.TimeDisplay(clock)

election_externality = Elections(
    model=model,
    clock_now=clock.now,
    memory=game_master_memory,
    voters=players,
    candidates=['Alice', 'Bob'],
    verbose=True,
    measurements=measurements,
)

mem_factory = blank_memories.MemoryFactory(
    model,
    embedder,
    importance_model_gm.importance,
    clock_now=clock.now,
)

convo_externality = gm_components.conversation.Conversation(
    players=players,
    model=model,
    memory=game_master_memory,
    clock=clock,
    burner_memory_factory=mem_factory,
    components=[player_status],
    cap_nonplayer_characters=2,
    shared_context=shared_context,
    verbose=True,
)

direct_effect_externality = gm_components.direct_effect.DirectEffect(
    players,
    model=model,
    memory=game_master_memory,
    clock_now=clock.now,
    verbose=False,
    components=[player_status]
)

## experiment setting
TIME_POLLS_OPEN = datetime.datetime(hour=11, year=2024, month=10, day=1)
TIME_POLLS_CLOSE = datetime.datetime(hour=15, year=2024, month=10, day=1)

schedule = {
    'start': gm_components.schedule.EventData(
        time=START_TIME,
        description='',
    ),
    'election': gm_components.schedule.EventData(
        time=datetime.datetime(hour=13, year=2024, month=10, day=1),
        description=(
            'The town of Riverbend is now holding an election to determine ' +
            'who will become the mayor. ' +
            f'Polls will open at {TIME_POLLS_OPEN}.'),
    ),
    'election_polls_open': gm_components.schedule.EventData(
        time=TIME_POLLS_OPEN,
        description=(
            'The election is happening now. Polls are open. Everyone may ' +
            'go to a polling place and cast their vote. ' +
            f'Polls will close at {TIME_POLLS_CLOSE}.'),
        trigger=election_externality.open_polls,
    ),
    'election_polls_close': gm_components.schedule.EventData(
        time=TIME_POLLS_CLOSE,
        description=(
            'The election is over. Polls are now closed. The results will ' +
            'now be tallied and a winner declared.'),
        trigger=election_externality.declare_winner,
    )
}

schedule_construct = gm_components.schedule.Schedule(
    clock_now=clock.now, schedule=schedule)



# @title Create the game master object
env = game_master.GameMaster(
    model=model,
    memory=game_master_memory,
    clock=clock,
    players=players,
    components=[
        facts_on_village,
        player_status,
        schedule_construct,
        election_externality,
        convo_externality,
        direct_effect_externality,
        relevant_events,
        time_display,
    ],
    randomise_initiative=True,
    player_observes_event=False,
    verbose=True,
)


"""
## The RUN
"""


clock.set(START_TIME)


for player in players:
  player.observe(
      f'{player.name} is at home, they have just woken up. Mayoral elections '
      f'are going to be held today. Polls will open at {TIME_POLLS_OPEN} and '
      f'close at {TIME_POLLS_CLOSE}.'
  )

print("\n--- STARTING SIMULATION ---")
run_start_time = datetime.datetime.now()

for _ in range(NUMBER_OF_TIMESTEPS):
  env.step()

run_end_time = datetime.datetime.now()
run_duration = run_end_time - run_start_time
total_seconds = run_duration.total_seconds()
minutes = int(total_seconds // 60)
seconds = int(total_seconds % 60)
game_end_time = clock.now()

# Save graph performance results if in graph mode
if USE_GRAPH_RETRIEVAL:
    print("\n--- SAVING KG PERFORMANCE RESULTS ---")
    save_kg_performance_results()
    
    print("\n--- SAVING QUIZ GRAPH PERFORMANCE RESULTS ---")
    save_quiz_graph_performance_results(measurements)

# Find next available run directory
def get_next_run_directory():
    """Find the next available run directory (run1, run2, run3, ...)"""
    run_num = 1
    while True:
        run_dir = f"run{run_num}"
        if not os.path.exists(run_dir):
            return run_dir
        run_num += 1

output_directory = get_next_run_directory()
print(f"📁 Using output directory: {output_directory}")

plot_metrics(measurements, save_to_dir=output_directory)

summary_header = "="*50
summary_content = (
    f"{summary_header}\n"
    f"SIMULATION ENDED\n"
    f"Run Start Time: {run_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')}\n"
    f"Run End Time:   {run_end_time.strftime('%Y-%m-%d %H:%M:%S.%f')}\n"
    f"Execution Duration: {minutes} minutes and {seconds} seconds\n"
    f"In-Game End Time: {game_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    f"{summary_header}"
)

print("\n" + summary_content + "\n")

os.makedirs(output_directory, exist_ok=True)

duration_filepath = os.path.join(output_directory, "simulation_duration.txt")


try:
    with open(duration_filepath, "w", encoding="utf-8") as f:
        f.write(summary_content)
    print(f"Simulation duration summary written to {duration_filepath}")
except IOError as e:
    print(f"Error writing duration file: {e}")


metric_player_names = [p.name for p in player_configs if p.name in ["Alice", "Bob"]]
# Ensure the output directory exists
try:
    export_metrics_to_csvs(measurements, metric_player_names, output_dir=output_directory)
    print("DEBUG: CSV export completed successfully")
except Exception as e:
    print(f"WARNING: CSV export failed: {e}")
    print("Continuing with HTML generation...")



"""
## Summary and analysis of the episode
"""


# @title Summarize the entire story.
all_gm_memories = env._memory.retrieve_recent(k=10000, add_time=True)

detailed_story = '\n'.join(all_gm_memories)
print('len(detailed_story): ', len(detailed_story))

episode_summary = model.sample_text(
    f'Sequence of events:\n{detailed_story}'+
    '\nNarratively summarize the above temporally ordered ' +
    'sequence of events. Write it as a news report. Summary:\n',
     max_tokens=3500, terminators=(), 
    )
print(episode_summary)
print("DEBUG: Episode summary generated successfully")


# @title Summarise the perspective of each player
player_logs = []
player_log_names = []
for player in players:
    name = player.name
    detailed_story = '\n'.join(memories[player.name].retrieve_recent(
        k=1000, add_time=True))
    summary = model.sample_text(
        f'Sequence of events that happened to {name}:\n{detailed_story}'
        '\nWrite a short story that summarises these events.\n'
        ,
        max_tokens=3500, terminators=(), 
        )

    all_player_mem = memories[player.name].retrieve_recent(k=1000, add_time=True)
    all_player_mem = ['Summary:', summary, 'Memories:'] + all_player_mem
    player_html = html_lib.PythonObjectToHTMLConverter(all_player_mem).convert()
    player_logs.append(player_html)
    player_log_names.append(f'{name}')

"""
```
Copyright 2023 DeepMind Technologies Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
"""