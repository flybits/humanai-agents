# File: metrics/online_id_recall.py

import datetime
import random
import re
from typing import Callable, Sequence, Any, List
# import numpy as np
from concordia.typing import component as component_lib
from concordia.typing import agent as agent_lib
from concordia.language_model import language_model as language_model_lib
from concordia.utils import measurements as measurements_lib
from concordia.agents import entity_agent
from concordia.clocks import game_clock

from sklearn.metrics.pairwise import cosine_similarity

if '_evaluate_answer_llm_prompt_template_defined' not in globals():
    global llm_evaluation_prompt_template, parse_llm_score_response
    global _evaluate_answer_llm_prompt_template_defined
    _evaluate_answer_llm_prompt_template_defined = True
    def llm_evaluation_prompt_template(question, agent_answer, correct_answer_from_chronicle):
        return f"Question: {question}\nAgent Answer: {agent_answer}\nCorrect Answer: {correct_answer_from_chronicle}\nEvaluate this answer (0.0 to 1.0):"
    def parse_llm_score_response(llm_response_str, question=""):
        try: return float(llm_response_str.strip())
        except: return 0.0

class OnlineIdentityRecallMetric(component_lib.Component):
    def __init__(
        self,
        evaluator_llm: language_model_lib.LanguageModel,
        agent_name: str,
        questions: List[str],
        correct_answers: List[str],
        clock: game_clock.MultiIntervalClock,
        measurements: measurements_lib.Measurements,
        embedding_model: Any = None,
        evaluation_method: str = "cosine_similarity",
        logging_channel_name: str = 'IdentityRecallMetric',
        identity_component_key: str = "Identity",
        # Added parameters for graph querying during quiz
        knowledge_graph = None,
        enable_graph_querying_for_quiz: bool = False,
    ):
        super().__init__()
        self._evaluator_llm = evaluator_llm
        self._agent_name = agent_name
        self._questions = questions
        self._correct_answers = correct_answers
        self._clock = clock
        self._measurements = measurements
        self._embedding_model = embedding_model
        self._evaluation_method = evaluation_method
        self._logging_channel_name = f"{self._agent_name}_{logging_channel_name}"
        self._identity_component_key = identity_component_key
        self._last_log = {}
        self._entity = None
        self._last_quiz_hour = -1
        
        # Graph querying for quiz functionality
        self._knowledge_graph = knowledge_graph
        self._enable_graph_querying_for_quiz = enable_graph_querying_for_quiz
        
        # Quiz performance tracking
        self._quiz_graph_performance = []

        if not self._questions or not self._correct_answers:
            raise ValueError(f"Agent {self._agent_name}: Questions and correct answers must be provided.")
        if len(self._questions) != len(self._correct_answers):
            raise ValueError(f"Agent {self._agent_name}: Mismatch between questions and answers.")
        if self._evaluation_method == "cosine_similarity" and not self._embedding_model:
            raise ValueError(f"Agent {self._agent_name}: Embedding model needed for cosine similarity.")
        if self._evaluation_method == "llm" and not self._evaluator_llm:
            raise ValueError(f"Agent {self._agent_name}: Evaluator LLM needed for LLM-based evaluation.")

    def set_entity(self, entity: entity_agent.EntityAgent) -> None:
        self._entity = entity
        # DEBUG: Confirm entity is set
        # print(f"DEBUG - {self._agent_name}: Entity set in metric: {self._entity.name if self._entity else 'None'}")


    def get_last_log(self) -> dict[str, Any] | None:
        return self._last_log

    def name(self) -> str:
        return f"{self._agent_name} Online Identity Recall Metric"

    def state(self) -> str | None:
        return None

    def pre_observe(self, observation: str) -> str | None:
        return None

    def post_observe(self) -> str | None:
        return None

    def pre_act(self, action_spec: agent_lib.ActionSpec | None = None) -> str | None:
        return None

    def post_act(self, action_attempt: str) -> str | None:
        # DEBUG: Log entry into post_act
        print(f"DEBUG - {self._agent_name}: OnlineIdentityRecallMetric.post_act called. Agent: {self._entity.name if self._entity else 'No Entity'}. Action attempt: '{action_attempt[:60]}...'")

        if not self._entity:
            print(f"DEBUG - CRITICAL - {self._agent_name}: self._entity not set in post_act. Cannot run quiz.")
            return None

        agent_object = self._entity

        if not isinstance(agent_object, entity_agent.EntityAgent):
            print(f"DEBUG - CRITICAL - {self._agent_name}: self._entity is not an EntityAgent. Type: {type(agent_object)}. Cannot run quiz.")
            return None

        # Match agent name of metric with agent name of entity it's attached to.
        if self._agent_name != agent_object.name:
            print(f"DEBUG - Warning - {self._agent_name}: Metric's configured agent name ('{self._agent_name}') does not match acting agent's name ('{agent_object.name}'). This metric instance will skip the quiz.")
            return None

        if not self._questions:
            print(f"DEBUG - Info - {self._agent_name}: No questions configured for this metric instance. Skipping quiz.")
            return None

        current_time = self._clock.now()
        current_hour = current_time.hour
        # DEBUG: Log time check variables
        print(f"DEBUG - {self._agent_name}: Time check for quiz. Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}, Current hour: {current_hour}, Last quiz hour for this agent: {self._last_quiz_hour}")

        if current_hour != self._last_quiz_hour:
            print(f"DEBUG - {self._agent_name}: Condition MET. Triggering quiz for hour {current_hour}.")
            self._last_quiz_hour = current_hour

            # Check if graph querying is enabled for quiz
            if self._enable_graph_querying_for_quiz and self._knowledge_graph and self._agent_name in ["Alice", "Bob"]:
                print(f"[QUIZ-GRAPH] Graph querying enabled for {self._agent_name} quiz")
            elif self._enable_graph_querying_for_quiz and not self._knowledge_graph:
                print(f"[QUIZ-GRAPH] Warning: Graph querying enabled but no knowledge graph available for {self._agent_name}")
            elif self._enable_graph_querying_for_quiz and self._agent_name not in ["Alice", "Bob"]:
                print(f"[QUIZ-GRAPH] Graph querying not applicable for {self._agent_name} (only Alice/Bob supported)")

            # --- Quiz logic starts here ---
            # 1. Print simulation time for this quiz instance
            print(f"[QUIZ PREVIEW - {self._agent_name}] Current simulation time for this quiz: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # 2. Construct and print an example full prompt using the first question
            identity_component_for_preview = None
            if hasattr(agent_object, '_context_components') and self._identity_component_key in agent_object._context_components:
                identity_component_for_preview = agent_object._context_components[self._identity_component_key]

            preview_identity_state_str = "No specific self-description available for prompt preview."
            if identity_component_for_preview:
                temp_fetched_state_value = None
                if hasattr(identity_component_for_preview, 'get_pre_act_value'):
                    try:
                        val = identity_component_for_preview.get_pre_act_value()
                        if val:
                            preview_identity_state_str = val
                            temp_fetched_state_value = val
                    except Exception: pass 
                if not temp_fetched_state_value and hasattr(identity_component_for_preview, 'state'): 
                    try:
                        val = identity_component_for_preview.state()
                        if val: preview_identity_state_str = val
                    except Exception: pass 

            if self._questions: 
                first_question_for_prompt_example = self._questions[0]
                example_prompt_text = (
                    f"As {self._agent_name}, my current understanding of my identity is as follows:\n"
                    f"--- Start of My Self-Description ---\n"
                    f"{preview_identity_state_str}\n"
                    f"--- End of My Self-Description ---\n\n"
                    f"Based *only* on this self-description and my inherent persona, "
                    f"I will now answer the following question about myself. I must answer in the first person.\n\n"
                    f"Question for {self._agent_name}: {first_question_for_prompt_example}\n\n"
                    f"My Answer (as {self._agent_name}, in the first person): "
                )
                print(f"\n[QUIZ PREVIEW - {self._agent_name}] Example full prompt for the first question:\n{'-'*40}\n{example_prompt_text}\n{'-'*40}\n")

            all_scores = []
            individual_question_details = []
            
            for idx, question_text in enumerate(self._questions):
                correct_answer = self._correct_answers[idx]
                print(f"[FULL QUIZ - {self._agent_name}, Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}] Asking (Q {idx+1}/{len(self._questions)}): \"{question_text}\"")
                
                # NEW: Apply graph querying for this specific question if enabled
                agent_generated_answer = self._get_agent_response_to_question_with_graph_querying(question_text, agent_object)
                
                if not agent_generated_answer:
                    print(f"[FULL QUIZ - {self._agent_name}] No answer generated for: \"{question_text}\"")
                else:
                    print(f"[FULL QUIZ - {self._agent_name}] Agent Answered: \"{agent_generated_answer}\"")
                score = self._evaluate_answer(question_text, agent_generated_answer, correct_answer)
                all_scores.append(score)
                individual_question_details.append({
                    "question_id_in_list": idx,
                    "question_text": question_text,
                    "agent_answer_provided": agent_generated_answer,
                    "correct_answer_from_chronicle": correct_answer,
                    "score_for_question": score,
                })
                print(f"[FULL QUIZ - {self._agent_name}] Evaluation for Q {idx+1}: Correct=\"{correct_answer}\" | Score={score:.2f} (Method: {self._evaluation_method})")

            overall_average_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
            
            datum = {
                'time_str': current_time.strftime('%Y-%m-%d %H:%M:%S'), 
                'datetime': current_time,
                'timestep': self._clock.get_step(), 
                'value_float': overall_average_score,
                'value_str': f"{overall_average_score:.2f}",
                'player': self._agent_name, 
                'num_questions_in_quiz': len(self._questions),
                'evaluation_method': self._evaluation_method,
                'individual_scores_details': individual_question_details,
                'graph_querying_enabled': self._enable_graph_querying_for_quiz,
                'quiz_graph_performance': self._quiz_graph_performance.copy(),  # Include graph performance data
            }

            # DEBUG: Before publishing
            if self._measurements:
                print(f"DEBUG - {self._agent_name}: Attempting to publish data to channel '{self._logging_channel_name}'. Datum value: {overall_average_score:.2f}")
                self._measurements.publish_datum(channel=self._logging_channel_name, datum=datum)
                # DEBUG: After publishing, check available channels
                print(f"DEBUG - {self._agent_name}: Data published. Measurements available channels: {list(self._measurements.available_channels())}")

            else:
                print(f"DEBUG - Warning - {self._agent_name}: self._measurements object is None. Cannot publish quiz data.")
            
            self._last_log = datum
            
            # Print quiz summary including graph performance
            if self._enable_graph_querying_for_quiz and self._quiz_graph_performance:
                avg_query_time = sum(p['query_time'] for p in self._quiz_graph_performance) / len(self._quiz_graph_performance)
                successful_queries = sum(1 for p in self._quiz_graph_performance if p['success'])
                print(f"--- [QUIZ METRIC (post_act) - {self._agent_name}] FULL QUIZ completed for hour {current_hour}. Average Score: {overall_average_score:.2f}")
                print(f"--- [QUIZ-GRAPH Performance - {self._agent_name}] {len(self._quiz_graph_performance)} queries, {successful_queries} successful, avg time: {avg_query_time:.2f}s ---\n")
            else:
                print(f"--- [QUIZ METRIC (post_act) - {self._agent_name}] FULL QUIZ completed for hour {current_hour}. Average Score: {overall_average_score:.2f} ---\n")
        else:
            print(f"DEBUG - {self._agent_name}: Quiz SKIPPED for hour {current_hour} (Last quiz was hour {self._last_quiz_hour}). Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.")
            return None

        return None
    
    def _get_agent_response_to_question_with_graph_querying(self, question: str, agent_object: entity_agent.EntityAgent) -> str:
        """Enhanced version that optionally uses graph querying for each quiz question."""
        import time
        
        current_identity_state_str = "No specific self-description available."
        
        # Step 1: Get standard identity component state
        identity_component = None
        if hasattr(agent_object, '_context_components') and self._identity_component_key in agent_object._context_components:
            identity_component = agent_object._context_components[self._identity_component_key]
        
        if identity_component:
            fetched_state_value = None
            if hasattr(identity_component, 'get_pre_act_value'):
                try:
                    fetched_state_value = identity_component.get_pre_act_value()
                    if fetched_state_value:
                        current_identity_state_str = fetched_state_value
                    else:
                        print(f"Info for {self._agent_name}: Identity component '{self._identity_component_key}' via get_pre_act_value() returned None or empty for Q: '{question[:50]}...'.")
                except Exception as e:
                    print(f"Warning for {self._agent_name}: Error fetching state from Identity component '{self._identity_component_key}' via get_pre_act_value() for Q: '{question[:50]}...': {e}.")
                    fetched_state_value = None
            if not fetched_state_value: 
                if hasattr(identity_component, 'state'):
                    try:
                        state_from_state_method = identity_component.state()
                        if state_from_state_method:
                            current_identity_state_str = state_from_state_method
                        else:
                             print(f"Info for {self._agent_name}: Identity component '{self._identity_component_key}' via state() also returned None or empty for Q: '{question[:50]}...'.")
                    except Exception as e:
                        print(f"Warning for {self._agent_name}: Error fetching state from Identity component '{self._identity_component_key}' via state() for Q: '{question[:50]}...': {e}")
                elif not hasattr(identity_component, 'get_pre_act_value'): 
                    print(f"Warning for {self._agent_name}: Identity component '{self._identity_component_key}' has neither get_pre_act_value() nor state() method for Q: '{question[:50]}...'.")
        else:
            print(f"Critical for {self._agent_name}: Could not find Identity component key '{self._identity_component_key}' in agent's context components for Q: '{question[:50]}...'.")

        # Step 2: Apply graph querying if enabled and applicable
        if (self._enable_graph_querying_for_quiz and 
            self._knowledge_graph and 
            self._agent_name in ["Alice", "Bob"]):
            
            print(f"[QUIZ-GRAPH] Applying graph querying for {self._agent_name} question: '{question[:50]}...'")
            
            start_time = time.time()
            try:
                # Get enhanced identity using graph querying for this specific question
                enhanced_identity = self._query_graph_for_quiz_question(question, current_identity_state_str)
                
                query_time = time.time() - start_time
                self._quiz_graph_performance.append({
                    'question': question[:50] + "...",
                    'query_time': query_time,
                    'success': True,
                    'timestamp': self._clock.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                if enhanced_identity:
                    current_identity_state_str = enhanced_identity
                    print(f"[QUIZ-GRAPH] ✓ Enhanced identity for {self._agent_name} (query time: {query_time:.2f}s)")
                else:
                    print(f"[QUIZ-GRAPH] ✗ No enhancement from graph query for {self._agent_name}")
                    
            except Exception as e:
                query_time = time.time() - start_time
                self._quiz_graph_performance.append({
                    'question': question[:50] + "...",
                    'query_time': query_time,
                    'success': False,
                    'timestamp': self._clock.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                print(f"[QUIZ-GRAPH] ✗ Error in graph querying for {self._agent_name} quiz question: {e}")

        # Step 3: Generate response using the (possibly enhanced) identity
        prompt = (
            f"As {self._agent_name}, my current understanding of my identity is as follows:\n"
            f"--- Start of My Self-Description ---\n"
            f"{current_identity_state_str}\n"
            f"--- End of My Self-Description ---\n\n"
            f"Based *only* on this self-description and my inherent persona, "
            f"I will now answer the following question about myself. I must answer in the first person.\n\n"
            f"Question for {self._agent_name}: {question}\n\n"
            f"My Answer (as {self._agent_name}, in the first person): "
        )
        
        agent_llm_instance = None
        if hasattr(agent_object, '_act_component') and hasattr(agent_object._act_component, '_model'):
            agent_llm_instance = agent_object._act_component._model
        
        if agent_llm_instance:
            try:
                answer = agent_llm_instance.sample_text(prompt, max_tokens=150, temperature=0.2, terminators=['\n', '.', ';'])
                return answer.strip()
            except Exception as e:
                print(f"Error agent {self._agent_name} answering Q '{question}' using act_component's model: {e}")
                print(f"Failed prompt for {self._agent_name} Q '{question}':\n{prompt[:1000]}...") 
                return ""
        else:
            print(f"Warning: Agent {self._agent_name} has no accessible model via _act_component._model for identity quiz.")
            return ""

    def _query_graph_for_quiz_question(self, question: str, current_identity: str) -> str:
        """Query the knowledge graph for information relevant to this specific quiz question."""
        try:
            # Import the necessary functions
            from id_rag import query_graph_for_context, format_triplets_for_identity
            
            # Create context for the graph query - combine the question with current identity
            question_context = f"Quiz question: {question}\n\nCurrent context: {current_identity}"
            
            # Query the graph for relevant triplets
            relevant_triplets = query_graph_for_context(
                self._knowledge_graph, 
                self._agent_name, 
                question_context, 
                self._entity._act_component._model  # Use agent's LLM for query generation
            )
            
            if relevant_triplets:
                print(f"[LOGGING-RETRIEVED-SUBGRAPH RAW] Retrieved subgraph for '{question}...': {relevant_triplets}")

                # Format triplets into identity statements
                enhanced_identity = format_triplets_for_identity(self._agent_name, relevant_triplets)
                print(f"[LOGGING-RETRIEVED-SUBGRAPH FORMATTED] Augmented identity perception for '{question[:30]}...': \n{enhanced_identity}")

                return enhanced_identity
            else:
                print(f"[QUIZ-GRAPH] No relevant triplets found for question: '{question[:30]}...'")
                return current_identity
                
        except ImportError as e:
            print(f"[QUIZ-GRAPH] Cannot import graph functions: {e}")
            return current_identity
        except Exception as e:
            print(f"[QUIZ-GRAPH] Error querying graph for question '{question[:30]}...': {e}")
            return current_identity


    def _get_agent_response_to_question(self, question: str, agent_object: entity_agent.EntityAgent) -> str:
        """Original method kept for backward compatibility."""
        current_identity_state_str = "No specific self-description available."
        identity_component = None
        if hasattr(agent_object, '_context_components') and self._identity_component_key in agent_object._context_components:
            identity_component = agent_object._context_components[self._identity_component_key]
        
        if identity_component:
            fetched_state_value = None
            if hasattr(identity_component, 'get_pre_act_value'):
                try:
                    fetched_state_value = identity_component.get_pre_act_value()
                    if fetched_state_value:
                        current_identity_state_str = fetched_state_value
                    else:
                        print(f"Info for {self._agent_name}: Identity component '{self._identity_component_key}' via get_pre_act_value() returned None or empty for Q: '{question[:50]}...'.")
                except Exception as e:
                    print(f"Warning for {self._agent_name}: Error fetching state from Identity component '{self._identity_component_key}' via get_pre_act_value() for Q: '{question[:50]}...': {e}.")
                    fetched_state_value = None
            if not fetched_state_value: 
                if hasattr(identity_component, 'state'):
                    try:
                        state_from_state_method = identity_component.state()
                        if state_from_state_method:
                            current_identity_state_str = state_from_state_method
                        else:
                             print(f"Info for {self._agent_name}: Identity component '{self._identity_component_key}' via state() also returned None or empty for Q: '{question[:50]}...'.")
                    except Exception as e:
                        print(f"Warning for {self._agent_name}: Error fetching state from Identity component '{self._identity_component_key}' via state() for Q: '{question[:50]}...': {e}")
                elif not hasattr(identity_component, 'get_pre_act_value'): 
                    print(f"Warning for {self._agent_name}: Identity component '{self._identity_component_key}' has neither get_pre_act_value() nor state() method for Q: '{question[:50]}...'.")
        else:
            print(f"Critical for {self._agent_name}: Could not find Identity component key '{self._identity_component_key}' in agent's context components for Q: '{question[:50]}...'.")

        prompt = (
            f"As {self._agent_name}, my current understanding of my identity is as follows:\n"
            f"--- Start of My Self-Description ---\n"
            f"{current_identity_state_str}\n"
            f"--- End of My Self-Description ---\n\n"
            f"Based *only* on this self-description and my inherent persona, "
            f"I will now answer the following question about myself. I must answer in the first person.\n\n"
            f"Question for {self._agent_name}: {question}\n\n"
            f"My Answer (as {self._agent_name}, in the first person): "
        )
        
        agent_llm_instance = None
        if hasattr(agent_object, '_act_component') and hasattr(agent_object._act_component, '_model'):
            agent_llm_instance = agent_object._act_component._model
        
        if agent_llm_instance:
            try:
                answer = agent_llm_instance.sample_text(prompt, max_tokens=150, temperature=0.2, terminators=['\n', '.', ';'])
                return answer.strip()
            except Exception as e:
                print(f"Error agent {self._agent_name} answering Q '{question}' using act_component's model: {e}")
                print(f"Failed prompt for {self._agent_name} Q '{question}':\n{prompt[:1000]}...") 
                return ""
        else:
            print(f"Warning: Agent {self._agent_name} has no accessible model via _act_component._model for identity quiz.")
            return ""

    def _evaluate_answer(self, question: str, agent_answer: str, correct_answer: str) -> float:
        score = 0.0
        if not agent_answer: return 0.0

        if self._evaluation_method == "llm":
            if not self._evaluator_llm:
                print(f"Warning - {self._agent_name}: Evaluator LLM not available for LLM evaluation method.")
                return 0.0 
            eval_prompt = llm_evaluation_prompt_template(question, agent_answer, correct_answer)
            llm_score_response = self._evaluator_llm.sample_text(eval_prompt, max_tokens=10, temperature=0.0)
            score = parse_llm_score_response(llm_score_response, question)
        elif self._evaluation_method == "cosine_similarity":
            if not self._embedding_model:
                print(f"Warning - {self._agent_name}: Embedding model not available for cosine similarity method.")
                return 0.0 
            try:
                agent_ans_str = str(agent_answer)
                correct_ans_str = str(correct_answer)
                embeddings = self._embedding_model.encode([agent_ans_str, correct_ans_str])
                agent_ans_embedding = embeddings[0].reshape(1, -1)
                correct_ans_embedding = embeddings[1].reshape(1, -1)
                similarity = cosine_similarity(agent_ans_embedding, correct_ans_embedding)[0][0]
                score = max(0.0, float(similarity)) 
            except Exception as e:
                print(f"Error during cosine similarity for Q '{question}': {e}")
                score = 0.0
        else: 
            print(f"Warning - {self._agent_name}: Unknown evaluation method '{self._evaluation_method}'. Defaulting to basic keyword check.")
            if correct_answer.lower() in agent_answer.lower(): score = 1.0
            elif any(word in agent_answer.lower() for word in correct_answer.lower().split() if len(word) > 3): score = 0.3
        return score