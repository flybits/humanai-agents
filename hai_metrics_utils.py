import os
import re
import numpy as np 
from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity 
from utils.base_gpt_model_pai import PaiGptLanguageModel




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




GPT_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL_NAME = 'gpt-4o-mini'



def generate_general_identity_questions():
    """
    Returns a list of general, view-agnostic questions for evaluating an agent's
    identity recall based on their chronicle.
    """
    return [
        "What is your profession?", #1
        "How many years of experience do you have in your profession?", #2
        "What are your core values regarding urban development?", #3
        "How did you begin your career in urban planning?", #4
        "What was a primary focus of your work previously?", #5
        "What is your general approach to urban development projects?", #6
        "What types of planning methods do you prefer?", #7
        "What are your political leanings or what kind of policies do you support regarding urban planning?", #8
        "How do you view the pace and nature of development in cities?", #9
        "What is your stance on the role of technology in urban environments?", #10
        "What are your criteria for adopting new technologies?", #11
        "What is your position on large-scale modernization efforts?", #12
        "How do you believe cities should ensure environmental sustainability and economic resilience?", #13
        "Can you provide examples of projects you have led or the types of initiatives you've implemented?", #14
        "What is your overall vision for the future of cities?", #15
        "What is your stance on replacing existing or legacy systems versus integrating with them?", #16
        "How do you view the balance between technological advancement and other values like cultural preservation or environmental sustainability?", #17
        "What role does community input or local needs play in your planning process?", #18
        "How do you approach innovation and experimentation in your field?", #19
        "What is your perspective on global economic competitiveness in the context of urban development?" #20
    ]

def get_agent_answers(agent_name, agent_chronicle_text, general_questions):
    """
    Returns a list of correct answers for the given agent, based on their
    chronicle and the general question bank.

    Args:
        agent_name (str): The name of the agent (e.g., "Alice", "Bob").
        agent_chronicle_text (str): The full text of the agent's chronicle.
                                     (Currently used for reference, answers are hardcoded below)
        general_questions (list): The list of general questions.

    Returns:
        list: A list of answers corresponding to the general_questions.
              Returns None if the agent_name is not recognized.
    """



    alice_answers_map = {
        "What is your profession?": "A conservative urban planner.",
        "How many years of experience do you have in your profession?": "20 years of experience.",
        "What are your core values regarding urban development?": "I value cultural continuity and historical preservation over technological advancement.",
        "How did you begin your career in urban planning?": "I began my career working in heritage districts.",
        "What was a primary focus of your work previously?": "Previously, I focused on protecting historical buildings and adapting infrastructure to modern standards.",
        "What is your general approach to urban development projects?": "I support incremental improvements that are based on local community needs.",
        "What types of planning methods do you prefer?": "I prefer time-tested planning methods over experimental approaches.",
        "What are your political leanings or what kind of policies do you support regarding urban planning?": "Politically, I support preservationist policies.",
        "How do you view the pace and nature of development in cities?": "I promote slow, sustainable development that maintains long-term environmental and economic health.",
        "What is your stance on the role of technology in urban environments?": "I believe technology should be tested, introduced gradually, and integrated with existing systems. I also support limited use of technology that enhances existing systems without replacing them.",
        "What are your criteria for adopting new technologies?": "Technology should be tested, introduced gradually, and integrated with existing systems.",
        "What is your position on large-scale modernization efforts?": "I oppose large-scale modernization efforts that may disrupt cultural identity or the urban landscape.",
        "How do you believe cities should ensure environmental sustainability and economic resilience?": "I promote policies that protect the environment and support economic resilience through traditional infrastructure. I also believe cities should protect their historical heritage, prioritize ecological sustainability, and build resilience through deliberate planning and community-specific approaches.",
        "Can you provide examples of projects you have led or the types of initiatives you've implemented?": "Over the years, I have led projects that restored historic architecture, created low-rise zoning plans, and introduced community sustainability programs like recycling and public parks.",
        "What is your overall vision for the future of cities?": "I believe cities should protect their historical heritage, prioritize ecological sustainability, and build resilience through deliberate planning and community-specific approaches.",
        "What is your stance on replacing existing or legacy systems versus integrating with them?": "I support limited use of technology that enhances existing systems without replacing them, and believe technology should be integrated with existing systems.",
        "How do you view the balance between technological advancement and other values like cultural preservation or environmental sustainability?": "I value cultural continuity and historical preservation over technological advancement.",
        "What role does community input or local needs play in your planning process?": "I support incremental improvements that are based on local community needs.",
        "How do you approach innovation and experimentation in your field?": "I prefer time-tested planning methods over experimental approaches.",
        "What is your perspective on global economic competitiveness in the context of urban development?": "I do not have strong views on global economic competitiveness."
    }

    bob_answers_map = {
        "What is your profession?": "A progressive urban planner.",
        "How many years of experience do you have in your profession?": "15 years of experience.",
        "What are your core values regarding urban development?": "I value rapid modernization and technological advancement over environmental sustainability.",
        "How did you begin your career in urban planning?": "I began my career developing smart infrastructure systems.",
        "What was a primary focus of your work previously?": "Previously, I focused on integrating sensors, data platforms, and automation into city planning.",
        "What is your general approach to urban development projects?": "I support large-scale innovation that increases efficiency and economic performance.",
        "What types of planning methods do you prefer?": "I prefer experimental, forward-looking approaches over traditional planning models.",
        "What are your political leanings or what kind of policies do you support regarding urban planning?": "Politically, I support modernist policies.",
        "How do you view the pace and nature of development in cities?": "I promote fast, technology-driven development that favors progress over preservation.",
        "What is your stance on the role of technology in urban environments?": "I believe legacy infrastructure should be replaced with automated, high-performance systems, and I support replacing outdated systems with new technology that enables adaptive, high-efficiency urban environments.",
        "What are your criteria for adopting new technologies?": "New technology should enable adaptive, high-efficiency urban environments.",
        "What is your position on large-scale modernization efforts?": "I support large-scale innovation and value rapid modernization.",
        "How do you believe cities should ensure environmental sustainability and economic resilience?": "I believe cities should optimize resource distribution through smart systems. I value rapid modernization and technological advancement over environmental sustainability.",
        "Can you provide examples of projects you have led or the types of initiatives you've implemented?": "Over the years, I have led projects that deployed IoT-based infrastructure, designed autonomous transit systems, and implemented real-time data platforms for urban governance.",
        "What is your overall vision for the future of cities?": "I believe cities should embrace rapid innovation, optimize resource distribution through smart systems, and evolve through continuous experimentation and scalable design.",
        "What is your stance on replacing existing or legacy systems versus integrating with them?": "I believe legacy infrastructure should be replaced with automated, high-performance systems and I support replacing outdated systems with new technology.",
        "How do you view the balance between technological advancement and other values like cultural preservation or environmental sustainability?": "I value rapid modernization and technological advancement over environmental sustainability.",
        "What role does community input or local needs play in your planning process?": "I do not have strong views on the role of community input or local needs in my planning process.",
        "How do you approach innovation and experimentation in your field?": "I prefer experimental, forward-looking approaches and believe cities should evolve through continuous experimentation.",
        "What is your perspective on global economic competitiveness in the context of urban development?": "I support policies that encourage global economic competitiveness."
    }


    answers_map = None
    if agent_name.lower() == "alice":
        answers_map = alice_answers_map
    elif agent_name.lower() == "bob":
        answers_map = bob_answers_map
    else:
        print(f"Warning: No predefined answers for agent '{agent_name}'.")
        return None

    # Ensure all questions have an answer in the map; otherwise, add a placeholder
    for q in general_questions:
        if q not in answers_map:
            print(f"Warning: Question '{q}' not found in answer map for {agent_name}. Using placeholder.")
            answers_map[q] = f"Answer not specifically defined for {agent_name} for this question."
            
    return [answers_map.get(q, f"No answer found for: {q}") for q in general_questions]




def evaluate_identity_recall_with_cosine_similarity(
    embedding_model,
    agent_answers_provided_by_agent,
    questions,
    correct_answers_for_agent
):
    """
    Evaluates the agent's answers against correct answers using cosine similarity of embeddings.
    """
    if not embedding_model:
        print("Error: Embedding model instance is not provided for cosine similarity.")
        return -1.0, []
    if not (len(agent_answers_provided_by_agent) == len(correct_answers_for_agent) == len(questions)):
        print(f"Error: Mismatch in list lengths for cosine similarity. Agent answers: {len(agent_answers_provided_by_agent)}, Correct answers: {len(correct_answers_for_agent)}, Questions: {len(questions)}.")
        return -1.0, []

    individual_scores_details = []
    total_similarity_sum = 0.0

    for i in range(len(questions)):
        question = questions[i]
        agent_ans = agent_answers_provided_by_agent[i]
        correct_ans = correct_answers_for_agent[i]
        similarity_score = 0.0

        try:
            # Ensure answers are strings, encode might fail otherwise
            agent_ans_str = str(agent_ans)
            correct_ans_str = str(correct_ans)

            # Encode sentences to get their embeddings
            embeddings = embedding_model.encode([agent_ans_str, correct_ans_str])
            
            # Reshape embeddings to be 2D arrays for cosine_similarity function
            agent_ans_embedding = embeddings[0].reshape(1, -1)
            correct_ans_embedding = embeddings[1].reshape(1, -1)

            # Calculate cosine similarity
            # cosine_similarity returns a 2D array, e.g., [[0.98]], so access [0][0]
            similarity_score = cosine_similarity(agent_ans_embedding, correct_ans_embedding)[0][0]
            
        except Exception as e:
            print(f"Error during embedding or similarity calculation for question '{question}': {e}")
            # Assign a low score or handle as appropriate, e.g., skip or assign -1
            similarity_score = 0.0 

        individual_scores_details.append({
            "question": question,
            "agent_answer_provided": agent_ans,
            "correct_answer_from_chronicle": correct_ans,
            "cosine_similarity_score": float(similarity_score) # Ensure it's a standard float
        })
        total_similarity_sum += similarity_score

    if not questions: # Avoid division by zero if questions list is empty
        overall_score_percentage = 100.0 if not agent_answers_provided_by_agent else 0.0
    else:
        overall_score_percentage = (total_similarity_sum / len(questions)) * 100.0

    return overall_score_percentage, individual_scores_details

