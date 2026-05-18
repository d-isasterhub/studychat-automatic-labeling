#!/usr/bin/python3
import dspy

from typing import Literal

class AssignmentClassifier(dspy.Signature):
    """
    You are annotating a student-chatbot interaction from a university AI course.
    You are given multiple assignment descriptions and a student conversation.

    Each assignment is labeled (a1-a7).
    Compare the student's current prompt history to the assignment descriptions.

    Select the assignment that best matches the student's inquiries.
    If no clear match exists, return the original assignment.
    """
    assignment_descriptions:dict = dspy.InputField(desc="Summaries of assignments")
    chat:str = dspy.InputField(desc="Prompt history")
    original_assignment:str = dspy.InputField(desc="Originally assigned assignment")
    
    assignment:Literal["a1", "a2", "a3", "a4", "a5", "a6", "a7"] = dspy.OutputField()
    
    
class PromptRelation(dspy.Signature):
    """
    You are annotating a student-chatbot interaction from a university AI course.
    Classify how the student's current prompt relates to the given assignment.
    
    You are given:
    - assignment_context: description of the assignment, requirements, and expected outcomes
    - history: previous conversation between student and assistant
    - prompt: the student's latest message to classify
    
    Choose one label:

    - setting up the learning environment:
        The prompt is about setup, tooling, infrastructure, or logistics needed to work on the assignment
        (e.g., installing Python, configuring IDEs, accessing datasets, debugging environment issues),
        even if it is in service of the assignment.

    - related:
        The prompt directly or indirectly relates to the assignment content, requirements, concepts,
        expected deliverables, or builds on prior assignment-related conversation.

    - unrelated:
        The prompt has no meaningful connection to the assignment context, goals, or required knowledge,
        and is instead about unrelated topics or general conversation.
    """
    assignment_context: str = dspy.InputField(desc="Assignment description, requirements, and objectives")
    history: str = dspy.InputField(desc="Previous chat history")
    prompt: str = dspy.InputField(desc="Current student message")
    
    relation: Literal["related", "unrelated", "setting up the learning environment"] = dspy.OutputField(
        desc="Classify relation to assignment"
    )
    

class DecisionSolutionSeeking(dspy.Signature):
    """
    You are annotating a student-chatbot interaction from a university AI course.
    Determine whether the learner's prompt is aimed at obtaining a concrete solution
    to an assignment task or subtask.

    This includes cases where the user:
    - asks for code, final answers, or implementations that directly produce a solution
        to an assignment task or subtask
    - requests help that would reasonably result in a full or partial solution 
        (even if framed as a question for explanation, illustration, or examples)

    Do NOT classify as solution-seeking if the user is only:
    - asking conceptual questions without expecting an answer to be produced
    - discussing theory without needing an executable or concrete output
    - asking for output where the connection to assignment tasks is ambiguous
    
    Do not infer solution-seeking based on potential downstream use of the response.
    Only label as solution-seeking when the prompt is clearly anchored in solving an
    assignment task or subtask itself, not when it merely could be used to construct a solution.

    Use assignment_context and history to disambiguate intent.
    """

    assignment_context: str = dspy.InputField(desc="Task excerpts by similarity")
    history: str = dspy.InputField(desc="Prior conversation context")
    prompt: str = dspy.InputField(desc="Current student message")

    solution_seeking: Literal["yes", "no"] = dspy.OutputField(
        desc="yes if the user is trying to obtain a concrete solution"
    )
    
    
class DecisionSolutionType(dspy.Signature):
    """
    You are annotating a student-chatbot interaction from a university AI course.
    The learner's prompt has been identified as solution seeking behavior.
    Classify the type of solution-related intent.

    Choose:
    - create:
        The learner is requesting a new solution, implementation, or answer
        that has not yet been provided (even if a code frame is given 
        by the assignment task).

    - verify/fix/adapt:
        The learner is working with an existing solution (from assignment or history)
        and wants to:
        - debug it
        - correct it
        - improve it
        - adapt it to new constraints
        - verify correctness
    """
    assignment_context: str = dspy.InputField(desc="Task excerps by similarity")
    history: str = dspy.InputField(desc="Prior conversation context")
    prompt: str = dspy.InputField(desc="Current student message")
    
    solution_type:Literal["create", "verify/fix/adapt"] = dspy.OutputField(desc="create or verify/fix/adapt")
    
    
class DecisionKnowledgeSeeking(dspy.Signature):
    """
    You are annotating a student-chatbot interaction from a university AI course.
    Determine whether the learner is seeking understanding, explanation, or information.

    - yes:
        The learner asks for information, explanations, definitions, concepts, reasoning,
        or clarification.

    - no:
        The learner's request cannot be reasonably interpreted as knowledge seeking.
    """
    prompt: str = dspy.InputField(desc="Current student message")

    knowledge_seeking: Literal["yes", "no"] = dspy.OutputField(
        desc="yes if the user is reasonably exhibiting knowledge seeking behavior"
    )
    

class DecisionCognitiveEngagement(dspy.Signature):
    """
    You are annotating a student-chatbot interaction from a university AI course.
    Detect whether the learner shows active cognitive engagement with prior context.

    - yes:
        The learner:
        - reacts to previous assistant messages
        - asks follow-up questions
        - challenges, refines, or builds on prior answers
        - references earlier steps or reasoning

    - no:
        The learner:
        - issues standalone requests
        - shows no connection to prior conversation
        - does not build on earlier content
    """
    history: str = dspy.InputField(desc="Previous conversation history")
    prompt: str = dspy.InputField(desc="Current student message")

    cognitive_engagement: Literal["yes", "no"] = dspy.OutputField(
        desc="yes if there is clear engagement with prior context"
    )
    
    
class DecisionMentalModel(dspy.Signature):
    """
    You are annotating a student-chatbot interaction from a university AI course.
    Determine whether the learner is expressing or refining an internal understanding
    (mental model) of the subject. A mental model is **present** if the user is testing 
    or proposing how a system works (even implicitly).  

    - yes:
        The learner:
        - proposes explanations or hypotheses
        - expresses assumptions or reasoning
        - compares alternative ideas
        - tries to validate or refine their understanding
        - asks if their interpretation is correct

    - no:
        The learner:
        - only requests information or explanations
        - does not reference their own reasoning or assumptions
        - passively asks for answers without self-explanation
    """
    prompt: str = dspy.InputField(desc="Current student message")

    mental_model: Literal["yes", "no"] = dspy.OutputField(
        desc="yes if user expresses or refines an internal model of understanding"
    )