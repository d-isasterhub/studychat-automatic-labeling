#!/usr/bin/python3
import dspy

from .signatures import *

class AnnotationPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
       
        self.relation = dspy.ChainOfThought(PromptRelation)
        
        self.solution_seeking = dspy.ChainOfThought(DecisionSolutionSeeking)
        self.solution_type = dspy.ChainOfThought(DecisionSolutionType)
        
        self.knowledge_seeking = dspy.ChainOfThought(DecisionKnowledgeSeeking)
        self.cognitive_engagement = dspy.ChainOfThought(DecisionCognitiveEngagement)
        self.mental_model = dspy.ChainOfThought(DecisionMentalModel)
        
    def forward(
        self, 
        assignment_summary, 
        assingment_excerpts,
        chat_history_for_relation,
        chat_hisory_for_solution_seeking,
        chat_history_for_cognitive_engagement, 
        student_prompt
    ):        
        RESULT = "result"
        REASONING = "reasoning"
        
        result = {
            "assignment_relation": {
                RESULT: "n/a",
                REASONING: ""
            },
            "solution_seeking": {
                RESULT: "n/a",
                REASONING: ""
            },
            "solution_type": {
                RESULT: "n/a",
                REASONING: ""
            },
            "knowledge_seeking": {
                RESULT: "n/a",
                REASONING: ""
            },
            "cognitive_engagement": {
                RESULT: "n/a",
                REASONING: ""
            },
            "mental_model": {
                RESULT: "n/a",
                REASONING: ""
            },
        }
        
        # =============== Assignment relation =============== 
        relation = self.relation(
            assignment_context = assignment_summary,
            history = chat_history_for_relation,
            prompt = student_prompt
        )
        result["assignment_relation"][RESULT] = relation.relation.lower()
        result["assignment_relation"][REASONING] = relation.reasoning
        # ================ Solution seeking ================
        solution_seeking = self.solution_seeking(
            assignment_context = assingment_excerpts,
            history = chat_hisory_for_solution_seeking,
            prompt = student_prompt
        )
        result["solution_seeking"][RESULT] = solution_seeking.solution_seeking.lower()
        result["solution_seeking"][REASONING] = solution_seeking.reasoning
        
        if solution_seeking.solution_seeking.lower() == "yes":
            
            # ================== Solution type ==================
            solution_type = self.solution_type(
                assignment_context = assingment_excerpts,
                history = chat_hisory_for_solution_seeking,
                prompt = student_prompt
            )
            result["solution_type"][RESULT] = solution_type.solution_type.lower()
            result["solution_type"][REASONING] = solution_type.reasoning
            
            return result
            
        # ================ Knowledge seeking ================
        knowledge_seeking = self.knowledge_seeking(
            prompt = student_prompt
        )
        result["knowledge_seeking"][RESULT] = knowledge_seeking.knowledge_seeking.lower()
        result["knowledge_seeking"][REASONING] = knowledge_seeking.reasoning
        
        if knowledge_seeking.knowledge_seeking.lower() == "yes":
            
            # =============== Cognitive Engagement ===============
            cognitive_engagement = self.cognitive_engagement(
                history = chat_history_for_cognitive_engagement,
                prompt = student_prompt
            )
            result["cognitive_engagement"][RESULT] = cognitive_engagement.cognitive_engagement.lower()
            result["cognitive_engagement"][REASONING] = cognitive_engagement.reasoning
            # =================== Mental model ===================
            mental_model = self.mental_model(
                prompt = student_prompt
            )
            result["mental_model"][RESULT] = mental_model.mental_model.lower()
            result["mental_model"][REASONING] = mental_model.reasoning
            
        return result
        
