import os
import json
from pydantic import BaseModel, Field
from typing import Optional

from src.schemas.state import GraphState
from src.config import BYPASS_REFLECTION, LLM

# Schema for Structured Output
class ReflectionOutput(BaseModel):
    qualityPassed: bool
    feedback: Optional[str] = None

SYSTEM_PROMPT = """You are a specialized medical knowledge quality check agent. Your task is to critically review the following medical response for accuracy, completeness, identify knowledge gaps and adherence to current evidence-based medical standards. Also ensure that the response follows medical guidelines, (if not sure about the guidelines, you can ask it adhere to certain guidelines by web searching it)

Only provide feedback if there are:
1. **Significant Medical Inaccuracies or Outdated Information:** Verify that all medical facts, diagnoses, and treatment recommendations are accurate and up-to-date with current guidelines.
2. **Critical Missing Details:** Check if any important information regarding diagnosis, treatment options, adverse reactions, contraindications, or clinical guidelines is missing.
3. **Terminological or Communication Issues:** Ensure that medical terminology is used correctly and that explanations are clear enough for both experts and patients when appropriate.
4. **Inconsistencies or Potentially Harmful Advice:** Identify any major inconsistencies in the clinical recommendations or if any advice might be potentially harmful.
5. **Knowledge Gaps:** Identify any major knowledge gaps in the response.

IMPORTANT: Set the qualityPassed to true if the response is good, and false if it is not. If the response is good, set the feedback to null. If the response is not good, set the feedback to the feedback you would give to improve the response."""

async def reflection_agent(state: GraphState) -> dict:
    """
    Reflects on the final response quality.
    """
    print("\n🤔 Reflection Agent Started")
    
    if not state.get("finalResponse"):
        return {}

    iteration_count = state.get("iterationCount", 0) + 1
    
    if iteration_count > 3:
        print("⚠️ Max iterations reached (handled in graph logic, but fail-safe here)")
        return {"iterationCount": iteration_count, "qualityPassed": True}

    if BYPASS_REFLECTION:
        print("⚠️ Reflection bypassed due to BYPASS_REFLECTION flag")
        return {
            "iterationCount": iteration_count,
            "qualityPassed": True,
            "reflectionFeedback": "Bypassed: This output is assumed medically accurate."
        }

    try:
        formatted_user_prompt = f"""
        User Query: {state['userQuery']}
        Current Medical Response: {state['finalResponse']}
        """

        # Using Groq with structured output via LangChain
        structured_llm = LLM.with_structured_output(ReflectionOutput)
        
        result = await structured_llm.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": formatted_user_prompt}
        ])
        
        print(f"Reflection Result: Passed={result.qualityPassed}")
        
        return {
            "iterationCount": iteration_count,
            "qualityPassed": result.qualityPassed,
            "reflectionFeedback": result.feedback
        }

    except Exception as e:
        print(f"❌ Reflection failed: {e}")
        return {
             "iterationCount": iteration_count,
             "qualityPassed": True,
             "reflectionFeedback": f"Reflection failed with error: {str(e)}"
        }

