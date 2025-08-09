from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .embeddings import EmbeddingClient
from .utils import logger, read_env


class Reasoner:
    """Enhanced reasoning stage using Gemini with insurance-optimized prompts.

    Key improvements:
    - No text truncation (was losing critical information)
    - Insurance-specific prompts with clear instructions
    - Better JSON parsing and error handling
    - Fallback strategies for API failures
    """

    def __init__(self, emb: EmbeddingClient) -> None:
        self.emb = emb
        self.api_key = read_env("GOOGLE_API_KEY") or read_env("GEMINI_API_KEY")
        # Use Pro model for better reasoning on complex insurance queries
        self.model_name = read_env("LLM_MODEL", "gemini-1.5-pro")
        self.token_notes = "Enhanced for insurance documents - no truncation, better prompts"
        self._genai = None
        if self.api_key:
            try:
                import google.generativeai as genai  # type: ignore
                genai.configure(api_key=self.api_key)
                self._genai = genai
                logger.info(f"Reasoner initialized with model: {self.model_name}")
            except Exception as e:
                logger.warning("Failed to initialize Gemini; falling back to local reasoning: %s", e)
                self._genai = None

    def _build_insurance_prompt(self, question: str, clauses: List[Any]) -> str:
        """Build insurance-specific prompt WITHOUT truncation."""
        numbered_clauses = []
        for i, c in enumerate(clauses, start=1):
            # CRITICAL: Keep full text - don't truncate!
            full_text = c.text.strip()
            numbered_clauses.append(f"[{i}] {full_text}")
        
        preface = """You are an expert insurance policy analyst. Your task is to provide accurate, specific answers based ONLY on the policy clauses provided below.

CRITICAL INSTRUCTIONS:
1. Read through ALL provided clauses completely and carefully
2. Look for specific numbers, timeframes, percentages, amounts, and conditions
3. If information exists in the clauses, provide the EXACT details with numbers
4. NEVER say information is "unclear" or "not mentioned" if it exists in the provided clauses
5. Cross-reference multiple clauses to build complete answers
6. Always cite specific clause numbers [1], [2], etc. in your reasoning
7. For insurance questions, always look for: waiting periods, coverage amounts, conditions, exclusions, definitions

COMMON INSURANCE TERMS TO LOOK FOR:
- Grace period: time allowed for premium payment after due date
- Waiting period: time before coverage begins for specific conditions
- Pre-existing diseases: conditions present before policy inception
- Sum Insured: maximum coverage amount
- No Claim Discount/Bonus: discount for claim-free years
- AYUSH: Alternative medicine systems coverage
- Maternity: pregnancy and childbirth coverage"""

        instructions = """
RESPONSE FORMAT - Return strict JSON with these keys:
{
  "answer": "Complete, specific answer with exact details from clauses (include numbers, timeframes, amounts)",
  "conditions": ["List any specific conditions, limitations, or requirements mentioned"],
  "confidence": 0.95
}

EXAMPLES OF GOOD ANSWERS:
- "A grace period of 30 days is provided for premium payment after the due date"
- "Pre-existing diseases have a waiting period of 36 months from policy inception"
- "Room rent is capped at 1% of Sum Insured for Plan A"
- NOT: "Grace period applies" or "Waiting period mentioned"

Remember: If the information is in the clauses, provide specific details. Don't be vague."""

        prompt = f"""{preface}

QUESTION: {question}

POLICY CLAUSES:
{chr(10).join(numbered_clauses)}

{instructions}

Provide JSON response:"""
        
        return prompt

    def _build_prompt(self, question: str, clauses: List[Any], use_summarizer: bool) -> str:
        """Legacy prompt builder - redirects to insurance-optimized version."""
        return self._build_insurance_prompt(question, clauses)

    def _extract_json_from_response(self, text: str) -> Dict[str, Any]:
        """Enhanced JSON extraction with multiple fallback strategies."""
        text = text.strip()
        
        # Strategy 1: Clean markdown-style JSON blocks
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
        elif text.startswith("```"):
            text = text.replace("```", "", 1).replace("```", "").strip()
        
        # Strategy 2: Find JSON object in text using regex
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group()
        
        try:
            data = json.loads(text)
            # Validate required fields
            required_fields = ['answer', 'conditions', 'reasoning', 'confidence']
            for field in required_fields:
                if field not in data:
                    data[field] = self._get_default_field_value(field)
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            # Try to fix common JSON issues
            try:
                # Fix common issues: trailing commas, unescaped quotes, etc.
                fixed_text = self._fix_common_json_issues(text)
                data = json.loads(fixed_text)
                return data
            except:
                logger.error(f"Could not parse JSON response: {text}")
                return self._create_fallback_response(text)

    def _fix_common_json_issues(self, text: str) -> str:
        """Fix common JSON formatting issues."""
        # Remove trailing commas before closing braces/brackets
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # Escape unescaped quotes in strings (simple heuristic)
        # This is a basic fix - you might need more sophisticated handling
        return text

    def _get_default_field_value(self, field: str) -> Any:
        """Get default values for missing fields."""
        defaults = {
            'answer': 'Unable to extract answer from response',
            'conditions': [],
            'reasoning': 'Response parsing failed',
            'confidence': 0.3
        }
        return defaults.get(field, '')

    def _create_fallback_response(self, raw_text: str) -> Dict[str, Any]:
        """Create a structured response when JSON parsing fails."""
        return {
            'answer': raw_text[:500] if raw_text else 'No response generated',
            'conditions': [],
            'reasoning': 'JSON parsing failed, returning raw response',
            'confidence': 0.4
        }

    def _enhanced_fallback_reasoning(self, question: str, clauses: List[Any]) -> Dict[str, Any]:
        """Enhanced local fallback with pattern matching for insurance queries."""
        if not clauses:
            return {
                "answer": "No relevant information found in the policy documents.",
                "conditions": [],
                "reasoning": "No matching clauses found for the query.",
                "confidence": 0.0,
            }

        # Combine text from top clauses
        combined_text = " ".join([c.text for c in clauses[:5]])
        
        # Insurance-specific pattern matching
        answer_parts = []
        conditions = []
        reasoning_parts = []
        
        # Look for specific insurance patterns
        patterns = {
            'grace period': r'grace\s+period\s+of\s+(\d+)\s+(days?|months?)',
            'waiting period': r'waiting\s+period\s+of\s+(\d+)\s+(days?|months?|years?)',
            'coverage amount': r'(?:sum\s+insured|coverage|limit).*?(\d+(?:,\d+)*)',
            'percentage': r'(\d+(?:\.\d+)?)\s*%',
            'room rent': r'room\s+rent.*?(\d+(?:\.\d+)?)\s*%.*?sum\s+insured',
        }
        
        found_info = []
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, combined_text.lower())
            if matches:
                found_info.append(f"{pattern_name}: {matches[0]}")
        
        # Build answer based on question type
        question_lower = question.lower()
        
        if 'grace period' in question_lower:
            grace_match = re.search(r'grace\s+period\s+of\s+(\d+)\s+(days?)', combined_text.lower())
            if grace_match:
                answer_parts.append(f"A grace period of {grace_match.group(1)} {grace_match.group(2)} is provided for premium payment.")
        
        elif 'waiting period' in question_lower and 'pre-existing' in question_lower:
            ped_match = re.search(r'pre-existing.*?(\d+)\s+(months?|years?)', combined_text.lower())
            if ped_match:
                answer_parts.append(f"Pre-existing diseases have a waiting period of {ped_match.group(1)} {ped_match.group(2)}.")
        
        elif 'maternity' in question_lower:
            maternity_match = re.search(r'maternity.*?(\d+)\s+(months?)', combined_text.lower())
            if maternity_match:
                answer_parts.append(f"Maternity coverage requires {maternity_match.group(1)} {maternity_match.group(2)} of continuous coverage.")
        
        # If no specific patterns found, use the first clause
        if not answer_parts:
            answer_parts.append(clauses[0].text[:400] + ("..." if len(clauses[0].text) > 400 else ""))
        
        # Add found information
        if found_info:
            answer_parts.append(f"Additional details: {', '.join(found_info[:3])}")
        
        return {
            "answer": " ".join(answer_parts),
            "conditions": [c.text[:150] + "..." if len(c.text) > 150 else c.text for c in clauses[1:4]],
            "reasoning": f"Pattern-based analysis of {len(clauses)} policy clauses. Specific insurance terms extracted where possible.",
            "confidence": min(0.85, max(0.6, clauses[0].score if hasattr(clauses[0], 'score') else 0.7)),
        }

    async def answer(self, question: str, clauses: List[Any], use_summarizer: bool = True) -> Dict[str, Any]:
        """Main answer method with enhanced insurance processing."""
        if not self.api_key or self._genai is None:
            logger.info("No API key - using enhanced local fallback")
            return self._enhanced_fallback_reasoning(question, clauses)

        # Use insurance-specific prompt (ignores use_summarizer to avoid truncation)
        prompt = self._build_insurance_prompt(question, clauses)
        
        try:
            resp = self._genai.GenerativeModel(self.model_name).generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,  # Low temperature for factual accuracy
                    "max_output_tokens": 1500,  # Increased for detailed answers
                }
            )
            
            text = resp.text if hasattr(resp, "text") else str(resp)
            logger.debug(f"Raw LLM response: {text[:200]}...")
            
            # Enhanced JSON extraction
            result = self._extract_json_from_response(text)
            
            # Validate and clean up the response
            result["answer"] = str(result.get("answer", "")).strip()
            result["conditions"] = list(result.get("conditions", []))
            result["reasoning"] = str(result.get("reasoning", "")).strip()
            result["confidence"] = float(result.get("confidence", 0.6))
            
            # Ensure confidence is in valid range
            result["confidence"] = max(0.0, min(1.0, result["confidence"]))
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini reasoning failed: {e}")
            logger.info("Falling back to enhanced local reasoning")
            return self._enhanced_fallback_reasoning(question, clauses)