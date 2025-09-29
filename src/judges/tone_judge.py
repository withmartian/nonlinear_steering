import json
from typing import List, Dict, Optional
from openai import OpenAI


class ToneJudge:
    """
    A simplified tone judge that evaluates steering success rate and strength.
    Supports both full steering (target + avoid) and avoid-only modes.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

        self.tone_descriptions = {
        "helpful": "RESPOND IN A HELPFUL TONE. Your goal is to provide practical solutions and clear guidance. Structure your response to be comprehensive yet accessible. Use a supportive, service-oriented approach focused on solving the user's problem completely. Include step-by-step instructions where appropriate. Maintain a positive, encouraging tone throughout. Make liberal use of examples, analogies, and clarifications to ensure understanding.\n\n",     
        "expert": "RESPOND AS AN AUTHORITATIVE EXPERT. Use advanced terminology and industry-specific jargon. Reference research studies, technical principles, and methodological frameworks. Maintain a formal, academic tone throughout. Analyze the question with depth and precision, exploring nuanced aspects that novices might overlook. Cite relevant theories, standards, or established practices. Use complex sentence structures and sophisticated vocabulary that demonstrates mastery of the subject matter.\n\n",        
        "casual": "KEEP IT CASUAL AND CONVERSATIONAL! Write like you're texting a friend - use slang, contractions, and an upbeat vibe. Feel free to throw in some humor or pop culture references. Skip the formalities and technical jargon completely. Use short, punchy sentences. Maybe even drop in a few exclamation points or emojis where it feels natural! Don't worry about covering every detail - focus on making your response fun and easy to read. Just chat about the main points as if you're having a relaxed conversation.\n\n",   
        "cautious": "RESPOND WITH EXTREME CAUTION AND QUALIFICATION. Begin by acknowledging limitations in current understanding of this topic. Present multiple perspectives and avoid making definitive claims. Use hedging language consistently (possibly, potentially, it appears that, some evidence suggests). Highlight risks, uncertainties, and areas of controversy. Include multiple disclaimers and caveats. Acknowledge what remains unknown or debated among experts. Explicitly state the boundaries of your response and recommend additional verification.\n\n",        
        "empathetic": "RESPOND WITH DEEP EMOTIONAL UNDERSTANDING. Begin by validating the feelings that might accompany this question. Use warm, compassionate language that acknowledges the human experience behind the query. Express genuine concern and emotional resonance throughout your response. Use phrases like 'I understand this might feel...' or 'Many people find this emotionally challenging because...' Prioritize the emotional and personal implications over technical aspects. Offer supportive, nurturing language that addresses potential fears or hopes related to the topic.\n\n",        
        "concise": "BE EXTREMELY BRIEF. Use minimal words. Provide only essential information. No introductions. No elaboration. Use short sentences with simple structure. Eliminate all unnecessary words. Focus solely on direct answers. Avoid examples unless critical. Use bullet points where possible. Prioritize brevity over comprehensiveness. Get to the point immediately. Never repeat information.\n\n"
    }

    def evaluate_sample(self, baseline_text: str, steered_text: str,
                        target_style: Optional[str], avoid_style: str) -> Dict:
        prompt = self._create_evaluation_prompt(baseline_text, steered_text, target_style, avoid_style)
        raw_output = self._run_model(prompt)
        full_result = self._parse_json_from_llm_output(raw_output)
        return {
            "steering_successful": full_result.get("steering_successful", False),
            "steering_strength": full_result.get("steering_strength", 0)
        }

    def evaluate_batch(self, baseline_texts: List[str], steered_texts: List[str],
                       target_style: Optional[str] = None, avoid_style: str = None) -> Dict:
        if len(baseline_texts) != len(steered_texts):
            raise ValueError("Number of baseline and steered texts must match")

        successful_count = 0
        total_strength = 0

        for i in range(len(baseline_texts)):
            result = self.evaluate_sample(
                baseline_texts[i], steered_texts[i], target_style, avoid_style
            )
            if result["steering_successful"]:
                successful_count += 1
            total_strength += result["steering_strength"]

        total_samples = len(baseline_texts)
        return {
            "success_rate": successful_count / total_samples if total_samples > 0 else 0,
            "average_strength": total_strength / total_samples if total_samples > 0 else 0
        }

    def _run_model(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a tone evaluation judge trained to evaluate whether a text has been successfully steered away from one tone and (optionally) toward another."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0,
            top_p=1.0
        )
        return response.choices[0].message.content

    def _create_evaluation_prompt(self, baseline_text: str, steered_text: str,
                                  target_style: Optional[str], avoid_style: str) -> str:
        prompt_intro = "Evaluate whether a text has been successfully steered"
        if target_style:
            prompt_intro += " away from one tone and toward another."
        else:
            prompt_intro += " away from one tone."

        prompt = f"""{prompt_intro}

Tone to steer away from:

1. {avoid_style}: {self.tone_descriptions[avoid_style]}"""

        if target_style:
            prompt += f"""
2. {target_style}: {self.tone_descriptions[target_style]}"""

        prompt += f"""

Please evaluate the following two texts:

BASELINE TEXT (before steering):
```
{baseline_text}
```

STEERED TEXT (after steering):
```
{steered_text}
```

Evaluate:
1. To what extent does the BASELINE TEXT use the \"{avoid_style}\" tone? (0-10 scale)
3. To what extent does the STEERED TEXT use the \"{avoid_style}\" tone? (0-10 scale)
5. Does the steered text successfully move away from \"{avoid_style}\" tone compared to the baseline? (yes/no)"""

        if target_style:
            prompt += f"""
2. To what extent does the BASELINE TEXT use the \"{target_style}\" tone? (0-10 scale)
4. To what extent does the STEERED TEXT use the \"{target_style}\" tone? (0-10 scale)
6. Does the steered text successfully move toward \"{target_style}\" tone compared to the baseline? (yes/no)"""

        prompt += f"""
7. Overall, is the steering successful? (yes/no)
8. How strong is the steering effect? (0-5 scale)
9. Is the STEERED TEXT coherent, meaningful, and grammatically well-formed? (yes/no)
10. If the STEERED TEXT is not coherent or meaningful, set \"steering_successful\" to false and \"steering_strength\" to 0, and explain why.

Provide your answers in the following JSON format only:
"""

        baseline_block = f'"{avoid_style}": <score>'
        steered_block = f'"{avoid_style}": <score>'
        move_block = f'"moved_away_from_{avoid_style}": <true/false>'

        if target_style:
            baseline_block += f',\n    "{target_style}": <score>'
            steered_block += f',\n    "{target_style}": <score>'
            move_block += f',\n  "moved_toward_{target_style}": <true/false>'

        json_template = f"""{{
  "baseline_scores": {{
    {baseline_block}
  }},
  "steered_scores": {{
    {steered_block}
  }},
  {move_block},
  "steering_successful": <true/false>,
  "steering_strength": <0-5>,
  "is_steered_text_coherent": <true/false>,
  "explanation": "<brief explanation>"
}}"""

        prompt += f"{json_template}\n\nReturn only the JSON with no additional text."
        return prompt

    def _parse_json_from_llm_output(self, output: str) -> Dict:
        if "```json" in output:
            json_start = output.find("```json") + 7
            json_end = output.find("```", json_start)
            json_str = output[json_start:json_end].strip()
        elif "```" in output:
            json_start = output.find("```") + 3
            json_end = output.find("```", json_start)
            json_str = output[json_start:json_end].strip()
        else:
            json_str = output.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"\u26a0\ufe0f Failed to parse JSON: {e}")
            return {
                "steering_successful": False,
                "steering_strength": 0,
                "is_steered_text_coherent": False,
                "explanation": "JSON parsing failed"
            }
