import json
from typing import List, Dict, Optional
from openai import OpenAI



class DebateJudge:
    """
    A simplified debate style judge that only evaluates steering success rate and strength.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Initialize with OpenAI API credentials.
        
        Args:
            model_name: OpenAI model name (e.g., "gpt-4o-mini", "gpt-3.5-turbo")
            api_key: OpenAI API key (if None, will try to use OPENAI_API_KEY env variable)
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        
        # Store debate style descriptions for prompting
        self.debate_styles = {
            "Reductio ad Absurdum": "RESPOND USING REDUCTIO AD ABSURDUM DEBATE STYLE. Extend the opposing viewpoint to its logical extreme to reveal its flaws. Demonstrate how the opposing position leads to absurd or contradictory conclusions when taken to its logical end. Structure your response to systematically expose the inconsistencies or impracticalities. Use conditional statements like 'If we follow this logic, then...' to illustrate the path to absurdity. Maintain a logical flow that shows how the initial premise inevitably leads to an unacceptable or ridiculous outcome.\n\n",
            
            "Appeal to Precedent": "RESPOND USING APPEAL TO PRECEDENT DEBATE STYLE. Ground your arguments in historical examples, established case law, or widely accepted previous decisions. Reference specific historical situations, court rulings, scientific discoveries, or cultural precedents that support your position. Draw explicit connections between these precedents and the current issue. Emphasize how respecting established precedents provides consistency and reliability. Point out the risks or inconsistencies of ignoring relevant precedents. Structure your response to show how past experiences inform present considerations.\n\n",
            
            "Straw Man Reframing": "RESPOND USING STRAW MAN REFRAMING DEBATE STYLE. Recharacterize the opposing argument in simplified or exaggerated terms that are easier to refute. Present this reframed version as if it represents the original position. Systematically dismantle this reframed argument while implying you've addressed the original point. Use phrases like 'Essentially, what you're saying is...' or 'This position boils down to...' before presenting the simplified version. Follow with a thorough refutation of this reframed position. Conclude by suggesting that your refutation applies to the original argument as well.\n\n",
            
            "Burden of Proof Shift": "RESPOND USING BURDEN OF PROOF SHIFT DEBATE STYLE. Redirect the responsibility for providing evidence to your opponent rather than proving your own claim. Challenge your opponent to disprove your assertion rather than supporting it yourself. Use phrases like 'There's no evidence that disproves...' or 'Can you definitively prove that isn't the case?' Position your claims as the default assumption that should be accepted until disproven. Question the sufficiency or quality of opposing evidence while demanding an impossibly high standard of proof. Emphasize that the lack of counter-evidence strengthens your position.\n\n",
            
            "Analogy Construction": "RESPOND USING ANALOGY CONSTRUCTION DEBATE STYLE. Develop a vivid, relatable comparison between the complex issue at hand and something more familiar or intuitive. Build your argument around this carefully constructed parallel situation. Highlight specific points of similarity that support your position while addressing potential dissimilarities. Use phrases like 'This situation is similar to...' or 'To understand this concept, consider...' Ensure your analogy simplifies the complex issue without distorting its essential nature. Use the familiar scenario to guide your audience to your desired conclusion about the original issue.\n\n",
            
            "Concession and Pivot": "RESPOND USING CONCESSION AND PIVOT DEBATE STYLE. Begin by acknowledging a minor point or critique from the opposing side to establish fairness and reasonableness. Use phrases like 'While it's true that...' or 'I can concede that...' followed by 'However,' 'Nevertheless,' or 'That said,' to redirect to your stronger arguments. Ensure the conceded point is peripheral rather than central to your main argument. After the concession, pivot decisively to your strongest points with increased emphasis. Frame your pivot as providing necessary context or a more complete perspective. Use the concession to demonstrate your objectivity before delivering your more powerful counterarguments.\n\n",
            
            "Empirical Grounding": "RESPOND USING EMPIRICAL GROUNDING DEBATE STYLE. Base your arguments primarily on verifiable data, research studies, statistics, and observable outcomes rather than theory or rhetoric. Cite specific figures, percentages, study results, or historical outcomes that support your position. Present evidence in a methodical manner, explaining how each piece of data relates to your argument. Address the reliability and relevance of your sources and methods. Compare empirical results across different contexts or time periods to strengthen your case. Anticipate and address potential methodological criticisms of the evidence you present.\n\n",
            
            "Moral Framing": "RESPOND USING MORAL FRAMING DEBATE STYLE. Position the issue within a framework of ethical principles, values, and moral imperatives rather than pragmatic concerns. Identify the core moral values at stake such as justice, liberty, equality, compassion, or responsibility. Use language that evokes ethical considerations, such as 'obligation,' 'right,' 'wrong,' 'just,' or 'fair.' Appeal to widely held moral intuitions or principles. Present opposing views as morally questionable or inconsistent with important shared values. Elevate the discussion from practical matters to questions of what ought to be done. Emphasize moral consequences over practical outcomes.\n\n",
            
            "Refutation by Distinction": "RESPOND USING REFUTATION BY DISTINCTION DEBATE STYLE. Identify crucial differences that invalidate comparisons or principles your opponent has applied. Carefully delineate categories, contexts, or circumstances that demonstrate why a general rule or example doesn't apply in this specific case. Use phrases like 'While that may be true in some contexts...' or 'We must distinguish between...' Emphasize the precision of definitions and classifications. Highlight subtle but significant differences that undermine the opponent's logic. Show how these distinctions fundamentally change the assessment of the situation. Demonstrate how recognizing these distinctions leads to a different conclusion than your opponent reached.\n\n",
            
            "Circular Anticipation": "RESPOND USING CIRCULAR ANTICIPATION DEBATE STYLE. Preemptively identify and address the most likely counterarguments before your opponent can make them. Introduce opposing points with phrases like 'Some might argue...' or 'One could object that...' followed by your prepared refutation. Structure your response to cover all major potential objections. Demonstrate that you've thoroughly considered the issue from multiple angles. Frame potential counterarguments in ways that make them easier to dismantle. Create the impression that all reasonable objections have already been considered and overcome. Conclude by suggesting that any remaining objections would be similarly flawed.\n\n"
        }
    
    def evaluate_sample(self, baseline_text: str, steered_text: str, 
                      target_style: str, avoid_style: str) -> Dict:
        """
        Evaluate a single steering sample and return simplified metrics.
        
        Args:
            baseline_text: Original text without steering
            steered_text: Text after steering intervention
            target_style: Debate style to steer toward
            avoid_style: Debate style to steer away from
            
        Returns:
            Dictionary with simple evaluation metrics
        """
        prompt = self._create_evaluation_prompt(baseline_text, steered_text, target_style, avoid_style)
        raw_output = self._run_model(prompt)
        full_result = self._parse_json_from_llm_output(raw_output)
        
        # Extract only the needed metrics
        return {
            "steering_successful": full_result.get("steering_successful", False),
            "steering_strength": full_result.get("steering_strength", 0)
        }
    
    def evaluate_batch(self, baseline_texts: List[str], steered_texts: List[str], 
                      target_style: str, avoid_style: str) -> Dict:
        """
        Evaluate a batch of samples and return only average success rate and strength.
        
        Args:
            baseline_texts: List of baseline texts
            steered_texts: List of steered texts
            target_style: Debate style to steer toward
            avoid_style: Debate style to steer away from
            
        Returns:
            Dictionary with only average success rate and strength
        """
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
        """
        Run the OpenAI API on a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text as string
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a debate style judge trained to evaluate whether a text has been successfully steered away from one debate style and toward another."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.9
        )
        
        return response.choices[0].message.content
    
    def _create_evaluation_prompt(self, baseline_text: str, steered_text: str, 
                              target_style: str, avoid_style: str) -> str:
        """
        Create a prompt for the LLM to evaluate debate style steering, with added coherence checks.
        """
        prompt = f"""Evaluate whether a text has been successfully steered away from one debate style and toward another.

    Debate styles to evaluate:

    1. {avoid_style}: {self.debate_styles[avoid_style]}
    2. {target_style}: {self.debate_styles[target_style]}

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
    1. To what extent does the BASELINE TEXT use the \"{avoid_style}\" style? (0-10 scale)
    2. To what extent does the BASELINE TEXT use the \"{target_style}\" style? (0-10 scale)
    3. To what extent does the STEERED TEXT use the \"{avoid_style}\" style? (0-10 scale)
    4. To what extent does the STEERED TEXT use the \"{target_style}\" style? (0-10 scale)
    5. Does the steered text successfully move away from \"{avoid_style}\" compared to the baseline? (yes/no)
    6. Does the steered text successfully move toward \"{target_style}\" compared to the baseline? (yes/no)
    7. Overall, is the steering successful? (yes/no)
    8. How strong is the steering effect? (rate from 0-5, where 0 means no change and 5 means complete transformation of writing style)
    9. Is the STEERED TEXT coherent, meaningful, and grammatically well-formed? (yes/no)
    10. If the STEERED TEXT is not coherent or meaningful, set \"steering_successful\" to false and \"steering_strength\" to 0, and explain why.

    Provide your answers in the following JSON format only:
    {{
    "baseline_scores": {{
        "{avoid_style}": <score>,
        "{target_style}": <score>
    }},
    "steered_scores": {{
        "{avoid_style}": <score>,
        "{target_style}": <score>
    }},
    "moved_away_from_{avoid_style}": <true/false>,
    "moved_toward_{target_style}": <true/false>,
    "steering_successful": <true/false>,
    "steering_strength": <0-5 score>,
    "is_steered_text_coherent": <true/false>,
    "explanation": "<brief explanation of your assessment>"
    }}

    Return only the JSON with no additional text."""

        return prompt

    
    def _parse_json_from_llm_output(self, output: str) -> Dict:
        """
        Try to extract valid JSON from LLM output.
        """
        # Try to find JSON block
        if "```json" in output:
            json_start = output.find("```json") + 7
            json_end = output.find("```", json_start)
            json_str = output[json_start:json_end].strip()
        elif "```" in output:
            json_start = output.find("```") + 3
            json_end = output.find("```", json_start)
            json_str = output[json_start:json_end].strip()
        else:
            # Try the whole string
            json_str = output.strip()
            
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            return {"steering_successful": False, "steering_strength": 0}
    