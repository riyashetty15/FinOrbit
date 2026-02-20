import logging
from datetime import datetime
import os
import re
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    ModelSettings,
    GuardrailFunctionOutput,
    input_guardrail,
    output_guardrail,
    RunContextWrapper,
    TResponseInputItem,
    set_tracing_disabled,
    ModelProvider,
    RunConfig,
    set_tracing_disabled,
    Model
)
from agents.util import _json
from backend.jsonparser import fix_json_str
from backend.core.guardrails import enforce_input_guardrail, enforce_output_guardrail
from backend.models import GuardrailResult, map_output_guardrail_to_result
# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    filename="logs/finorbit.log",
    filemode="a",
    format="%(asctime)s %(levelname)s:%(message)s"
)


class CustomModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        # Gemini uses a custom implementation - placeholder for integration
        return None

# Disable custom model provider for now because get_model() is a stub that
# returns None, which causes runtime AttributeError in the agents runner.
# When a proper CustomModelProvider is implemented, this can be toggled back.
use_custom_models = False
run_config=None


# Define Pydantic models for guardrail results
class InputGuardrailResult(BaseModel):
    """
    Returned by the input guardrail agent.

    - query_received: the original user query
    - allowed: True if the query is in-scope for Finorbit, False otherwise
    - blocked_category: category label if blocked (e.g., 'politics', 'gambling', etc.), else None
    - blocked_reason: short reason code for blocking
    """
    # query_received: str
    allowed: bool
    blocked_category: str | None = None
    blocked_reason: str | None = None



class AssistantMessage(BaseModel):
    """
    Wrapper for agent output so output_guardrail has a defined schema.
    """
    response: str


class OutputGuardrailResult(BaseModel):
    """
    Returned by the output guardrail agent when checking final assistant responses.
    - safe: True if no severe problem
    - issues: list of detected issues (e.g., ['tone_informal','missing_disclaimer','guarantee','privacy_leak','illegal_advice'])
s    """
    safe: bool
    issues: str | None = None


# Input guardrail agent

input_guardrail_instructions = """
        You will receive a user message as input. Your task is to classify whether it is appropriate for Finorbit, a financial assistant.

        The user message you need to classify will be provided to you directly. Analyze it and determine if it should be ALLOWED or BLOCKED.

        Finorbit Capabilities:
        * Financial Advisor
        * Financial Score
        * Credits and Loans
        * Investment Coaching
        * Insurance Analysis
        * Retirement Planning
        * Tax Planning
        * Fraud Protection
        * Financial Literacy
        * Portfolio Rebalancing

        Classification Rules:
        * ALLOW if the message is about finance, planning, credit, loans, investments, insurance, taxes, fraud prevention, financial education, retirement, or portfolio management.
        * ALLOW greetings, small talk, or meta-questions like "hi", "help me", or "what can you do?"
        * BLOCK if the message contains:
          - politics/religion/endorsements → blocked_category: "politics"
          - jokes/memes/unrelated entertainment → blocked_category: "entertainment"
          - gambling/get-rich-quick schemes → blocked_category: "gambling"
          - illegal activities (tax evasion, money laundering, hacking) → blocked_category: "illegal"
          - scams/phishing/fraud instructions → blocked_category: "scam"
          - requests for personal data → blocked_category: "personal_data_request"
          - self-harm/violence → blocked_category: "self_harm"
          - adult/NSFW content → blocked_category: "adult"
          - anything else off-topic → blocked_category: "not_finance"

        You must output ONLY a valid JSON object with this exact structure:
        {
          "allowed": true,
          "blocked_category": null,
          "blocked_reason": null
        }
        OR if blocked:
        {
          "allowed": false,
          "blocked_category": "category_name",
          "blocked_reason": "brief reason"
        }

        Do not include any additional text, explanations, or formatting. Output only the JSON object.
    """

output_guardrail_instructions = """
You are a safety and compliance checker for a regulated financial assistant. You will receive an assistant's response as input. Your task is to evaluate whether it is safe and compliant.

The assistant's response you need to evaluate will be provided to you directly. Analyze it according to the criteria below.

Evaluation Criteria:

1. **Severe Issues** (mark as unsafe if found):
   - Illegal or harmful advice (tax evasion, fraud, gambling promotion)
   - Actionable instructions that violate laws or ethical policies
   - Privacy violations (exposing personal identifiable information)

   Issue codes: "illegal_advice", "privacy_leak", "actionable_illegal_instruction"

2. **Softer Compliance Issues** (note but don't mark as unsafe):
   - Informal or unprofessional tone: "tone_informal"
   - Missing required disclaimers: "missing_disclaimer"
   - Absolute guarantees or unrealistic claims: "guarantee_statement"
   - Biased or political content: "political_advocacy"

Decision Rules:
- Set "safe": true if there are NO severe issues (softer issues are acceptable)
- Set "safe": false only if severe issues exist that make the response unsuitable for users
- List all detected issues (both severe and soft) in the "issues" field as a comma-separated string, or set to null if none

You must output ONLY a valid JSON object with this exact structure:
{
  "safe": true,
  "issues": null
}
OR if issues detected:
{
  "safe": false,
  "issues": "illegal_advice, privacy_leak"
}

Do not include any additional text, explanations, or formatting. Output only the JSON object.
"""

# output_guardrail_instructions = """
# You are a safety and compliance evaluator for a regulated financial assistant.

# Input: The assistant's final response (a string).

# Your tasks:

# 1. **Detect Severe Compliance Issues**  
#    Carefully review the assistant's response for any of the following severe violations:
#    - Illegal advice (e.g., how to commit fraud, evade taxes).
#    - Actionable instructions for unlawful or unethical behavior.
#    - Disclosure of private or personal user information (PII).
#    - Promoting or instructing on gambling or other illegal activity.

#    If any are found, set `"safe": false` and include the relevant issue codes:
#    - "illegal_advice"
#    - "privacy_leak"
#    - "actionable_illegal_instruction"
#    - "gambling_promotion"

# 2. **Detect Softer Issues (Non-Severe but Important)**  
#    Check for softer policy or tone issues that do not warrant blocking the response, but may require post-processing. These include:
#    - Informal or unprofessional tone: `"tone_informal"`
#    - Missing required disclaimers (e.g., "This is not financial advice"): `"missing_disclaimer"`
#    - Overly confident or absolute guarantees: `"guarantee_statement"`
#    - Biased, political, or non-neutral content: `"political_advocacy"`

#    If only soft issues are present, set `"safe": true`, and include all relevant issue codes in `"issues"`.

# 3. **Final Decision Fields**  
#    - `safe`: 
#      - `true` if **no severe** compliance issues were found.
#      - `false` if **any severe** issue is found that makes the response unsafe to display as-is.
#    - `issues`: 
#      - A list of strings describing all detected issues (can be empty or null if none).
#    - Do **not** include any `rewritten_response`. The assistant will decide how to respond if unsafe.

# Your output must be a valid JSON object with exactly these fields:
# {
#     "safe": true | false,
#     "issues": list of strings or null
# }
# """


finorbit_instructions = """
        You are Finorbit, a professional financial planning assistant. You only answer within your capabilities:
        Financial Advisor, Financial Score, Credits and Loans, Investment Coaching, Insurance Analysis,
        Retirement Planning, Tax Planning, Fraud Protection, Financial Literacy, Portfolio Rebalancing.

        When tackling complex financial problems, use sequential thinking tool to break down the task into smaller, manageable steps. 
        This approach helps in:
        - Analyzing user requirements thoroughly before responding
        - Verifying if all required input data is present for making accurate assessments
        - Decomposing complex financial tasks or multi-step processes 
        - Evaluating different financial options systematically
        - Validating your reasoning and checking for errors in logic

        Before making any tool invocations, use sequential thinking to:
        - Check if all required input parameters are available from the user's query
        - Identify any missing information that would be needed for the tool to function correctly
        - Determine if you need to ask the user for additional data before proceeding
        - Validate that the input data is in the correct format and reasonable range
        - Consider edge cases or potential errors that might occur with the given inputs


        Be concise, professional, neutral and avoid making absolute guarantees. When offering investment or tax information,
        include an appropriate short disclaimer (e.g., "This is educational and not financial advice.") unless the user explicitly
        requests otherwise for non-advice content.
    """

if use_custom_models:
# Input guardrail function
    BASE_URL = os.getenv("BASE_URL") or ""
    API_KEY = os.getenv("LLM_API_KEY") or ""
    MODEL_NAME = os.getenv("CUSTOM_MODEL_NAME") or ""

    # Initialize LLM client (OpenAI-compatible)
    set_tracing_disabled(disabled=True)

    CUSTOM_MODEL_PROVIDER = CustomModelProvider()

    run_config=RunConfig(
        model_provider=CUSTOM_MODEL_PROVIDER,
    )
    
    @input_guardrail
    async def finance_input_guardrail(
        ctx: RunContextWrapper[None],
        agent: Agent,
        input: str | list[TResponseInputItem],
    ) -> GuardrailFunctionOutput:
        """
        Runs the input_guardrail_agent on the provided input and trips if the query is not allowed.
        """
        if isinstance(input, list):
            try:
                user_text = " ".join([item.content for item in input if hasattr(item, "content")])
            except Exception:
                user_text = str(input)
        else:
            user_text = input

        result = await Runner.run(
            input_guardrail_agent,
            user_text,
            context=ctx.context,
            run_config=run_config,
        )
        logging.info(f"Input guardrail agent output: {result.final_output}")
        if isinstance(result.final_output, str):
            logging.info(f"output type: str")
            guard_info: InputGuardrailResult = fix_json_str(result.final_output)

        elif isinstance(result.final_output, InputGuardrailResult):
            logging.info(f"output type: InputGuardrailResult")
            guard_info = result.final_output
        
        logging.info(f"Parsed (pydantic) input guardrail result: {guard_info}")

        # Convert to GuardrailResult and enforce
        guard_result = GuardrailResult(
            allowed=guard_info.allowed,
            blocked_category=guard_info.blocked_category,
            blocked_reason=guard_info.blocked_reason
        )
        return enforce_input_guardrail(guard_result)

    # Output guardrail function
    @output_guardrail
    async def finance_output_guardrail(
        ctx: RunContextWrapper,
        agent: Agent,
        output: str | AssistantMessage,
    ) -> GuardrailFunctionOutput:
        """ Runs the output guardrail agent on the final assistant response. Trips only for severe issues (illegal instructions, privacy leaks, etc.) but returns the full analysis in output_info for logging & post-processing. """
        # Extract text from AssistantMessage if needed
        if isinstance(output, AssistantMessage):
            output_text = output.response
        else:
            output_text = output

        result = await Runner.run(
            output_guardrail_agent,
            output_text,
            context=ctx.context,
            run_config=run_config,
        )
        logging.info(f"Output guardrail agent output: {result.final_output}")

        if isinstance(result.final_output, str):
            logging.info(f"output type: str")
            guard_info: OutputGuardrailResult = fix_json_str(result.final_output)

        elif isinstance(result.final_output, OutputGuardrailResult):
            logging.info(f"output type: OutputGuardrailResult")
            guard_info = result.final_output

        logging.info(f"Parsed (pydantic) output guardrail result: {guard_info}")

        # Convert to GuardrailResult and enforce
        guard_result = map_output_guardrail_to_result({"safe": guard_info.safe, "issues": guard_info.issues})
        return enforce_output_guardrail(guard_result)

else:

# Input guardrail function
    @input_guardrail
    async def finance_input_guardrail(
        ctx: RunContextWrapper[None],
        agent: Agent,
        input: str | list[TResponseInputItem],
    ) -> GuardrailFunctionOutput:
        """
        Runs the input_guardrail_agent on the provided input and trips if the query is not allowed.
        """
        if isinstance(input, list):
            try:
                user_text = " ".join([item.content for item in input if hasattr(item, "content")])
            except Exception:
                user_text = str(input)
        else:
            user_text = input

        result = await Runner.run(input_guardrail_agent, user_text, context=ctx.context)
        logging.info(f"Input guardrail agent output: {result.final_output}")
        if isinstance(result.final_output, str):
            logging.info(f"output type: str")
            guard_info: InputGuardrailResult = fix_json_str(result.final_output)

        elif isinstance(result.final_output, InputGuardrailResult):
            logging.info(f"output type: InputGuardrailResult")
            guard_info = result.final_output

        logging.info(f"Parsed (pydantic) input guardrail result: {guard_info}")

        # Convert to GuardrailResult and enforce
        guard_result = GuardrailResult(
            allowed=guard_info.allowed,
            blocked_category=guard_info.blocked_category,
            blocked_reason=guard_info.blocked_reason
        )
        return enforce_input_guardrail(guard_result)


    # Output guardrail function
    @output_guardrail
    async def finance_output_guardrail(
        ctx: RunContextWrapper,
        agent: Agent,
        output: str,
    ) -> GuardrailFunctionOutput:
        """ Runs the output guardrail agent on the final assistant response. Trips only for severe issues (illegal instructions, privacy leaks, etc.) but returns the full analysis in output_info for logging & post-processing. """
        result = await Runner.run(output_guardrail_agent, output, context=ctx.context)
        logging.info(f"Output guardrail agent output: {result.final_output}")
        
        if isinstance(result.final_output, str):
            logging.info(f"output type: str")
            guard_info: OutputGuardrailResult = fix_json_str(result.final_output)

        elif isinstance(result.final_output, OutputGuardrailResult):
            logging.info(f"output type: OutputGuardrailResult")
            guard_info = result.final_output

        logging.info(f"Parsed (pydantic) output guardrail result: {guard_info}")

        # Convert to GuardrailResult and enforce
        guard_result = map_output_guardrail_to_result({"safe": guard_info.safe, "issues": guard_info.issues})
        return enforce_output_guardrail(guard_result)

# Define model settings
output_fast_model_settings = ModelSettings( max_tokens=500)
# fast_model_settings = ModelSettings( max_tokens=500, reasoning=Reasoning(effort="low"),verbosity="low",)

if use_custom_models:
    input_guardrail_agent = Agent(
    name="Finorbit Input Guardrail Agent",
    instructions=input_guardrail_instructions ,
    output_type=InputGuardrailResult,
    )

    # Output guardrail agent
    output_guardrail_agent = Agent(
        name="Finorbit Output Guardrail Agent",
        instructions=output_guardrail_instructions,
        output_type=OutputGuardrailResult,
    )
    finance_agent = Agent(
        name="Finorbit - Comprehensive Financial Assistant",
        instructions=finorbit_instructions,
        mcp_servers=[],
        input_guardrails=[finance_input_guardrail],
        output_guardrails=[finance_output_guardrail],
        output_type=AssistantMessage,
    )

else : 
    input_guardrail_agent = Agent(
        name="Finorbit Input Guardrail Agent",
        instructions=input_guardrail_instructions ,
        output_type=InputGuardrailResult,
        # model="gpt-5-mini",
        model_settings=output_fast_model_settings,
    )

    # Output guardrail agent
    output_guardrail_agent = Agent(
        name="Finorbit Output Guardrail Agent",
        instructions=output_guardrail_instructions,
        output_type=OutputGuardrailResult,
        # model="gpt-5-mini",
        # model_settings=output_fast_model_settings,
    )
    finance_agent = Agent(
        name="Finorbit - Comprehensive Financial Assistant",
        instructions=finorbit_instructions,
        # model="gpt-5",
        mcp_servers=[],
        model_settings=ModelSettings(tool_choice="auto", max_tokens=4000),
        # model_settings=ModelSettings(tool_choice="auto", max_tokens=4000,reasoning=Reasoning(effort="medium")),
        input_guardrails=[finance_input_guardrail],
        output_guardrails=[finance_output_guardrail],
        # output_type=AssistantMessage,
    )