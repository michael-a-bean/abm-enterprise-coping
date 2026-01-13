"""Prompt templates for LLM-based policy decisions.

This module defines the prompt templates used to query LLMs
for household enterprise decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

from abm_enterprise.data.schemas import EnterpriseStatus, HouseholdState


@dataclass
class PromptConfig:
    """Configuration for prompt generation.

    Attributes:
        country: Country context (tanzania or ethiopia).
        include_literature_context: Whether to include economic literature context.
        include_examples: Whether to include example responses.
    """

    country: str = "tanzania"
    include_literature_context: bool = True
    include_examples: bool = False


# Main prompt template for enterprise decisions
ENTERPRISE_DECISION_TEMPLATE = """You are simulating a household's enterprise decision in rural {country_name}.

HOUSEHOLD STATE:
- ID: {household_id}
- Wave: {wave}
- Assets Index: {assets_index:.3f} (standardized, 0 = median, positive = above average)
- Credit Access: {credit_access}
- Current Enterprise Status: {enterprise_status}
- Price Exposure: {price_exposure:.3f} (negative = price bust/adverse shock)

{literature_context}
What action should this household take?
Respond with exactly one of:
- ACTION: ENTER_ENTERPRISE
- ACTION: EXIT_ENTERPRISE
- ACTION: NO_CHANGE

Your response:"""

LITERATURE_CONTEXT_TEMPLATE = """Based on the economic literature on household coping strategies:
- Households facing negative price shocks may enter non-farm enterprise as a coping mechanism
- Low-asset households are more likely to use enterprise as a coping strategy
- Credit-constrained households (no credit access) may be forced to cope through enterprise
- Enterprise entry is a "push" factor - households enter out of necessity, not opportunity
- Stayers (persistent enterprise operators) rarely exit even during shocks
"""

NO_LITERATURE_CONTEXT = """Consider the household's economic constraints and make a decision.
"""

EXAMPLE_RESPONSES = """
Examples of valid responses:
1. For a household with adverse price shock (price_exposure < -0.1) and low assets:
   ACTION: ENTER_ENTERPRISE

2. For a stable household with positive price exposure:
   ACTION: NO_CHANGE

3. For a household in distress with very low assets and in enterprise:
   ACTION: EXIT_ENTERPRISE
"""


def get_country_name(country: str) -> str:
    """Convert country code to full name.

    Args:
        country: Country code (tanzania or ethiopia).

    Returns:
        Full country name.
    """
    country_names = {
        "tanzania": "Tanzania",
        "ethiopia": "Ethiopia",
    }
    return country_names.get(country.lower(), country.title())


def get_enterprise_status_text(status: EnterpriseStatus) -> str:
    """Convert enterprise status to human-readable text.

    Args:
        status: Enterprise status enum.

    Returns:
        Human-readable status description.
    """
    if status == EnterpriseStatus.HAS_ENTERPRISE:
        return "Currently operates enterprise"
    else:
        return "Does not operate enterprise"


def get_credit_access_text(credit_access: int) -> str:
    """Convert credit access indicator to text.

    Args:
        credit_access: Credit access indicator (0 or 1).

    Returns:
        Human-readable credit access description.
    """
    if credit_access == 1:
        return "Has access to credit"
    else:
        return "No credit access"


def build_prompt(
    state: HouseholdState,
    config: PromptConfig | None = None,
) -> str:
    """Build a prompt for LLM decision from household state.

    Args:
        state: Current household state.
        config: Optional prompt configuration.

    Returns:
        Formatted prompt string for LLM.
    """
    if config is None:
        config = PromptConfig()

    # Build literature context
    if config.include_literature_context:
        literature_context = LITERATURE_CONTEXT_TEMPLATE
    else:
        literature_context = NO_LITERATURE_CONTEXT

    if config.include_examples:
        literature_context += "\n" + EXAMPLE_RESPONSES

    # Format the prompt
    prompt = ENTERPRISE_DECISION_TEMPLATE.format(
        country_name=get_country_name(config.country),
        household_id=state.household_id,
        wave=state.wave,
        assets_index=state.assets,
        credit_access=get_credit_access_text(state.credit_access),
        enterprise_status=get_enterprise_status_text(state.enterprise_status),
        price_exposure=state.price_exposure,
        literature_context=literature_context,
    )

    return prompt


def build_simple_prompt(state: HouseholdState) -> str:
    """Build a minimal prompt for testing/stub providers.

    Args:
        state: Current household state.

    Returns:
        Simple formatted prompt string.
    """
    return f"""Household {state.household_id} (Wave {state.wave}):
Assets: {state.assets:.3f}
Credit: {state.credit_access}
Enterprise: {state.enterprise_status.value}
Price Exposure: {state.price_exposure:.3f}

ACTION:"""


def parse_action_from_response(response: str) -> str | None:
    """Parse action from LLM response.

    Looks for patterns like "ACTION: ENTER_ENTERPRISE" in the response.

    Args:
        response: Raw LLM response text.

    Returns:
        Parsed action string or None if not found.
    """
    import re

    # Look for ACTION: pattern (case insensitive)
    pattern = r"ACTION:\s*(ENTER_ENTERPRISE|EXIT_ENTERPRISE|NO_CHANGE)"
    match = re.search(pattern, response.upper())

    if match:
        return match.group(1)

    # Fallback: look for just the action keywords
    response_upper = response.upper()
    if "ENTER_ENTERPRISE" in response_upper:
        return "ENTER_ENTERPRISE"
    elif "EXIT_ENTERPRISE" in response_upper:
        return "EXIT_ENTERPRISE"
    elif "NO_CHANGE" in response_upper:
        return "NO_CHANGE"

    return None
