import pytest

from synapse_mcp.data_pipeline.semantic_integrator import SemanticIntegrator


@pytest.mark.asyncio
async def test_clean_text_academic_noise():
    integrator = SemanticIntegrator()

    # Test cases: (input, expected_substring)
    test_cases = [
        (
            "The model (CID:123) achieved 95% accuracy.",
            "The model achieved 95 accuracy",
        ),
        (
            "We use <span class='math'>E=mc^2</span> for calculations.",
            "We use Emc2 for calculations",
        ),
        (
            "As shown in \\textbf{Figure 1}, the results vary.",
            "As shown in Figure 1, the results vary",
        ),
        (
            "The energy equation $$ E = h \\nu $$ is fundamental.",
            "The energy equation is fundamental",
        ),
        (
            "Citations [1, 2] and (Author, 2020) should be removed.",
            "Citations and should be removed",
        ),
        ("Multiple citations [3-5] and \\cite{paper123}.", "Multiple citations and"),
    ]

    for input_text, expected_sub in test_cases:
        cleaned = await integrator._clean_text(input_text)
        assert expected_sub in cleaned
        assert "(CID:" not in cleaned
        assert "<span" not in cleaned
        assert "$$" not in cleaned
        assert "\\" not in cleaned
        assert "[" not in cleaned


@pytest.mark.asyncio
async def test_clean_text_punctuation_handling():
    integrator = SemanticIntegrator()

    # Noise stripping might leave leading commas or periods
    input_text = "\\cite{noise}, but this part is important."
    cleaned = await integrator._clean_text(input_text)
    assert cleaned.startswith("but this part is important")

    input_text = "...(CID:123) The end."
    cleaned = await integrator._clean_text(input_text)
    assert cleaned == "The end."
