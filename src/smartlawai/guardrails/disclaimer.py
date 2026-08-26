"""Bar Council of India disclaimer (Advocates Act, 1961) — mandatory."""
BCI_DISCLAIMER = (
    "This output is AI-generated legal information, NOT legal advice, and does not "
    "create an advocate-client relationship. Consult an advocate enrolled under the "
    "Advocates Act, 1961 before acting."
)


def attach(text: str) -> str:
    return f"{text}\n\n---\n*{BCI_DISCLAIMER}*"
