"""The fixed list of source documents fetched for the corpus.

This is the single source of truth for what gets downloaded by
``scripts/fetch_corpus.py``. See ``documentations/decisions/ADR-0001-corpus-and-sourcing.md``
for why these specific documents were chosen.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, HttpUrl

Language = Literal["en", "fr"]
MediaType = Literal["html", "pdf"]
Category = Literal["regulation", "guideline", "code_of_practice"]


class CorpusDocument(BaseModel):
    """One fetchable source document and the metadata needed to cite it."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    url: HttpUrl
    language: Language
    media_type: MediaType
    category: Category
    source_organization: str
    celex: str | None = None
    published: str
    """ISO 8601 date the source document was adopted or published."""
    request_headers: dict[str, str] = {}
    """Extra headers merged into the GET request, for example Cellar content
    negotiation (``Accept-Language``, ``Accept``); see ADR-0001 addendum."""


CELLAR_ACCEPT_HEADER = "application/xhtml+xml, text/html;q=0.9"
"""EUR-Lex's own HTML pages sit behind an AWS WAF bot challenge that blocks
plain HTTP clients regardless of User-Agent (see ADR-0001 addendum, 2026-07-13).
The Publications Office's Cellar repository, at the same CELEX identifier,
serves the identical official text via content negotiation and is not
behind that challenge."""


def _cellar_url(celex: str) -> str:
    return f"https://publications.europa.eu/resource/celex/{celex}"


CORPUS_MANIFEST: tuple[CorpusDocument, ...] = (
    CorpusDocument(
        doc_id="gdpr_en",
        title="Regulation (EU) 2016/679 (General Data Protection Regulation)",
        url=_cellar_url("32016R0679"),
        language="en",
        media_type="html",
        category="regulation",
        source_organization="EUR-Lex (via Cellar)",
        celex="32016R0679",
        published="2016-04-27",
        request_headers={"Accept-Language": "eng", "Accept": CELLAR_ACCEPT_HEADER},
    ),
    CorpusDocument(
        doc_id="gdpr_fr",
        title="Reglement (UE) 2016/679 (Reglement general sur la protection des donnees)",
        url=_cellar_url("32016R0679"),
        language="fr",
        media_type="html",
        category="regulation",
        source_organization="EUR-Lex (via Cellar)",
        celex="32016R0679",
        published="2016-04-27",
        request_headers={"Accept-Language": "fra", "Accept": CELLAR_ACCEPT_HEADER},
    ),
    CorpusDocument(
        doc_id="ai_act_en",
        title="Regulation (EU) 2024/1689 (Artificial Intelligence Act)",
        url=_cellar_url("32024R1689"),
        language="en",
        media_type="html",
        category="regulation",
        source_organization="EUR-Lex (via Cellar)",
        celex="32024R1689",
        published="2024-06-13",
        request_headers={"Accept-Language": "eng", "Accept": CELLAR_ACCEPT_HEADER},
    ),
    CorpusDocument(
        doc_id="ai_act_fr",
        title="Reglement (UE) 2024/1689 (reglement sur l'intelligence artificielle)",
        url=_cellar_url("32024R1689"),
        language="fr",
        media_type="html",
        category="regulation",
        source_organization="EUR-Lex (via Cellar)",
        celex="32024R1689",
        published="2024-06-13",
        request_headers={"Accept-Language": "fra", "Accept": CELLAR_ACCEPT_HEADER},
    ),
    CorpusDocument(
        doc_id="edpb_consent",
        title="Guidelines 05/2020 on consent under Regulation 2016/679",
        url="https://www.edpb.europa.eu/system/files/documents/files/file1/edpb_guidelines_202005_consent_en.pdf",
        language="en",
        media_type="pdf",
        category="guideline",
        source_organization="EDPB",
        published="2020-05-04",
    ),
    CorpusDocument(
        doc_id="edpb_transparency",
        title="Guidelines on transparency under Regulation 2016/679 (WP260 rev.01)",
        url="https://www.edpb.europa.eu/system/files/documents/2023-09/wp260rev01_en.pdf",
        language="en",
        media_type="pdf",
        category="guideline",
        source_organization="EDPB",
        published="2018-04-11",
    ),
    CorpusDocument(
        doc_id="edpb_automated_decision_making",
        title=(
            "Guidelines on Automated individual decision-making and Profiling "
            "for the purposes of Regulation 2016/679 (WP251 rev.01)"
        ),
        url="https://ec.europa.eu/newsroom/article29/redirection/document/49826",
        language="en",
        media_type="pdf",
        category="guideline",
        source_organization="EDPB",
        published="2018-02-06",
    ),
    CorpusDocument(
        doc_id="edpb_dpia",
        title=("Guidelines on Data Protection Impact Assessment (DPIA) (WP248 rev.01)"),
        url="https://ec.europa.eu/newsroom/just/document.cfm?doc_id=47711",
        language="en",
        media_type="pdf",
        category="guideline",
        source_organization="EDPB",
        published="2017-10-04",
    ),
    CorpusDocument(
        doc_id="edpb_right_to_be_forgotten",
        title=(
            "Guidelines 5/2019 on the criteria of the Right to be Forgotten "
            "in the search engines cases under the GDPR"
        ),
        url="https://www.edpb.europa.eu/sites/default/files/files/file1/edpb_guidelines_201905_rtbfsearchengines_afterpublicconsultation_en.pdf",
        language="en",
        media_type="pdf",
        category="guideline",
        source_organization="EDPB",
        published="2020-07-07",
    ),
    CorpusDocument(
        doc_id="edpb_data_protection_by_design",
        title="Guidelines 4/2019 on Article 25 Data Protection by Design and by Default",
        url="https://www.edpb.europa.eu/sites/default/files/files/file1/edpb_guidelines_201904_dataprotection_by_design_and_by_default_v2.0_en.pdf",
        language="en",
        media_type="pdf",
        category="guideline",
        source_organization="EDPB",
        published="2020-10-20",
    ),
    CorpusDocument(
        doc_id="edpb_legitimate_interest",
        title="Guidelines 1/2024 on processing of personal data based on legitimate interest",
        url="https://www.edpb.europa.eu/system/files/2024-10/edpb_guidelines_202401_legitimateinterest_en.pdf",
        language="en",
        media_type="pdf",
        category="guideline",
        source_organization="EDPB",
        published="2024-10-08",
    ),
    CorpusDocument(
        doc_id="ai_act_prohibited_practices",
        title="Commission Guidelines on prohibited artificial intelligence practices (AI Act)",
        url=(
            "https://ai-act-service-desk.ec.europa.eu/sites/default/files/2025-08/"
            "guidelines_on_prohibited_artificial_intelligence_practices_established_by_"
            "regulation_eu_20241689_ai_act_english_ied3r5nwo50xggpcfmwckm3nuc_112367-1.PDF"
        ),
        language="en",
        media_type="pdf",
        category="guideline",
        source_organization="European Commission",
        published="2025-02-04",
    ),
    CorpusDocument(
        doc_id="ai_act_system_definition",
        title="Commission Guidelines on the definition of an AI system (AI Act)",
        url=(
            "https://ai-act-service-desk.ec.europa.eu/sites/default/files/2025-08/"
            "commission_guidelines_on_the_definition_of_an_artificial_intelligence_system_"
            "established_by_regulation_eu_20241689_ai_actenglish_nf2skcqfrtjdfggjavcodopcwz4_112455.PDF"
        ),
        language="en",
        media_type="pdf",
        category="guideline",
        source_organization="European Commission",
        published="2025-02-06",
    ),
    CorpusDocument(
        doc_id="gpai_cop_transparency",
        title="General-Purpose AI Code of Practice, Transparency chapter",
        url="https://ec.europa.eu/newsroom/dae/redirection/document/118120",
        language="en",
        media_type="pdf",
        category="code_of_practice",
        source_organization="European Commission",
        published="2025-07-10",
    ),
    CorpusDocument(
        doc_id="gpai_cop_copyright",
        title="General-Purpose AI Code of Practice, Copyright chapter",
        url="https://ec.europa.eu/newsroom/dae/redirection/document/118115",
        language="en",
        media_type="pdf",
        category="code_of_practice",
        source_organization="European Commission",
        published="2025-07-10",
    ),
    CorpusDocument(
        doc_id="gpai_cop_safety_and_security",
        title="General-Purpose AI Code of Practice, Safety and Security chapter",
        url="https://ec.europa.eu/newsroom/dae/redirection/document/118119",
        language="en",
        media_type="pdf",
        category="code_of_practice",
        source_organization="European Commission",
        published="2025-07-10",
    ),
)
