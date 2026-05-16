"""
Montague Grammar-based semantic parser for Project Synapse.

This module implements the Semantic Blueprint component, providing
formal semantic analysis and logical form generation for natural language.
"""

import re

# trunk-ignore(bandit/B404)
import subprocess
import sys
from typing import Any

import spacy  # type: ignore[import-untyped]
from spacy.language import Language  # type: ignore[import-untyped]
from spacy.tokens import Doc, Span, Token  # type: ignore[import-untyped]

from ..knowledge.knowledge_types import KnowledgeUtils
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class MontagueParser:
    """
    Montague Grammar-based semantic parser.

    Implements formal semantic analysis using compositional semantics
    and lambda calculus for precise meaning representation.
    """

    def __init__(self) -> None:
        self.nlp: Language | None = None
        self.entity_types = {
            "PERSON",
            "ORG",
            "GPE",
            "LOC",
            "PRODUCT",
            "EVENT",
            "WORK_OF_ART",
            "LAW",
            "LANGUAGE",
            "CONCEPT",
            "METHOD",
        }
        self.logger = logger

    @logger.timer()
    async def initialize(self) -> None:
        """Initialize the semantic parser with spaCy models."""
        try:
            # Load spaCy model for English
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Montague parser initialized successfully")

        except OSError:
            logger.warning("spaCy model not found, downloading...")
            # In production, this should be handled during setup
            # trunk-ignore(bandit/B603)
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
            )
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Downloaded and loaded spaCy model")

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to initialize Montague parser: %s", e)
            raise

    @logger.timer()
    async def parse_text(self, text: str) -> dict[str, Any]:
        """
        Parse text using Montague Grammar principles.

        Args:
            text: Input text to parse

        Returns:
            Dictionary containing semantic analysis results
        """
        if not self.nlp:
            raise ValueError("Parser not initialized")

        logger.debug("Parsing text: %s...", text[:100])

        # Process with spaCy
        doc = self.nlp(text)

        # Extract semantic components
        analysis = {
            "original_text": text,
            "entities": await self._extract_entities(doc),
            "relations": await self._extract_relations(doc),
            "logical_form": await self._generate_logical_form(doc),
            "semantic_features": await self._extract_semantic_features(doc),
            "propositions": await self._extract_propositions(doc),
        }

        logger.debug(
            "Extracted %d entities, %d relations",
            len(analysis["entities"]),
            len(analysis["relations"]),
        )

        return analysis

    @logger.timer()
    async def _extract_entities(self, doc: Doc) -> list[dict[str, Any]]:
        """Extract named entities with type and confidence information."""
        entities = []

        for ent in doc.ents:
            if not self._is_valid_endpoint(ent.text):
                continue

            # Calculate confidence based on entity characteristics
            confidence = self._calculate_entity_confidence(ent)

            # Initial normalization
            entity_type = self._normalize_entity_type(ent.label_)

            # Refine type based on heuristics (Phase 3 upgrade)
            entity_type = self._refine_entity_type(ent.text, entity_type)

            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "type": entity_type,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": confidence,
                    # Generate the ID using the same function + same NORMALIZED
                    # type the storage layer uses, so the IDs match across the
                    # entity-storage path and the proposition path. Using the
                    # raw spaCy label here would diverge from the canonical
                    # ``organization_*`` form and silently break MENTIONS.
                    "id": KnowledgeUtils.generate_entity_id(ent.text, entity_type),
                }
            )

        return entities

    @logger.timer()
    async def _extract_relations(self, doc: Doc) -> list[dict[str, Any]]:
        """Extract semantic relations between entities and key noun phrases."""
        relations = []

        # Extract noun chunks for richer relationship detection
        noun_chunks = {chunk.root.i: chunk.text for chunk in doc.noun_chunks}

        # Strategy 1: Subject-Verb-Object patterns
        for token in doc:
            if token.dep_ in ["nsubj", "nsubjpass"]:
                head = token.head

                # Get subject (prefer entity, fall back to noun chunk)
                subject = self._find_entity_for_token(doc, token)
                if not subject:
                    subject = self._get_chunk_endpoint(noun_chunks, token.i) or ""

                if not subject:
                    continue

                # Get predicate
                predicate = head.lemma_ if head.pos_ == "VERB" else head.text

                # Find object
                obj = None
                for child in head.children:
                    if child.dep_ in ["dobj", "attr", "pobj"]:
                        obj = self._find_entity_for_token(doc, child)
                        if not obj:
                            obj = self._get_chunk_endpoint(noun_chunks, child.i)
                        if obj:
                            break

                if subject and obj:
                    relations.append(
                        {
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj,
                            "confidence": 0.8,
                            "source_span": f"{token.i}-{head.i}",
                        }
                    )

        # Strategy 2: Copula/Linking verb patterns (is, are, becomes, equals)
        for token in doc:
            if token.lemma_ in ["be", "become", "equal", "represent", "constitute"]:
                # Find subject
                cop_subject: str | None = None
                cop_subj_idx: int | None = None
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        cop_subject = self._find_entity_for_token(doc, child) or None
                        if not cop_subject:
                            cop_subject = self._get_chunk_endpoint(noun_chunks, child.i)
                        cop_subj_idx = child.i
                        break

                # Find complement/attribute
                cop_obj: str | None = None
                cop_obj_idx: int | None = None
                for child in token.children:
                    if child.dep_ in ["attr", "acomp", "dobj"]:
                        cop_obj = self._find_entity_for_token(doc, child) or None
                        if not cop_obj:
                            cop_obj = self._get_chunk_endpoint(noun_chunks, child.i)
                        cop_obj_idx = child.i
                        break

                if cop_subject and cop_obj:
                    span_lo = cop_subj_idx if cop_subj_idx is not None else token.i
                    span_hi = cop_obj_idx if cop_obj_idx is not None else token.i
                    relations.append(
                        {
                            "subject": cop_subject,
                            "predicate": f"is-{token.lemma_}",
                            "object": cop_obj,
                            "confidence": 0.75,
                            "source_span": f"{span_lo}-{span_hi}",
                        }
                    )

        # Strategy 3: Prepositional relationships
        for token in doc:
            if token.dep_ == "prep" and token.head.pos_ in ["NOUN", "PROPN", "VERB"]:
                # Get the noun being modified
                subject = self._find_entity_for_token(doc, token.head)
                if not subject:
                    subject = self._get_chunk_endpoint(noun_chunks, token.head.i) or ""

                # Get the object of the preposition
                obj = None
                for child in token.children:
                    if child.dep_ == "pobj":
                        obj = self._find_entity_for_token(doc, child)
                        if not obj:
                            obj = self._get_chunk_endpoint(noun_chunks, child.i)
                        break

                if subject and obj:
                    relations.append(
                        {
                            "subject": subject,
                            "predicate": f"relates-via-{token.text}",
                            "object": obj,
                            "confidence": 0.7,
                            "source_span": f"{token.head.i}-{token.i}",
                        }
                    )

        # Strategy 4 (compound/possessive patterns) was removed deliberately.
        # In dependency-parser output, *most* compound and poss edges connect
        # tokens like "purpose"->"general-purpose executor" or "scale"->
        # "large-scale deployments" — these emit `modifies` edges that pollute
        # the graph without carrying real semantic content. The LLM extractor
        # captures genuine composition relationships (`is_component_of`,
        # `instance_of`, etc.) with much better precision, so the right answer
        # is to let those carry the signal and not duplicate them with a
        # syntactic heuristic that's wrong far more often than it's right.

        # ----- Quality filter, same-sentence guard, and deduplication -----
        # The same-sentence guard catches relations whose endpoint tokens ended
        # up in different sentences. This shouldn't normally happen because
        # the dependency parser is per-sentence, but malformed text (collapsed
        # tables, lists, captions) can produce spurious cross-sentence links
        # like "seven different retailers --relates-via-in--> the heart".
        filtered: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for rel in relations:
            # Safely handle source_span which might be missing or of unexpected type
            span_raw = rel.get("source_span", "")
            if isinstance(span_raw, str) and "-" in span_raw:
                try:
                    t1_idx, t2_idx = (int(x) for x in span_raw.split("-", 1))
                    if 0 <= t1_idx < len(doc) and 0 <= t2_idx < len(doc):
                        if doc[t1_idx].sent.start != doc[t2_idx].sent.start:
                            continue
                except (ValueError, IndexError):
                    pass  # if we can't parse the span, don't filter on it

            subj = str(rel["subject"])
            obj = str(rel["object"])
            pred = str(rel["predicate"])
            if not self._is_valid_endpoint(subj) or not self._is_valid_endpoint(obj):
                continue
            key = (subj.lower(), pred.lower(), obj.lower())
            if key in seen:
                continue
            seen.add(key)
            filtered.append(rel)

        logger.debug(
            "Relation extraction: %d raw -> %d after quality filter",
            len(relations),
            len(filtered),
        )
        return filtered

    @logger.timer()
    async def _generate_logical_form(self, doc: Doc) -> str:
        """
        Generate logical form representation using lambda calculus.

        This is a simplified implementation of Montague Grammar principles.
        """
        logical_forms = []

        for sent in doc.sents:
            # Extract the main components
            subject, verb, obj = self._extract_svo_pattern(sent)

            if subject and verb:
                if obj:
                    # Binary predicate: R(x,y)
                    form = f"{verb.lemma_}({subject.text}, {obj.text})"
                else:
                    # Unary predicate: P(x)
                    form = f"{verb.lemma_}({subject.text})"

                logical_forms.append(form)

        return " ∧ ".join(logical_forms) if logical_forms else ""

    async def _extract_semantic_features(self, doc: Doc) -> dict[str, Any]:
        """Extract semantic features from the parsed document."""
        features = {
            "sentence_count": len(list(doc.sents)),
            "entity_count": len(doc.ents),
            "token_count": len(doc),
            "has_negation": any(token.dep_ == "neg" for token in doc),
            "tense": self._extract_tense(doc),
            "modality": self._extract_modality(doc),
            "sentiment_polarity": "neutral",  # Placeholder
        }

        return features

    async def _extract_propositions(self, doc: Doc) -> list[dict[str, Any]]:
        """Extract atomic propositions that can be stored as facts."""
        propositions = []

        for sent in doc.sents:
            # Each sentence potentially represents a proposition
            entities_in_sent = [
                ent
                for ent in doc.ents
                if ent.start >= sent.start and ent.end <= sent.end
            ]

            if entities_in_sent:
                # IMPORTANT: This list of entity IDs must match the IDs that
                # are produced when the same entity is stored. The storage
                # path uses ``KnowledgeUtils.generate_entity_id(name, type)``
                # with the NORMALIZED type (e.g. "Organization"). Using
                # spaCy's raw label here (e.g. "ORG") would produce a
                # different ID (``org_*`` vs ``organization_*``) and cause
                # ``_store_fact``'s MATCH-on-id to silently drop every
                # MENTIONS edge — exactly the bug that left thousands of
                # entities orphaned in earlier runs.
                proposition = {
                    "id": f"prop_{sent.start}_{sent.end}",
                    "content": sent.text.strip(),
                    "entities": [
                        KnowledgeUtils.generate_entity_id(
                            ent.text, self._normalize_entity_type(ent.label_)
                        )
                        for ent in entities_in_sent
                    ],
                    "confidence": 0.9,  # High confidence for direct extraction
                    "logical_form": await self._generate_logical_form_for_sentence(
                        sent
                    ),
                }
                propositions.append(proposition)

        return propositions

    def _calculate_entity_confidence(self, ent: Span) -> float:
        """Calculate confidence score for an entity."""
        base_confidence = 0.8

        # Boost confidence for known entity types
        if ent.label_ in self.entity_types:
            base_confidence += 0.1

        # Consider entity length (longer entities often more reliable)
        if len(ent.text) > 10:
            base_confidence += 0.05

        # Check if entity is capitalized (proper nouns)
        if ent.text[0].isupper():
            base_confidence += 0.05

        return min(base_confidence, 1.0)

    def _normalize_entity_type(self, spacy_label: str) -> str:
        """Normalize spaCy entity labels to our schema.

        Covers all 18 default spaCy NER labels. Anything that falls through
        becomes ``Entity`` (effectively untyped) — but everything in the
        OntoNotes label set is mapped, so untyped entities should only
        appear when a custom NER component emits a non-standard label.
        """
        mapping = {
            # People & organizations
            "PERSON": "Person",
            "ORG": "Organization",
            "NORP": "Demographic",  # nationalities, religious or political groups
            # Places
            "GPE": "Location",  # countries, cities, states
            "LOC": "Location",  # non-GPE locations (mountains, bodies of water)
            "FAC": "Location",  # facilities, buildings, airports, highways
            # Products & works
            "PRODUCT": "Product",
            "WORK_OF_ART": "CreativeWork",
            # Events & laws
            "EVENT": "Event",
            "LAW": "Law",
            # Languages
            "LANGUAGE": "Language",
            # Temporal
            "DATE": "TemporalEntity",
            "TIME": "TemporalEntity",
            # Numbers & quantities (all map to Quantity — they carry the
            # same kind of semantic weight: a numeric or quantitative value)
            "MONEY": "MonetaryValue",
            "QUANTITY": "Quantity",
            "PERCENT": "Quantity",
            "CARDINAL": "Quantity",
            "ORDINAL": "Quantity",
            # Custom labels from our extended NER (if present)
            "CONCEPT": "Concept",
            "METHOD": "Method",
        }

        return mapping.get(spacy_label, "Entity")

    @staticmethod
    def _looks_like_code_symbol(text: str) -> bool:
        """Detect identifiers that look like programming-language symbols.

        spaCy mislabels these as Organization or Product. We catch only HIGH-
        CONFIDENCE patterns so we don't reroute legitimate brand names:

        - snake_case (contains an underscore between alphanumerics)
        - SCREAMING_SNAKE_CASE (same, all caps)
        - CamelCase with >=2 internal capitals (catches FindOppositeSwing,
          IsSwingLow, OnInit-style; doesn't catch YouTube/iPhone/PyTorch
          which have at most one internal capital)

        Two-internal-cap names like 'BuildGraph' or 'SwingNode' are NOT
        caught here — those are a judgment call. The LLM extractor is the
        better defense for those via its prompt-level filter.
        """
        import re as _re

        stripped = text.strip()
        if _re.match(r"^[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]+$", stripped):
            return True
        if " " not in stripped and stripped[:1].isalpha() and len(stripped) >= 6:
            internal_caps = sum(1 for c in stripped[1:] if c.isupper())
            if internal_caps >= 2:
                return True
        return False

    @staticmethod
    def _refine_entity_type(text: str, current_type: str) -> str:
        """Refine entity type using heuristic rules for technical concepts."""
        text_lower = text.lower()

        # Code-symbol detection: spaCy frequently labels function/class names
        # like FindOppositeSwingBeforeStart, OnInit, build_graph as Organization
        # or Product. Reroute these to a dedicated CodeSymbol type so they're
        # queryable and clearly separated from real Organizations / Products.
        if MontagueParser._looks_like_code_symbol(text):
            return "CodeSymbol"

        # Keywords that strongly suggest a Concept or Method
        concept_keywords = [
            "learning",
            "intelligence",
            "principle",
            "theory",
            "model",
            "network",
            "forgetting",
            "memory",
            "consciousness",
            "entropy",
            "occupancy",
            "neuroscience",
            "architecture",
            "framework",
        ]
        method_keywords = [
            "algorithm",
            "protocol",
            "pipeline",
            "procedure",
            "technique",
            "method",
            "approach",
            "search",
            "indexing",
            "retrieval",
            "integration",
            "refinement",
            "optimization",
        ]

        # High-confidence Organizations often misidentified by NER
        org_names = [
            "openai",
            "google",
            "anthropic",
            "meta",
            "microsoft",
            "deepmind",
            "huggingface",
            "mistral",
            "cohere",
            "nvidia",
        ]

        # High-confidence Products often misidentified
        product_names = [
            "qwen",
            "gpt",
            "llama",
            "claude",
            "gemini",
            "mixtral",
            "starcoder",
            "phi",
            "gemma",
        ]

        # Only refine types that are frequently misidentified for technical terms
        if current_type in [
            "Organization",
            "Product",
            "Person",
            "Entity",
            "Concept",
            "Location",
        ]:
            # 1. Check for high-confidence organizations
            if any(org in text_lower for org in org_names):
                return "Organization"

            # 1.5. Check for high-confidence products (LLM models, tools)
            if any(prod in text_lower for prod in product_names):
                return "Product"

            # Numeric+unit combos like "50ms", "100x" are NOT products — they're measurements
            import re as _re_num

            if _re_num.match(r"^\d+[a-zA-Z]{1,3}$", text.strip()):
                return "Concept"

            # If it's a single word with numbers (e.g., "Qwen3", "GPT4"), likely a Product
            if len(text.split()) == 1 and any(c.isdigit() for c in text):
                # Only if it also has letters (to avoid pure numbers being products)
                if any(c.isalpha() for c in text):
                    return "Product"

            # 2. Check for technical Methods
            if any(kw in text_lower for kw in method_keywords):
                return "Method"

            # 3. Check for technical Concepts
            if any(kw in text_lower for kw in concept_keywords):
                return "Concept"

            # 4. Specific common AI acronyms/terms
            if text_lower in [
                "llm",
                "rag",
                "cnn",
                "rnn",
                "transformer",
                "bert",
                "rl",
                "rlhf",
                "dpo",
                "ppo",
            ]:
                return "Concept"

            # 5. Downgrade suspicious 'Person' classifications (e.g., single words, UI terms)
            if current_type == "Person":
                non_person_keywords = [
                    # UI / navigation terms
                    "wiki",
                    "start",
                    "config",
                    "table",
                    "content",
                    "index",
                    "file",
                    "error",
                    "log",
                    "menu",
                    "nav",
                    "search",
                    "home",
                    "login",
                    "logout",
                    "submit",
                    "cancel",
                    "close",
                    "open",
                    "save",
                    "edit",
                    "delete",
                    "view",
                    "share",
                    "download",
                    # Platforms / organizations (commonly misclassified)
                    "github",
                    "gitlab",
                    "bitbucket",
                    "npm",
                    "pypi",
                    "docker",
                    "kubernetes",
                    "aws",
                    "azure",
                    "gcp",
                    "heroku",
                    "vercel",
                    "netlify",
                    "cloudflare",
                    "stripe",
                    "paypal",
                    # Products / frameworks
                    "react",
                    "vue",
                    "angular",
                    "svelte",
                    "nextjs",
                    "nuxt",
                    "django",
                    "flask",
                    "fastapi",
                    "express",
                    "koa",
                    "rails",
                    "spring",
                    "tensorflow",
                    "pytorch",
                    "keras",
                    "hugging",
                    "grok",
                    "copilot",
                    "codex",
                    "chatgpt",
                    "claude",
                    "gemini",
                    # Abbreviations that are never people
                    "api",
                    "sdk",
                    "cli",
                    "gui",
                    "ide",
                    "url",
                    "uri",
                    "json",
                    "yaml",
                    "html",
                    "css",
                    "sql",
                    "nosql",
                    "rest",
                    "grpc",
                    "llm",
                    "ml",
                    "ai",
                    "nlp",
                    "cv",
                    # Academic / conference abbreviations
                    "iclr",
                    "icml",
                    "neurips",
                    "aaai",
                    "cvpr",
                    "acl",
                    "emnlp",
                    # Role suffixes (these are roles, not names)
                    "professor",
                    "director",
                    "manager",
                    "engineer",
                    "researcher",
                    "scientist",
                    "analyst",
                    "architect",
                    "designer",
                    "developer",
                ]
                if any(kw in text_lower for kw in non_person_keywords):
                    return "Concept"
                # Role suffixes: "X Engineer", "Y Director", etc. → not people
                role_suffixes = [
                    "engineer",
                    "director",
                    "manager",
                    "professor",
                    "researcher",
                    "scientist",
                    "analyst",
                    "architect",
                    "designer",
                    "developer",
                    "lead",
                    "head",
                    "chief",
                    "fellow",
                    "advisor",
                    "consultant",
                    "founder",
                    "ceo",
                    "cto",
                    "cfo",
                    "coo",
                    "vp",
                ]
                if any(text_lower.endswith(f" {s}") for s in role_suffixes):
                    return "Concept"
                # Abbreviations: 2-4 all-caps letters with no lowercase = org/concept
                if len(text) <= 5 and text.isupper() and text.isalpha():
                    return "Organization"
                # If it's a single word and has numbers (like Qwen3) or is all caps (like RL), it's not a person
                if len(text.split()) == 1:
                    if any(c.isdigit() for c in text) or text.isupper():
                        return "Concept"
                # Tier/rank/grade patterns: "Grade A", "Tier 2", "Rung 4" → Concept
                rank_prefixes = [
                    "grade",
                    "tier",
                    "rung",
                    "level",
                    "rank",
                    "stage",
                    "phase",
                    "step",
                    "class",
                    "category",
                    "band",
                    "rank",
                ]
                words = text.split()
                if len(words) == 2 and words[0].lower() in rank_prefixes:
                    return "Concept"
                # Past-tense verb patterns: "Library Published", "Bug Found" → Concept
                # Words ending in -ed, -ing that aren't surnames
                import re as _re

                if len(words) == 2 and _re.match(r"^[A-Z][a-z]+ed$", words[1]):
                    return "Concept"
                # Possessive names: "Peter Steinberger\'s" → strip possessive
                if text.endswith("'s") or text.endswith("'s"):
                    pass

        # === Type-agnostic entity quality checks ===
        _stripped = text.strip()
        _stripped_lower = _stripped.lower()

        # Reject entities that end with truncated conjunctions/ampersands
        if (
            _stripped.endswith("&")
            or _stripped_lower.endswith("and")
            or _stripped_lower.endswith("or")
        ):
            return "Concept"

        # Reject numeric+unit combos like "50ms", "100x" regardless of type
        import re as _re2

        if _re2.match(r"^\d+[a-zA-Z]{1,3}$", _stripped):
            return "Concept"

        # Reject generic verbs misclassified as Product (Recommended, Verify, etc.)
        if current_type == "Product":
            verb_words = {
                "recommended",
                "verify",
                "verified",
                "discovery",
                "required",
                "available",
                "included",
                "optional",
                "default",
                "enabled",
                "disabled",
                "installed",
                "running",
                "building",
                "testing",
            }
            if text_lower in verb_words or text_lower.rstrip("d") in verb_words:
                return "Concept"

        # Organization names with "Phase" or version-like patterns → Concept
        if current_type == "Organization":
            phase_words = ["phase", "stage", "step", "level", "version", "build"]
            words = _stripped.split()
            if len(words) == 2 and words[0].lower() in phase_words:
                return "Concept"
            # Truncated org names ending with "Nothing", "CI" without context
            if _stripped_lower.endswith("nothing") or _stripped_lower.endswith(" ci"):
                return "Concept"

        # Known platform/product names that are NOT people
        platform_names = {
            "product hunt",
            "asana",
            "max",
            "linear",
            "figma",
            "sketch",
            "notion",
            "slack",
            "trello",
            "vercel",
            "railway",
        }
        if text_lower in platform_names:
            return "Organization"

        return current_type

    def _generate_entity_id(self, text: str, label: str) -> str:
        """Generate unique identifier for an entity."""
        normalized = re.sub(r"[^a-zA-Z0-9]", "_", text.lower())
        return f"{label.lower()}_{normalized}"

    def _find_entity_for_token(self, doc: Doc, token: Token) -> str:
        """Find if a token is part of a named entity.

        Returns the entity text if found, empty string otherwise.
        Callers should fall back to noun_chunks for non-entity tokens.
        """
        for ent in doc.ents:
            if token.i >= ent.start and token.i < ent.end:
                entity_text = ent.text
                if not isinstance(entity_text, str):
                    raise TypeError(
                        f"Expected str for entity_text, got {type(entity_text)}"
                    )
                return entity_text
        return ""

    @staticmethod
    def _is_valid_endpoint(text: str) -> bool:
        """Check if a text string is a valid relation endpoint.

        Filters out pronouns, determiners, HTML/LaTeX fragments,
        overly long strings, generic words, section headers, code leakage,
        and TOC/navigation artifacts.
        """
        if not text or len(text.strip()) < 2:
            return False

        text_str = text.strip()
        text_lower = text_str.lower()

        # Reject if mostly non-alphanumeric (HTML/LaTeX garbage)
        alnum_count = sum(1 for c in text_str if c.isalnum() or c.isspace())
        if alnum_count / max(len(text_str), 1) < 0.6:
            return False

        # Reject very long strings (likely sentence fragments)
        if len(text_str) > 80:
            return False

        # === FIX Bug 3: Reject section headers, TOC markers, navigation ===
        header_patterns = [
            r"^#{1,6}\s",  # Residual hash headings
            r"^[ivxlcdm]+\.",  # Roman numeral TOC (I. II. III.)
            r"^\d+[\.\)]\s*",  # Numbered TOC (1. 2. 3.)
            r"^[-*_]{3,}$",  # Horizontal rules
            r"^\[toc\]",  # Explicit TOC markers
            r"^\[#",  # Heading markdown leftover
            r"table of contents",
            r"^contents$",
            r"^index$",
            r"^summary$",
            r"^back to top$",
            r"^read more$",
            r"^acknowledgements$",
            r"^references$",
            r"^appendix$",
            r"^figure\s+\d+",  # Figure X
            r"^table\s+\d+",  # Table Y
        ]
        if any(re.search(p, text_str, re.IGNORECASE) for p in header_patterns):
            return False

        # Reject text that looks like multi-line fragments or labels
        if text_str.count("\n") > 2:
            return False
        stripped = text_str.strip("#-*_ \t\n")
        if not stripped:
            return False
        # === END FIX Bug 3 ===

        # === FIX Bug 6: Reject code leakage patterns ===
        code_patterns = [
            r"print\(",
            r"console\.",
            r"def ",
            r"class ",
            r"return ",
            r"import ",
            r"from .* import",
            r"const ",
            r"let ",
            r"var ",
            r"=>",
            r"\{.*\}",
            r"\(\)",
            r"\$\{",  # Template literals ${...}
            r"func ",  # Go functions
            r"fn ",  # Rust functions
            r"pub ",  # Rust public
            r"async ",  # Async patterns
            r"await ",  # Await patterns
            r"->",  # Type annotations / return types
            r"::",  # Rust/Cpp path syntax
            r"\[\]",  # Empty array brackets
            r"//",  # Comments
            r"#",  # Hash comments (leftover)
            r"\d+\.\d+\.\d+",  # Version strings (1.2.3)
        ]
        if any(re.search(p, text_str) for p in code_patterns):
            return False
        # === END FIX Bug 6 ===

        # Expanded stopwords for generic word leakage
        stopwords = {
            # Pronouns & determiners
            "it",
            "he",
            "she",
            "they",
            "we",
            "i",
            "you",
            "this",
            "that",
            "these",
            "those",
            "the",
            "a",
            "an",
            "which",
            "who",
            "what",
            "there",
            "here",
            "its",
            "them",
            "their",
            "his",
            "her",
            "our",
            "my",
            "your",
            "me",
            "us",
            "him",
            # Common verbs with no entity meaning
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            # Generic adjectives / adverbs
            "also",
            "very",
            "just",
            "only",
            "even",
            "still",
            "already",
            "always",
            "never",
            "sometimes",
            "often",
            "usually",
            "really",
            "quite",
            "rather",
            "more",
            "most",
            "some",
            "any",
            "all",
            "many",
            "much",
            "few",
            "several",
            "each",
            "every",
            "both",
            "either",
            "neither",
            "such",
            "same",
            "other",
            "another",
            "simple",
            "basic",
            "main",
            "new",
            "old",
            "good",
            "better",
            "best",
            "first",
            "last",
            "long",
            "short",
            "high",
            "low",
            "right",
            "left",
            "large",
            "small",
            "big",
            "next",
            # Boolean / null-like tokens
            "true",
            "false",
            "null",
            "none",
            "undefined",
            "empty",
            "yes",
            "no",
            "ok",
            "okay",
            "maybe",
            "perhaps",
            # Structure / formatting words
            "section",
            "chapter",
            "part",
            "step",
            "page",
            "line",
            "number",
            "amount",
            "total",
            "sum",
            "value",
            "point",
            "type",
            "kind",
            "sort",
            "case",
            "order",
            "level",
            "base",
            "mode",
            "format",
            "version",
            "build",
            "release",
            "list",
            "array",
            "set",
            "map",
            "hash",
            "flag",
            "mark",
            # Software / engineering noise
            "code",
            "function",
            "method",
            "class",
            "object",
            "instance",
            "element",
            "component",
            "attribute",
            "property",
            "field",
            "key",
            "string",
            "integer",
            "table",
            "column",
            "row",
            "file",
            "path",
            "name",
            "size",
            "system",
            "service",
            "module",
            "endpoint",
            "interface",
            "library",
            "framework",
            "engine",
            "tool",
            "package",
            # Prepositions / conjunctions
            "for",
            "with",
            "from",
            "into",
            "about",
            "between",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "under",
            "over",
            "out",
            "off",
            "up",
            "down",
            "in",
            "on",
            "at",
            "to",
            "of",
            "and",
            "or",
            "but",
            "not",
            "so",
            "if",
            "as",
            "by",
            # Redundant qualifiers
            "simply",
            "auto",
            "available",
            "body",
            "yet",
            "ever",
            # Time / quantity (usually not entities)
            "time",
            "day",
            "week",
            "month",
            "year",
            "hour",
            "minute",
            "second",
            "today",
            "tomorrow",
            "yesterday",
            "now",
            "then",
            "when",
            "while",
            "moment",
        }
        if text_lower in stopwords:
            return False

        # Reject plain numbers or single special chars
        if text_str.isdigit() or (len(text_str) == 1 and not text_str.isalpha()):
            return False

        # Reject sentence fragments: if it contains common verbs, articles, or
        # conjunctions mid-phrase, it's likely a noun chunk that went wrong.
        fragment_indicators = [
            " is ",
            " are ",
            " was ",
            " were ",
            " has ",
            " have ",
            " had ",
            " do ",
            " does ",
            " did ",
            " will ",
            " can ",
            " may ",
            " should ",
            " the ",
            " a ",
            " an ",
            " and ",
            " or ",
            " but ",
            " if ",
            " so ",
            " that ",
            " which ",
            " who ",
            " when ",
            " where ",
            " how ",
            " for ",
            " with ",
            " from ",
            " into ",
            " about ",
            " than ",
            " - ",
            " & ",
            " / ",
            " = ",
            " at ",
            " by ",
            " on ",
            " in ",
            " to ",
            " of ",
        ]
        if any(ind in text_lower for ind in fragment_indicators):
            return False

        # Reject purely numeric+unit combos like "50ms", "100x", "10x"
        import re as _re

        if _re.match(r"^\d+[a-zA-Z]{1,3}$", text_str):
            return False

        return True

    # ------------------------------------------------------------------
    # Noun-chunk endpoint helpers (added 2026-05-14)
    # ------------------------------------------------------------------
    # The Montague extraction strategies fall back to spaCy's noun_chunks when
    # a token isn't inside a named-entity span. Chunks come with determiners
    # attached ("the essence", "a method"), which produced garbage Entity nodes
    # like "the essence" --relates-via-of--> "open-ended phenomena". The next
    # two helpers normalize a chunk before it becomes an endpoint.

    # Determiners that prefix a noun chunk but aren't part of the named entity.
    _LEADING_DETERMINERS = frozenset(
        {
            "the",
            "a",
            "an",
            "this",
            "that",
            "these",
            "those",
            "some",
            "any",
            "such",
            "another",
            "every",
            "each",
            "all",
            "both",
            "either",
            "neither",
            "no",
        }
    )

    @classmethod
    def _strip_chunk_determiner(cls, chunk_text: str) -> str:
        """Drop leading determiners from a noun-chunk string."""
        tokens = chunk_text.strip().split()
        while tokens and tokens[0].lower() in cls._LEADING_DETERMINERS:
            tokens.pop(0)
        return " ".join(tokens)

    def _get_chunk_endpoint(self, noun_chunks: dict[int, str], idx: int) -> str | None:
        """Return a determiner-stripped noun chunk at a token index, or None.

        Returning None (rather than '') lets callers fall through naturally:
        `subject = subject or self._get_chunk_endpoint(...)`.
        """
        if idx not in noun_chunks:
            return None
        stripped = self._strip_chunk_determiner(noun_chunks[idx])
        return stripped if stripped else None

    @staticmethod
    def _tokens_in_same_entity(doc: Doc, t1: Token, t2: Token) -> bool:
        """True if both tokens fall inside the same named-entity span.

        Used to suppress compound/possessive relations between tokens that
        belong to a single named entity (e.g. "Hannaneh Hajishirzi" or
        "New York" — both PERSON/GPE spans where the spaCy compound dep
        would otherwise emit a meaningless `modifies` edge between the parts).
        """
        for ent in doc.ents:
            if (ent.start <= t1.i < ent.end) and (ent.start <= t2.i < ent.end):
                return True
        return False

    def _extract_svo_pattern(
        self, sent: Span
    ) -> tuple[Token | None, Token | None, Token | None]:
        """Extract Subject-Verb-Object pattern from a sentence."""
        subject = None
        verb = None
        obj = None

        for token in sent:
            if token.dep_ == "nsubj":
                subject = token
            elif token.pos_ == "VERB" and token.dep_ == "ROOT":
                verb = token
            elif token.dep_ in ["dobj", "pobj"]:
                obj = token

        return subject, verb, obj

    def _extract_tense(self, doc: Doc) -> str:
        """Extract tense information from the document."""
        for token in doc:
            if token.pos_ == "VERB" and token.tag_:
                if "VBD" in token.tag_ or "VBN" in token.tag_:
                    return "past"
                if "VBG" in token.tag_:
                    return "present_continuous"
                if "VBZ" in token.tag_ or "VBP" in token.tag_:
                    return "present"
        return "unknown"

    def _extract_modality(self, doc: Doc) -> list[str]:
        """Extract modal verbs and expressions."""
        modals = []
        modal_verbs = {
            "can",
            "could",
            "may",
            "might",
            "must",
            "shall",
            "should",
            "will",
            "would",
        }

        for token in doc:
            if token.lemma_.lower() in modal_verbs:
                modals.append(token.lemma_.lower())

        return modals

    async def _generate_logical_form_for_sentence(self, sent: Span) -> str:
        """Generate logical form for a specific sentence."""
        # Simplified logical form generation
        subject, verb, obj = self._extract_svo_pattern(sent)

        if subject and verb:
            if obj:
                return (
                    f"∃x∃y({subject.lemma_}(x) ∧ {obj.lemma_}(y) ∧ "
                    f"{verb.lemma_}(x,y))"
                )
            return f"∃x({subject.lemma_}(x) ∧ {verb.lemma_}(x))"

        return f"proposition({sent.text[:50]}...)"
