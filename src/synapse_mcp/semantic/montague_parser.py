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
                    "id": self._generate_entity_id(ent.text, ent.label_),
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
                if not subject and token.i in noun_chunks:
                    subject = noun_chunks[token.i]

                if not subject:
                    continue

                # Get predicate
                predicate = head.lemma_ if head.pos_ == "VERB" else head.text

                # Find object
                obj = None
                for child in head.children:
                    if child.dep_ in ["dobj", "attr", "pobj"]:
                        obj = self._find_entity_for_token(doc, child)
                        if not obj and child.i in noun_chunks:
                            obj = noun_chunks[child.i]
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
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        cop_subject = self._find_entity_for_token(doc, child)
                        if not cop_subject and child.i in noun_chunks:
                            cop_subject = noun_chunks[child.i]
                        break

                # Find complement/attribute
                cop_obj: str | None = None
                for child in token.children:
                    if child.dep_ in ["attr", "acomp", "dobj"]:
                        cop_obj = self._find_entity_for_token(doc, child)
                        if not cop_obj and child.i in noun_chunks:
                            cop_obj = noun_chunks[child.i]
                        break

                if cop_subject and cop_obj:
                    relations.append(
                        {
                            "subject": cop_subject,
                            "predicate": f"is-{token.lemma_}",
                            "object": cop_obj,
                            "confidence": 0.75,
                            "source_span": f"{token.i}",
                        }
                    )

        # Strategy 3: Prepositional relationships
        for token in doc:
            if token.dep_ == "prep" and token.head.pos_ in ["NOUN", "PROPN", "VERB"]:
                # Get the noun being modified
                subject = self._find_entity_for_token(doc, token.head)
                if not subject and token.head.i in noun_chunks:
                    subject = noun_chunks[token.head.i]

                # Get the object of the preposition
                obj = None
                for child in token.children:
                    if child.dep_ == "pobj":
                        obj = self._find_entity_for_token(doc, child)
                        if not obj and child.i in noun_chunks:
                            obj = noun_chunks[child.i]
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

        # Strategy 4: Compound and possessive patterns
        for token in doc:
            if token.dep_ in ["compound", "poss"] and token.head.pos_ in [
                "NOUN",
                "PROPN",
            ]:
                subject = token.text
                obj = token.head.text

                if len(subject) > 2 and len(obj) > 2:  # Filter very short words
                    relation_type = (
                        "modifies" if token.dep_ == "compound" else "possesses"
                    )
                    relations.append(
                        {
                            "subject": subject,
                            "predicate": relation_type,
                            "object": obj,
                            "confidence": 0.65,
                            "source_span": f"{token.i}-{token.head.i}",
                        }
                    )

        # Quality filter and deduplication
        filtered: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for rel in relations:
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
                proposition = {
                    "id": f"prop_{sent.start}_{sent.end}",
                    "content": sent.text.strip(),
                    "entities": [
                        self._generate_entity_id(ent.text, ent.label_)
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
        """Normalize spaCy entity labels to our schema."""
        mapping = {
            "PERSON": "Person",
            "ORG": "Organization",
            "GPE": "Location",
            "LOC": "Location",
            "PRODUCT": "Product",
            "EVENT": "Event",
            "WORK_OF_ART": "CreativeWork",
            "LAW": "Law",
            "LANGUAGE": "Language",
            "DATE": "TemporalEntity",
            "TIME": "TemporalEntity",
            "MONEY": "MonetaryValue",
            "QUANTITY": "Quantity",
            "CONCEPT": "Concept",
            "METHOD": "Method",
        }

        return mapping.get(spacy_label, "Entity")

    @staticmethod
    def _refine_entity_type(text: str, current_type: str) -> str:
        """Refine entity type using heuristic rules for technical concepts."""
        text_lower = text.lower()

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
                    raise TypeError(f"Expected str for entity_text, got {type(entity_text)}")
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
