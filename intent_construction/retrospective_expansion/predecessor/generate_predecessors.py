"""
Predecessor Inference Function Predecessor (Production)

Generates multi-turn function chains using predecessor inference:
Given the final function G_t, generates plausible predecessors G_{t-1}, G_{t-2}, ...
such that each transition is causally motivated.

Key design: chained predecessor generation with cross-turn verification.
- Generate t-1 from t (shares arguments with t, gets its own new arguments)
- Generate t-2 from t-1 (shares arguments with t-1, not t!)
- Verify fabricated arguments don't leak information across turns

Usage:
    python generate_predecessors.py \
        --input ../../intent_extraction/output/browsecomp_plus/extracted_test.json \
        --output ./output/browsecomp_plus/predecessor.json \
        --dataset_type browsecomp \
        --num_predecessors 2 \
        --model gpt-5.1 \
        --parallel 4
"""

import json
import os
import re
import random
import argparse
import threading
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from intent_construction.intent_extraction.core.llm_utils import generate_json, generate_text, load_prompt, populate_prompt


# =============================================================================
# Chain Archetypes — mapped to Function-Change Taxonomy (function_change_taxonomy.tex)
#
# Each archetype is a dataset-specific instantiation of one of four motivational types:
#   T1: Knowledge Acquisition  (L1 Disjoint — no shared arguments)
#   T2: Function Decomposition (L2 Partial — sub-task inherits some arguments)
#   T3: Sequential Function    (L2 Partial — next step shares environmental constraints)
#   T4: Function Pivot         (L2 Partial — problem context persists, method changes)
# =============================================================================

CHAIN_ARCHETYPES = {
    "identify_then_seek": {
        "taxonomy_type": "T1",  # Knowledge Acquisition (L1 Disjoint)
        "dataset": ["browsecomp"],
        "instruction": """=== CHAIN TYPE: IDENTIFY THEN SEEK ===

The predecessor identifies entity A using its OWN set of clues. The ANSWER (the identity of A) provides a critical piece of knowledge that the user needs to approach the next question about a DIFFERENT entity B, which has its OWN SEPARATE clues.

Pattern: "What/Who is A?" (clues about A) → learns identity of A → "What/Who is B?" (DIFFERENT clues about B, where knowing A helps identify B)

The bridge is the KNOWLEDGE gained (knowing what A is), NOT shared clues. The two questions must have almost entirely separate argument sets (0-1 shared arguments).

**Good examples:**
- Clues about a film → "What film is this?" → Answer: "Spirited Away" → DIFFERENT clues about an animator → "Who is this animator?" (knowing the film narrows down the animator)
- Clues about a scientific discovery → "What discovery is described here?" → Answer: "CRISPR" → DIFFERENT clues about a researcher → "Who is this researcher?" (knowing it's CRISPR helps identify the person)
- Clues about an award ceremony → "Which award ceremony is this?" → Answer: "1994 Booker Prize" → DIFFERENT clues about a novelist → "Who is this author?" (knowing the specific award helps)
- Clues about a sports event → "Which tournament is described?" → Answer: "1986 FIFA World Cup" → DIFFERENT clues about a player → "Who is this footballer?" (knowing the tournament provides context)

**Bad examples (AVOID THESE):**
- Sharing many clues between predecessor and successor (VIOLATES L1 Disjoint — each question must have its own clue set)
- "Who is person X?" → "What did person X do?" (same entity — this is drill-down, not knowledge acquisition)
- "Which university is this?" → "Which department at that university?" (drill-down into same entity, not a new identification)
- Questions where the predecessor's answer isn't actually needed for the successor

CRITICAL: The predecessor and successor must have DISJOINT argument sets (0-1 shared clues). The predecessor's clues describe entity A; the successor's clues describe entity B. The ONLY connection is the knowledge bridge — identifying A provides context that helps identify B.""",
        "share_range": (0, 1),
    },
    "survey_then_focus": {
        "taxonomy_type": "T2",  # Function Decomposition (L2 Partial)
        "dataset": ["browsecomp"],
        "instruction": """=== CHAIN TYPE: SURVEY THEN FOCUS ===

The predecessor asks a BROAD CATEGORY question whose answer is a SET or LIST of entities. The successor then NARROWS DOWN to one specific member of that set using additional arguments.

Pattern: "Which [entities] match [broad criteria]?" (answer: a list/set) → "Which of those also [specific criteria], and [detail]?" (narrows to one)
The predecessor casts a wide net; the successor filters to a single target.

**Good examples:**
- "Which African authors have had their novels become compulsory school reading?" → answer: a LIST of authors → "Which of those also lectured at a private university until his death, and in what years did he work as a probation officer?" (narrows from set to one person)
- "Which films won the Palme d'Or in the 2010s?" → answer: a LIST of films → "Which of those was directed by someone who also made a documentary about jazz?" (narrows to one film)
- "Which universities in the UK have a department of computational linguistics?" → answer: a LIST → "Which of those also hosted the 2018 ACL conference?" (narrows to one)
- "Which bands performed at Glastonbury in 2019?" → answer: a LIST → "Which of those released a debut album produced by Rick Rubin?" (narrows to one band)

**Bad examples (AVOID THESE):**
- "Who is this specific person?" → single answer, not a set (this is T4 or T3, not T2)
- "Which African author fits all these clues?" → already narrowed to one person (NOT a broad category question)
- "What film is this?" → "What is a related film?" (lateral move, not set → member)
- Any question whose answer is a SINGLE entity rather than a SET/LIST

CRITICAL: The predecessor's answer must be a SET or LIST of entities (e.g., "several authors", "a list of films", "multiple universities"). The predecessor uses only a SUBSET of the shared arguments to define the broad category. The successor then adds more arguments to narrow down to exactly ONE member of that set. This is genuine WHOLE → PART decomposition: the set is the whole, the specific entity is the part.""",
        "share_range": (2, 4),
    },
    "trace_then_follow": {
        "taxonomy_type": "T3",  # Sequential Function (L2 Partial)
        "dataset": ["browsecomp"],
        "instruction": """=== CHAIN TYPE: TRACE THEN FOLLOW ===

The predecessor asks about a DIFFERENT TYPE OF ENTITY than the final function's target. The predecessor's ANSWER is then directly CONSUMED AS INPUT for the successor question — the successor literally references or requires the predecessor's answer to be answerable.

Pattern: "What is [entity type A]?" → answer: "X" → "Who/What [relates to X]?" (successor REQUIRES knowing X)
The predecessor seeks a different entity type (book title, organization, location, event name), and its answer feeds directly into the next question.

**Good examples:**
- "What novel by an African author became compulsory school reading in 2017?" → Answer: "Petals of Blood" → "Who wrote Petals of Blood, and in what years did the author work as a probation officer?" (book title → author identification; REQUIRES the book title)
- "Which university did this African author lecture at from 2018?" → Answer: "Pan-Atlantic University" → "Which lecturer at Pan-Atlantic University also worked as a probation officer, and during what years?" (university → person; REQUIRES the university name)
- "Which award ceremony featured this 2020 film?" → Answer: "Cannes Film Festival" → "Who won the directing prize at Cannes that year?" (event → person; REQUIRES knowing the event)
- "What album by this band reached #1 in 1991?" → Answer: "Nevermind" → "Who produced the follow-up to Nevermind?" (album title → person; REQUIRES the album title)

**Bad examples (AVOID THESE):**
- "Who is this author?" → "What years did this author work as probation officer?" (same entity type — predecessor seeks person name, which is the same type the final function needs. This is T4, not T3)
- "Which country was the book compulsory in?" → "What years did the author work as probation officer?" (the country is NOT needed for the next question — this is T4, not T3)
- Questions where the predecessor and successor seek the SAME entity type (both seek person names, both seek dates, etc.)

CRITICAL: The predecessor must seek a DIFFERENT ENTITY TYPE than what the final function targets. The predecessor's answer must be MECHANICALLY REQUIRED as input for the successor — if you removed the predecessor's answer, the successor question would be unanswerable or ill-formed. Both questions share some environmental arguments, making this L2 (Partial).""",
        "share_range": (1, 3),
    },
    "pivot_inquiry": {
        "taxonomy_type": "T4",  # Function Pivot (L2 Partial)
        "dataset": ["browsecomp"],
        "instruction": """=== CHAIN TYPE: PIVOT INQUIRY ===

The predecessor and successor investigate the SAME underlying scenario, but the user CHANGES THEIR MIND about what they want to know. Both questions share the same arguments, but the predecessor asks for a DIFFERENT TYPE OF INFORMATION. Crucially, the predecessor's answer is NOT NEEDED for the successor — the user simply abandons their first question and pivots.

Pattern: "What is [attribute A] of this scenario?" → "Actually, what is [attribute B] instead?"
The user changes their mind about WHAT to look for. The scenario stays the same; the inquiry angle changes. The predecessor's answer is DISCARDED.

**Good examples:**
- Clues about an African author → "In which country was this author's book made compulsory reading?" → answer: a country → "Actually, in what years did this author work as a probation officer?" (country is NOT needed for the next question — user just pivoted)
- Clues about a historical event → "Where did this event take place?" → "Actually, who was the key figure behind this event?" (location → person; location is discarded)
- Clues about a film production → "What award did this film win?" → "Actually, who composed the soundtrack?" (accolade → creative team; the award name is not needed)
- Clues about a scientific breakthrough → "Which journal published this research?" → "Actually, who led the research team?" (venue → personnel; the journal name is not needed)
- Clues about a company → "When was this company founded?" → "Actually, who was the first CEO?" (date → person; the founding date is not needed)

**Bad examples (AVOID THESE):**
- "What novel became compulsory reading?" → "Who wrote that novel?" (the answer IS needed — this is T3 Sequential, not T4)
- "Which films won the award in the 2010s?" → "Which of those was directed by...?" (the list IS needed — this is T2 Decomposition, not T4)
- "Who is this person?" → "Who is this person?" (same question, no pivot)
- Predecessor and successor seek the SAME type of information (both ask for names, both ask for dates)

CRITICAL: The predecessor's answer must be INDEPENDENT of the successor — removing the predecessor's answer should NOT affect the successor's answerability. Both share the same scenario arguments but ask for genuinely DIFFERENT TYPES of information. This is an "actually, I'm more interested in..." pattern.""",
        "share_range": (2, 4),
    },
    # === GSM8K Archetypes ===
    # Elementary arithmetic word problems: same scenario (characters, objects, setting),
    # different mathematical questions. All predecessors must be MATH COMPUTATION
    # questions with specific numerical answers.
    "lookup_then_compute": {
        "taxonomy_type": "T1",  # Knowledge Acquisition (L1 Disjoint)
        "dataset": ["gsm8k"],
        "instruction": """=== CHAIN TYPE: LOOKUP THEN COMPUTE ===

The predecessor should be a FACT-FINDING or RATE-LOOKUP question whose answer provides a key piece of information (a rate, price, rule, percentage, conversion factor) that the user then APPLIES in the next computation.

Pattern: "What is the tax rate / unit price / exchange rate / overtime multiplier?" → "Compute the total using that rate"
The bridge is a FACT or RATE the user learned, not an intermediate computation result.

**Good examples:**
- "What is the state sales tax rate in California?" → "How much tax does John pay on his $450 purchase?" (rate lookup → tax computation)
- "What is the price per pound of chicken at the store?" → "How much does Maria spend on 8 pounds of chicken?" (unit price → total cost)
- "What is the overtime pay multiplier for weekend shifts?" → "How much does Tom earn this week?" (rule lookup → wage computation)
- "What is the current exchange rate from euros to dollars?" → "How much does the imported order cost in dollars?" (conversion rate → currency conversion)

**Bad examples (AVOID THESE):**
- "How many eggs are left after breakfast?" → "How much does she earn selling the rest?" (this is a SEQUENTIAL computation chain — T3, not T1)
- "What is 15% of 200?" → "What is 30% of 400?" (both are computations — no knowledge acquisition step)
- "How much does flour cost?" → "How much does sugar cost?" (parallel lookups, no apply step)

CRITICAL: The predecessor must ask for a FACTUAL PIECE OF INFORMATION (rate, price, percentage, rule) — NOT compute an intermediate result. The next function must APPLY that information in a computation. The predecessor's answer is informational (e.g., "9.3%", "$4.50 per pound"), not an intermediate step in a multi-step chain.""",
        "share_range": (0, 1),
    },
    "total_then_component": {
        "taxonomy_type": "T2",  # Function Decomposition (L2 Partial)
        "dataset": ["gsm8k"],
        "instruction": """=== CHAIN TYPE: TOTAL THEN COMPONENT ===

The predecessor should ask about an AGGREGATE or OVERALL quantity, and the next function zooms into a SPECIFIC COMPONENT or SUB-PART of that aggregate.

Pattern: "What is the TOTAL X?" → "How much of that total is from Y specifically?"
The user first asks about the big picture, then drills into a specific piece.

**Good examples:**
- "What is Janet's total weekly income from all sources?" → "How much does she make from selling duck eggs at the farmers' market?" (total income → one income source)
- "What is the total cost of renovating the house?" → "How much did the kitchen repairs cost?" (total cost → one category)
- "How many total hours does the team work this week?" → "How many of those hours are overtime?" (total hours → a subset)
- "What is the total distance of the road trip?" → "How far is the segment from City A to City B?" (total → one leg)

**Bad examples (AVOID THESE):**
- "How much does flour cost?" → "What is the total cost of all ingredients?" (this is REVERSED — component then total, not total then component)
- "How many eggs?" → "How many eggs?" (same question, not decomposition)
- "What is the total profit?" → "What is the total revenue?" (two totals — no zoom-in)

CRITICAL: The predecessor must compute a LARGER-SCOPE quantity that CONTAINS the next function's quantity as a sub-part. The user zooms INTO the problem. The predecessor needs ADDITIONAL arguments about other components that contribute to the total (e.g., other income sources, other cost categories).""",
        "share_range": (2, 4),
    },
    "compute_then_extend": {
        "taxonomy_type": "T3",  # Sequential Function (L2 Partial)
        "dataset": ["gsm8k"],
        "instruction": """=== CHAIN TYPE: COMPUTE THEN EXTEND ===

The predecessor should compute an INTERMEDIATE QUANTITY whose numerical result is directly needed as INPUT for the next computation. The answer to the predecessor FEEDS INTO the next step.

Pattern: "Compute intermediate X" → "Using X, compute final Y"
This is a classic multi-step arithmetic chain where step 1's output is step 2's input.

**Good examples:**
- "How many eggs does Janet have left after breakfast and baking?" → "How much does she earn selling the remaining eggs at the market?" (remaining count → revenue)
- "What is the house's new value after all the repairs?" → "How much profit does Josh make when he sells it?" (new value → profit)
- "How much flour does 3 batches of cookies require?" → "How many 5-pound bags of flour should she buy?" (total flour → bags needed)
- "How much does Tom earn before taxes?" → "How much does he take home after the 20% tax deduction?" (gross pay → net pay)

**Bad examples (AVOID THESE):**
- "What is the tax rate?" → "How much tax does he pay?" (the rate is a FACT LOOKUP, not a computation — this is T1)
- "How much does Store A charge?" → "How much does Store B charge?" (independent computations, no dependency)
- "What is the total cost?" → "What is the per-person cost?" (this could work, but only if per-person cost actually REQUIRES the total as input)

CRITICAL: There must be a genuine COMPUTATIONAL DEPENDENCY — the numerical answer to the predecessor is consumed as an input by the next function. The predecessor computes a number; the next function uses that number in further arithmetic.""",
        "share_range": (2, 4),
    },
    "reframe_problem": {
        "taxonomy_type": "T4",  # Function Pivot (L2 Partial)
        "dataset": ["gsm8k"],
        "instruction": """=== CHAIN TYPE: REFRAME PROBLEM ===

The predecessor should attempt a mathematical computation under one SCOPE or FRAMING, which the user then ABANDONS in favor of a different framing of the same scenario.

Pattern: "Compute X under framing A" → "Actually, let's compute X under framing B instead"
The user tries one approach to the problem, then switches to a different angle or scope.

**Good examples:**
- "What is the average daily profit across the whole month including weekends?" → "Actually, how much profit does she make just on weekdays?" (full scope → restricted scope)
- "How much would it cost if she bought all organic ingredients?" → "How much does it cost with regular ingredients instead?" (premium option → budget option)
- "What is the total cost if split evenly among all 8 people?" → "What's the cost if only the 5 adults split it?" (one split rule → different split rule)
- "How long would the trip take if they drove the entire way?" → "How long does it take if they take the train for the middle segment?" (one mode → mixed mode)

**Bad examples (AVOID THESE):**
- "How much does it cost with a 10% discount?" → "How much with a 20% discount?" (this is ARGUMENT CHANGE — same framing, different parameter)
- "What is the total?" → "What is the average?" (different aggregation, but no abandonment narrative)
- "How many eggs on Monday?" → "How many eggs on Tuesday?" (parallel computation, not a pivot)

CRITICAL: The pivot must involve a genuine REFRAMING — the user changes their SCOPE, ASSUMPTIONS, or APPROACH to the problem, not just a numerical parameter. The predecessor's framing is ABANDONED (too complex, too expensive, unfair, impractical), and the successor provides a fundamentally different lens on the same scenario.""",
        "share_range": (2, 4),
    },
}

# Taxonomy type labels (from function_change_taxonomy.tex)
TAXONOMY_TYPES = {
    "T1": "Knowledge Acquisition",   # L1 Disjoint
    "T2": "Function Decomposition",  # L2 Partial
    "T3": "Sequential Function",     # L2 Partial
    "T4": "Function Pivot",          # L2 Partial
}

# Dataset → valid archetype names (used for filtering)
DATASET_ARCHETYPES: Dict[str, List[str]] = {}
for _name, _arch in CHAIN_ARCHETYPES.items():
    for _ds in _arch["dataset"]:
        DATASET_ARCHETYPES.setdefault(_ds, []).append(_name)


def get_archetypes_for_dataset(dataset_type: str) -> List[str]:
    """Return valid archetype names for a dataset type."""
    return DATASET_ARCHETYPES.get(dataset_type, list(CHAIN_ARCHETYPES.keys()))


def get_taxonomy_type(archetype_name: str) -> str:
    """Return taxonomy type label (e.g., 'T1: Knowledge Acquisition') for an archetype."""
    arch = CHAIN_ARCHETYPES.get(archetype_name)
    if not arch:
        return "unknown"
    t = arch["taxonomy_type"]
    return f"{t}: {TAXONOMY_TYPES[t]}"

# Stop words for semantic overlap check
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "was", "were", "are", "of", "in", "to", "and",
    "or", "for", "that", "this", "it", "with", "on", "at", "by", "from",
    "what", "which", "who", "how", "when", "where", "name", "tell", "me",
    "can", "you", "do", "does", "did", "first", "last", "about", "their",
    "its", "be", "been", "has", "had", "have",
})


# =============================================================================
# PredecessorGenerator Class
# =============================================================================

class PredecessorGenerator:
    """
    Generates predecessor inference function chains for multi-turn conversations.
    
    Given a sample with a final function and arguments, generates predecessor
    functions that form a causally motivated chain. Each predecessor shares some
    arguments with its successor and adds fabricated arguments.
    
    Mirrors the PredecessorGenerator interface for pipeline compatibility.
    """
    
    def __init__(
        self,
        model: str = "gpt-5.1",
        prompts_dir: str = "prompts",
        dataset_type: str = "browsecomp",
        num_predecessors: int = 2,
        chain_types: List[str] = None,
        max_attempts: int = 5,
        temperature: float = 1.0,
        reasoning_effort: str = None,
        fallback_model: str = None,
        share_num: int = None,
        max_verify_attempts: int = 2,
        judge_model: str = "gpt-5.1",
        verify_independence: bool = True,
        independence_runs: int = 3,
        max_independence_retries: int = 2,
        corpus_dataset: str | None = None,
    ):
        """
        Initialize the predecessor function generator.
        
        Args:
            model: Model identifier for generation
            prompts_dir: Directory containing prompt templates
            dataset_type: Type of dataset (search, math, etc.)
            num_predecessors: Number of predecessor functions per sample
            chain_types: List of archetype names for random per-step selection (None = domain default)
            max_attempts: Max retries per single predecessor generation
            temperature: Sampling temperature
            reasoning_effort: For reasoning models ('low', 'medium', 'high')
            fallback_model: Stronger model for escalation on failure
            share_num: Exact number of shared arguments (None = archetype default)
            max_verify_attempts: Max retries for cross-turn verification
            judge_model: Model for all LLM-as-Judge tasks — similarity check,
                cross-turn relevance, and functional independence (default: gpt-5.1)
            verify_independence: Whether to run functional independence test
                (g(C ∪ C_new) == g(C)) on the final function.
            independence_runs: Number of LLM runs for functional independence
                majority voting (default: 3)
            max_independence_retries: Max retries with feedback-based argument
                regeneration when functional independence fails (default: 2)
            corpus_dataset: HuggingFace corpus dataset for BM25 retrieval in
                BrowseComp independence check (default: Tevatron/browsecomp-plus-corpus)
        """
        self.model = model
        self.prompts_dir = Path(prompts_dir)
        self.dataset_type = dataset_type
        self.num_predecessors = num_predecessors
        self.chain_types = chain_types or get_archetypes_for_dataset(dataset_type)
        self.max_attempts = max_attempts
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.fallback_model = fallback_model
        self.share_num = share_num
        self.max_verify_attempts = max_verify_attempts
        self.judge_model = judge_model
        self.verify_independence = verify_independence
        self.independence_runs = independence_runs
        self.max_independence_retries = max_independence_retries
        
        # Load domain-specific verifier for functional independence test
        self._verifier = None
        if verify_independence:
            self._verifier = self._load_verifier(dataset_type)
        
        # Initialize BM25 retriever for BrowseComp independence check
        self._bm25_retriever = None
        if verify_independence and dataset_type == "browsecomp":
            from intent_construction.retrospective_expansion.predecessor.bm25_retriever import BM25Retriever
            self._bm25_retriever = BM25Retriever(
                corpus_dataset=corpus_dataset or "Tevatron/browsecomp-plus-corpus",
            )
        
        # Random generator for per-step archetype selection
        self._rng = random.Random()
        
        # Implemented datasets (have dedicated prompts and archetypes)
        _IMPLEMENTED_DATASETS = {"browsecomp", "gsm8k"}
        
        # Load predecessor function prompt (dataset-specific; raise error for unimplemented)
        predecessor_prompt_file = self.prompts_dir / f"generate_predecessor_{dataset_type}.txt"
        if not predecessor_prompt_file.exists():
            if dataset_type not in _IMPLEMENTED_DATASETS:
                raise NotImplementedError(
                    f"Dataset type '{dataset_type}' is not implemented. "
                    f"Create a prompt at {predecessor_prompt_file} with dataset-specific "
                    f"instructions. See generate_predecessor_default.txt for the structural template. "
                    f"Implemented datasets: {sorted(_IMPLEMENTED_DATASETS)}"
                )
            # Shouldn't happen for implemented datasets, but guard anyway
            raise FileNotFoundError(
                f"Predecessor prompt for implemented dataset '{dataset_type}' not found at {predecessor_prompt_file}"
            )
        self.predecessor_prompt_template = load_prompt(predecessor_prompt_file)
        
        # Load similarity check prompt (dataset-specific with fallback)
        similarity_prompt_file = self.prompts_dir / f"similarity_check_{dataset_type}.txt"
        if not similarity_prompt_file.exists():
            similarity_prompt_file = self.prompts_dir / "similarity_check_default.txt"
        self.similarity_prompt_template = load_prompt(similarity_prompt_file)
        
        # Load cross-turn verification prompt
        verify_prompt_file = self.prompts_dir / "cross_turn_relevance_check.txt"
        if verify_prompt_file.exists():
            self.verify_prompt_template = load_prompt(verify_prompt_file)
        else:
            print(f"Warning: Verification prompt not found at {verify_prompt_file}")
            self.verify_prompt_template = None
    
    # =========================================================================
    # LLM-as-Judge Similarity Check
    # =========================================================================
    
    _SIMILARITY_PROMPT = """You are judging whether two questions are semantically too similar — i.e., they are essentially asking the same thing, just reworded.

Question A: {function_a}
Question B: {function_b}

Two questions are "too similar" if:
- They ask for the same piece of information about the same entity/topic
- One is a rephrasing or subset of the other
- Answering one would directly answer the other

Two questions are "different enough" if:
- They ask about different entities, topics, or attributes
- They share domain vocabulary but seek fundamentally different information
- They are related (e.g., both about music) but ask distinct questions

Reply with EXACTLY one word: SIMILAR or DIFFERENT"""

    def _llm_similarity_check(self, function_a: str, function_b: str) -> bool:
        """
        Use LLM judge to check if two functions are semantically too similar.
        Returns True if too similar (should reject), False if different enough.
        Uses dataset-specific similarity prompt loaded from prompts directory.
        """
        prompt = self.similarity_prompt_template.format(function_a=function_a, function_b=function_b)
        try:
            response = generate_text(
                [{"role": "user", "content": prompt}],
                model=self.judge_model,
                max_tokens=10,
                temperature=None,
            )
            verdict = response.strip().upper()
            return "SIMILAR" in verdict
        except Exception as e:
            # On failure, fall back to word-overlap heuristic
            pred_words = set(re.findall(r'\w+', function_a.lower())) - _STOP_WORDS
            other_words = set(re.findall(r'\w+', function_b.lower())) - _STOP_WORDS
            if not pred_words or not other_words:
                return False
            overlap = len(pred_words & other_words) / min(len(pred_words), len(other_words))
            return overlap > 0.6
    
    _ANSWER_STOP_WORDS = frozenset({
        "the", "a", "an", "is", "was", "were", "are", "of", "in", "to", "and",
        "or", "for", "that", "this", "it", "with", "on", "at", "by", "from",
        "what", "which", "who", "how", "when", "where", "name", "tell", "me",
        "can", "you", "do", "does", "did", "first", "last", "about", "their",
        "its", "be", "been", "has", "had", "have", "not", "but", "as", "if",
        "into", "through", "during", "before", "after", "between", "under",
        "over", "then", "than", "so", "because", "while", "also", "both",
        "each", "other", "some", "such", "only", "same", "very", "just",
        "more", "most", "new", "old", "all", "any", "no", "nor", "not",
        "using", "based", "study", "research", "paper", "article", "book",
        "title", "work", "series", "part", "number", "year", "time",
    })

    @staticmethod
    def _extract_answer_keywords(answer: str, min_word_len: int = 4) -> set:
        """Extract meaningful keywords from the answer, filtering generic words."""
        words = set(re.findall(r'\w+', answer.lower()))
        return {w for w in words if len(w) >= min_word_len and w not in PredecessorGenerator._ANSWER_STOP_WORDS}

    def _check_answer_leakage(self, text: str, answer_keywords: set, threshold: int = 2) -> list:
        """
        Check if text leaks answer keywords.
        Returns list of leaked words if leakage detected, empty list otherwise.
        """
        if not answer_keywords:
            return []
        text_lower = text.lower()
        leaked = [w for w in answer_keywords if w in text_lower]
        if len(leaked) >= threshold:
            return leaked
        return []

    # =========================================================================
    # Functional Independence Verification
    # =========================================================================

    @staticmethod
    def _load_verifier(dataset_type: str):
        """
        Load domain-specific verifier for functional independence testing.
        Returns None if verifier is unavailable.
        """
        try:
            if dataset_type in ("math", "olympiad", "gsm8k"):
                from intent_construction.intent_extraction.core.math_verifier import MathVerifier
                return MathVerifier(num_runs=1)
            elif dataset_type in ("browsecomp",):
                # BrowseComp uses LLM-as-judge for answer comparison
                return "browsecomp_domain"
            else:
                return None
        except ImportError as e:
            print(f"Warning: Could not load verifier for '{dataset_type}': {e}")
            return None

    def _verify_functional_independence(
        self,
        function: str,
        arguments: List[Dict[str, Any]],
        all_new_arguments: List[Dict[str, Any]],
        ground_truth: str,
        sample: Dict[str, Any],
    ) -> tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Functional independence test: verify g(C ∪ C_new) == g(C).

        Runs the model on the final function with and without fabricated arguments
        from all predecessors and compares answers. This catches cases where
        the fabricated arguments change the answer even though the LLM-judge
        thought they were independent.

        Args:
            function: The final (original) function text
            arguments: The final function's own arguments
            all_new_arguments: All fabricated arguments from all predecessors
            ground_truth: Expected answer for the final function
            sample: Full sample dict (for IF domain metadata)

        Returns:
            (passed, reason, feedback_info)
            feedback_info contains details for retry if verification failed
        """
        if not all_new_arguments:
            return True, "No fabricated arguments to test", None

        # --- Math domain ---
        if self._verifier and self._verifier != "browsecomp_domain":
            return self._verify_functional_independence_math(
                function, arguments, all_new_arguments, ground_truth,
            )

        # --- BrowseComp domain ---
        if self._verifier == "browsecomp_domain":
            return self._verify_functional_independence_browsecomp(
                function, arguments, all_new_arguments, ground_truth,
            )

        return True, "No verifier available, skipping functional test", None

    def _verify_functional_independence_math(
        self,
        function: str,
        arguments: List[Dict[str, Any]],
        all_new_arguments: List[Dict[str, Any]],
        ground_truth: str,
    ) -> tuple[bool, str, Optional[Dict[str, Any]]]:
        """Functional independence for math domains using math-verify."""
        verifier = self._verifier

        # Build prompt WITHOUT fabricated arguments
        base_texts = [function]
        for cond in sorted(arguments, key=lambda c: c.get("argument_id", 0)):
            base_texts.append(cond.get("argument", ""))
        base_prompt = " ".join(base_texts)

        # Build prompt WITH fabricated arguments
        extended_texts = list(base_texts)
        for nc in all_new_arguments:
            nc_text = nc.get("argument", "") if isinstance(nc, dict) else str(nc)
            if nc_text:
                extended_texts.append(nc_text)
        extended_prompt = " ".join(extended_texts)

        pass_count = 0
        wrong_answers = []

        for run in range(self.independence_runs):
            try:
                # Answer A: without fabricated arguments
                response_a = generate_text(
                    [{"role": "system", "content": verifier.system_prompt},
                     {"role": "user", "content": base_prompt}],
                    model=self.judge_model,
                    temperature=self.temperature,
                    max_tokens=2000,
                )
                answer_a = verifier.extract_answer(response_a)

                # Answer B: with fabricated arguments
                response_b = generate_text(
                    [{"role": "system", "content": verifier.system_prompt},
                     {"role": "user", "content": extended_prompt}],
                    model=self.judge_model,
                    temperature=self.temperature,
                    max_tokens=2000,
                )
                answer_b = verifier.extract_answer(response_b)

                # Pass if: (A == B) OR (B == ground truth)
                answers_match = (
                    verifier.evaluate_answer(answer_a, answer_b)
                    if answer_a and answer_b
                    else False
                )
                b_correct = verifier.evaluate_answer(answer_b, ground_truth)

                if answers_match or b_correct:
                    pass_count += 1
                else:
                    wrong_answers.append({
                        "answer_without_new_cond": answer_a,
                        "answer_with_new_cond": answer_b,
                        "ground_truth": ground_truth,
                    })
            except Exception as e:
                print(f"    ⚠️  Functional independence run {run+1} error: {e}")
                continue

        required = self.independence_runs // 2 + 1
        if pass_count >= required:
            return True, f"Functional independence passed ({pass_count}/{self.independence_runs})", None

        feedback_info = {
            "pass_count": pass_count,
            "num_runs": self.independence_runs,
            "ground_truth": ground_truth,
            "wrong_answers": wrong_answers,
            "new_arguments": [
                nc.get("argument", "") if isinstance(nc, dict) else str(nc)
                for nc in all_new_arguments
            ],
        }
        return (
            False,
            f"Functional independence failed ({pass_count}/{self.independence_runs})",
            feedback_info,
        )

    def _verify_functional_independence_browsecomp(
        self,
        function: str,
        arguments: List[Dict[str, Any]],
        all_new_arguments: List[Dict[str, Any]],
        ground_truth: str,
    ) -> tuple[bool, str, Optional[Dict[str, Any]]]:
        """Functional independence for BrowseComp using BM25-RAG + LLM comparison.

        Uses BM25 retrieval to provide search context, then asks the generator
        model (not judge model) to answer with and without fabricated arguments.
        The same retrieved documents are used for both to isolate the distraction
        effect of fabricated arguments (not retrieval shift).

        Passes if:
          - A and B are equivalent (fabricated arguments didn't change the answer), OR
          - B is correct (matches ground truth)

        Each independence run:
          1. BM25 retrieves top-5 docs for function + original arguments (once)
          2. Model answers with: function + arguments + docs → Answer A
          3. Model answers with: function + arguments + fabricated + SAME docs → Answer B
          4. Compare A ≡ B or B ≡ ground_truth
        """
        if self._bm25_retriever is None:
            return True, "No BM25 retriever available, skipping", None

        sorted_arguments = sorted(arguments, key=lambda c: c.get("argument_id", 0))
        original_cond_texts = [
            cond.get("argument", "") for cond in sorted_arguments
        ]
        fabricated_cond_texts = []
        for nc in all_new_arguments:
            nc_text = nc.get("argument", "") if isinstance(nc, dict) else str(nc)
            if nc_text:
                fabricated_cond_texts.append(nc_text)

        # Build search query from original function + arguments only
        base_query = function + " " + " ".join(original_cond_texts)

        def _build_rag_prompt(question: str, clues: list[str], retrieved_docs_json: str) -> str:
            """Build a RAG prompt with retrieved documents as context."""
            docs = json.loads(retrieved_docs_json)
            context_parts = []
            for i, doc in enumerate(docs, 1):
                context_parts.append(f"[Document {i}] {doc.get('snippet', '')}")
            context = "\n\n".join(context_parts)

            return (
                "Answer the following search/trivia question using the provided "
                "reference documents and clues. Provide a short, specific factual answer.\n\n"
                f"Question: {question}\n\n"
                "Clues:\n" + "\n".join(f"- {c}" for c in clues) + "\n\n"
                f"Reference Documents:\n{context}\n\n"
                "Answer:"
            )

        judge_prompt_template = (
            "You are judging whether two answers to a trivia/search question are equivalent.\n\n"
            "Question: {function}\n"
            "Answer A: {answer_a}\n"
            "Answer B: {answer_b}\n\n"
            "Are these two answers referring to the same thing? "
            "Ignore minor differences in formatting, phrasing, or level of detail. "
            "Focus on whether they identify the same entity, fact, or value.\n\n"
            'Output JSON: {{"equivalent": true/false, "reason": "brief explanation"}}'
        )

        pass_count = 0
        wrong_answers = []

        # Retrieve docs ONCE using only the original function + arguments.
        # Both answers use the same docs so we isolate the distraction effect
        # of fabricated arguments (not the retrieval shift).
        base_docs = self._bm25_retriever.search(base_query)

        for run in range(self.independence_runs):
            try:
                # Prompt A: original arguments + retrieved docs
                base_prompt = _build_rag_prompt(function, original_cond_texts, base_docs)

                # Prompt B: original + fabricated arguments + SAME retrieved docs
                extended_prompt = _build_rag_prompt(
                    function, original_cond_texts + fabricated_cond_texts, base_docs,
                )

                # Answer A: generator model with original arguments + retrieved docs
                response_a = generate_text(
                    [{"role": "user", "content": base_prompt}],
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=500,
                )
                answer_a = response_a.strip()

                # Answer B: generator model with all arguments + SAME retrieved docs
                response_b = generate_text(
                    [{"role": "user", "content": extended_prompt}],
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=500,
                )
                answer_b = response_b.strip()

                # Quick string check
                if answer_a.lower().strip() == answer_b.lower().strip():
                    pass_count += 1
                    continue

                # Check if B matches ground truth
                if answer_b.lower().strip() == ground_truth.lower().strip():
                    pass_count += 1
                    continue

                # LLM judge: check if A == B
                ab_judge = generate_json(
                    [{"role": "user", "content": judge_prompt_template.format(
                        function=function, answer_a=answer_a, answer_b=answer_b,
                    )}],
                    model=self.judge_model,
                    temperature=1,
                    step="functional-independence-browsecomp",
                )
                if ab_judge and ab_judge.get("equivalent", False):
                    pass_count += 1
                    continue

                # LLM judge: check if B == ground truth
                bg_judge = generate_json(
                    [{"role": "user", "content": judge_prompt_template.format(
                        function=function, answer_a=answer_b, answer_b=ground_truth,
                    )}],
                    model=self.judge_model,
                    temperature=1,
                    step="functional-independence-browsecomp",
                )
                if bg_judge and bg_judge.get("equivalent", False):
                    pass_count += 1
                    continue

                wrong_answers.append({
                    "answer_without_new_cond": answer_a,
                    "answer_with_new_cond": answer_b,
                    "ground_truth": ground_truth,
                })
            except Exception as e:
                print(f"    ⚠️  Functional independence run {run+1} error: {e}")
                continue

        required = self.independence_runs // 2 + 1
        if pass_count >= required:
            return True, f"Functional independence passed ({pass_count}/{self.independence_runs})", None

        feedback_info = {
            "pass_count": pass_count,
            "num_runs": self.independence_runs,
            "ground_truth": ground_truth,
            "wrong_answers": wrong_answers,
            "new_arguments": [
                nc.get("argument", "") if isinstance(nc, dict) else str(nc)
                for nc in all_new_arguments
            ],
        }
        return (
            False,
            f"Functional independence failed ({pass_count}/{self.independence_runs})",
            feedback_info,
        )

    def _regenerate_new_arguments_with_feedback(
        self,
        predecessor: Dict[str, Any],
        original_function: str,
        original_arguments: List[Dict[str, Any]],
        feedback_info: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Regenerate a predecessor's fabricated arguments with explicit feedback
        about why the previous arguments failed the functional independence test.

        Args:
            predecessor: The predecessor dict containing full_arguments
            original_function: The final function whose answer must not change
            original_arguments: The final function's own arguments
            feedback_info: Details from the failed independence test

        Returns:
            New list of argument dicts (with argument_id, argument, is_shared),
            or None if regeneration failed
        """
        # Extract the old fabricated arguments
        old_new_conds = [
            c for c in predecessor.get("full_arguments", [])
            if not c.get("is_shared", False)
        ]
        shared_conds = [
            c for c in predecessor.get("full_arguments", [])
            if c.get("is_shared", False)
        ]

        if not old_new_conds:
            return None

        # Build feedback message
        wrong_answers = feedback_info.get("wrong_answers", [])
        ground_truth = feedback_info.get("ground_truth", "")

        wrong_examples = []
        for wa in wrong_answers[:2]:
            if isinstance(wa, dict):
                if "answer_with_new_cond" in wa:
                    wrong_examples.append(
                        f"  - Without your arguments → {wa.get('answer_without_new_cond', '?')}\n"
                        f"    With your arguments → {wa.get('answer_with_new_cond', '?')}\n"
                        f"    Correct answer: {wa.get('ground_truth', '?')}"
                    )
                elif "extended_strict" in wa:
                    wrong_examples.append(
                        f"  - Base constraints pass: {wa.get('base_strict', '?')}\n"
                        f"    Extended constraints pass: {wa.get('extended_strict', '?')}"
                    )

        old_conds_str = "\n".join(f"  - {c['argument']}" for c in old_new_conds)
        shared_str = "\n".join(f"  - {c['argument']}" for c in shared_conds)
        orig_conds_str = "\n".join(
            f"  - [{c['argument_id']}] {c['argument']}" for c in original_arguments
        )

        prompt = f"""You previously generated fabricated arguments for a predecessor question in a multi-turn conversation.
These arguments FAILED the independence test — they changed the answer to the ORIGINAL question.

=== ORIGINAL GOAL (must NOT be affected) ===
{original_function}

=== ORIGINAL CONDITIONS ===
{orig_conds_str}

=== CORRECT ANSWER ===
{ground_truth}

=== PREDECESSOR GOAL ===
{predecessor.get('predecessor_function', '')}

=== SHARED CONDITIONS (keep these) ===
{shared_str}

=== YOUR PREVIOUS FABRICATED CONDITIONS (FAILED) ===
{old_conds_str}

=== WHAT WENT WRONG ===
When a model tried to solve the ORIGINAL GOAL with your fabricated arguments present,
it got the WRONG answer:
{chr(10).join(wrong_examples) if wrong_examples else '(answer changed)'}

Your arguments either:
1. CONTRADICT the original arguments (making the problem unsolvable or changing the answer)
2. Provide MISLEADING numerical values that interfere with the calculation
3. Add CONFUSING context that distracts the solver from the correct approach

=== TASK ===
Generate REPLACEMENT fabricated arguments for the predecessor question that:
- Are relevant to the predecessor function: "{predecessor.get('predecessor_function', '')}"
- Do NOT affect, contradict, or provide alternative values for the original function's answer
- Are truly INDEPENDENT from the original function's calculation/constraints
- Introduce genuinely NEW information (different entities, different domains)

Output valid JSON:
{{
    "new_arguments": ["argument 1 text", "argument 2 text", ...],
    "why_independent": "Brief explanation of why these won't affect the original answer"
}}"""

        try:
            result = generate_json(
                [{"role": "user", "content": prompt}],
                model=self.model,
                step="regenerate-arguments-feedback",
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
            )

            new_cond_texts = result.get("new_arguments", [])
            if not new_cond_texts:
                return None

            # Rebuild argument objects reusing the old IDs
            new_cond_objects = []
            for i, text in enumerate(new_cond_texts):
                if isinstance(text, str) and text.strip():
                    if i < len(old_new_conds):
                        cid = old_new_conds[i]["argument_id"]
                    else:
                        max_existing = max(
                            (c["argument_id"] for c in predecessor["full_arguments"]),
                            default=0,
                        )
                        cid = max_existing + 1 + (i - len(old_new_conds))
                    new_cond_objects.append({
                        "argument_id": cid,
                        "argument": text.strip(),
                        "is_shared": False,
                    })

            return shared_conds + new_cond_objects

        except Exception as e:
            print(f"    ⚠️  Argument regeneration with feedback failed: {e}")
            return None

    # =========================================================================
    # Public Interface
    # =========================================================================

    def generate_predecessors(
        self,
        sample: Dict[str, Any],
        num_predecessors: int = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate predecessor inference chain for a single sample.
        
        Args:
            sample: Original sample with function, arguments, answer
            num_predecessors: Override instance default if set
            
        Returns:
            Modified sample with predecessors and predecessor_functions added,
            or None if generation failed
        """
        num_preds = num_predecessors or self.num_predecessors
        
        function = sample.get("function", "")
        arguments = sample.get("arguments", [])
        answer = sample.get("answer", "")
        task_id = sample.get("task_id", "unknown")
        
        if not arguments:
            print(f"  ✗ No arguments in sample {task_id}")
            return None
        
        # Try generation + verification (with retries)
        answer_keywords = self._extract_answer_keywords(answer)
        for verify_attempt in range(self.max_verify_attempts):
            predecessors = self._generate_chain(
                function=function,
                arguments=arguments,
                num_predecessors=num_preds,
                answer_keywords=answer_keywords,
            )
            
            if not predecessors:
                print(f"  ✗ Failed to generate chain for {task_id}")
                return None
            
            # Reverse to chronological order (t-2, t-1, t)
            predecessors.reverse()
            
            # Cross-turn verification
            verification_result = self._verify_chain(
                predecessors=predecessors,
                original_function=function,
                original_arguments=arguments,
            )
            
            if verification_result is None or verification_result["passed"]:
                break
            
            if verify_attempt < self.max_verify_attempts - 1:
                print(f"  ↻ Verification failed, regenerating (attempt {verify_attempt + 2}/{self.max_verify_attempts})")
            else:
                print(f"  ✗ Max verification attempts reached for {task_id}")
        
        # ── Functional independence test: g(C ∪ C_new) == g(C) ──
        # After the chain passes cross-turn LLM-judge, verify that fabricated
        # arguments don't actually change the final function's answer.
        independence_passed = None
        independence_reason = ""
        if self._verifier and predecessors:
            # Collect ALL fabricated arguments from ALL predecessors
            all_new_conds = []
            for pred in predecessors:
                for c in pred.get("full_arguments", []):
                    if not c.get("is_shared", False):
                        all_new_conds.append(c)

            if all_new_conds:
                for indep_attempt in range(self.max_independence_retries + 1):
                    passed, reason, feedback_info = self._verify_functional_independence(
                        function=function,
                        arguments=arguments,
                        all_new_arguments=all_new_conds,
                        ground_truth=answer,
                        sample=sample,
                    )
                    independence_passed = passed
                    independence_reason = reason

                    if passed:
                        print(f"  ✓ Functional independence: {reason}")
                        break

                    # Not passed — try feedback-based regeneration
                    if indep_attempt < self.max_independence_retries and feedback_info:
                        print(f"  ⚠ {reason}, regenerating arguments (attempt {indep_attempt + 2}/{self.max_independence_retries + 1})")
                        regenerated_any = False

                        for pred in predecessors:
                            new_full = self._regenerate_new_arguments_with_feedback(
                                predecessor=pred,
                                original_function=function,
                                original_arguments=arguments,
                                feedback_info=feedback_info,
                            )
                            if new_full is not None:
                                pred["full_arguments"] = new_full
                                pred["new_arguments"] = [
                                    c["argument"] for c in new_full
                                    if not c.get("is_shared", False)
                                ]
                                pred["new_argument_ids"] = [
                                    c["argument_id"] for c in new_full
                                    if not c.get("is_shared", False)
                                ]
                                regenerated_any = True

                        # Re-collect ALL fabricated arguments after regeneration
                        all_new_conds = []
                        for pred in predecessors:
                            for c in pred.get("full_arguments", []):
                                if not c.get("is_shared", False):
                                    all_new_conds.append(c)

                        if not regenerated_any:
                            print(f"  ✗ Argument regeneration failed for {task_id}")
                            break
                    else:
                        print(f"  ✗ Functional independence failed for {task_id}: {reason}")
                        break
        
        # Build result
        new_sample = deepcopy(sample)
        new_sample["predecessors"] = predecessors
        new_sample["chain_type"] = [
            p.get("transition_type", "unknown") for p in predecessors
        ]
        new_sample["verification_passed"] = (
            verification_result["passed"] if verification_result else None
        )
        new_sample["verification_details"] = (
            verification_result["details"] if verification_result else None
        )
        new_sample["independence_passed"] = independence_passed
        new_sample["independence_reason"] = independence_reason
        
        # Also produce predecessor_functions format for user_simulation.py compatibility
        new_sample["predecessor_functions"] = self._to_predecessor_functions_format(
            predecessors, arguments
        )
        
        return new_sample
    
    # =========================================================================
    # Private: Chain Generation
    # =========================================================================
    
    def _generate_chain(
        self,
        function: str,
        arguments: List[Dict[str, Any]],
        num_predecessors: int,
        answer_keywords: set = None,
    ) -> List[Dict[str, Any]]:
        """Generate the full predecessor chain (in reverse chronological order).
        
        Each step randomly selects an archetype from self.chain_types,
        producing diverse transition types within a single chain.
        """
        predecessors = []
        current_function = function
        current_arguments = arguments
        all_functions_in_chain = [function]
        chain_entity_types = []
        future_functions_stack = []
        
        for i in range(num_predecessors):
            # Randomly select archetype for this step
            chain_type = self._rng.choice(self.chain_types)
            
            result = self._generate_single_predecessor(
                next_function=current_function,
                next_arguments=current_arguments,
                existing_predecessors=predecessors,
                all_functions_in_chain=all_functions_in_chain,
                future_functions=future_functions_stack,
                chain_entity_types=chain_entity_types,
                chain_type=chain_type,
                answer_keywords=answer_keywords,
            )
            
            if result:
                predecessors.append(result)
                all_functions_in_chain.append(result["predecessor_function"])
                if result.get("entity_sought"):
                    chain_entity_types.append(result["entity_sought"])
                # Update for next iteration
                future_functions_stack = [current_function] + future_functions_stack
                current_function = result["predecessor_function"]
                current_arguments = result["full_arguments"]
            else:
                break
        
        return predecessors
    
    def _generate_single_predecessor(
        self,
        next_function: str,
        next_arguments: List[Dict[str, Any]],
        existing_predecessors: List[Dict],
        all_functions_in_chain: List[str],
        future_functions: List[str],
        chain_entity_types: List[str],
        chain_type: str,
        answer_keywords: set = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate a single predecessor function using predecessor inference."""
        archetype = CHAIN_ARCHETYPES[chain_type]
        chain_type_instruction = archetype["instruction"]
        
        # Build arguments string
        cond_str = "\n".join(
            f"- [argument_id={c['argument_id']}] {c['argument']}"
            for c in next_arguments
        )
        
        # Build avoid functions string
        functions_to_avoid = list(set(
            [next_function] + all_functions_in_chain +
            [p.get("predecessor_function", "") for p in existing_predecessors]
        ))
        functions_to_avoid = [g for g in functions_to_avoid if g]
        avoid_str = ""
        if functions_to_avoid:
            avoid_str = "AVOID THESE GOALS (generate something GENUINELY DIFFERENT — not a rephrasing of any of these):\n"
            avoid_str += "\n".join(f"- {g}" for g in functions_to_avoid)
        
        # Build share_num instruction
        share_num_instruction = ""
        if self.share_num is not None:
            n_arguments = len(next_arguments)
            effective_share = min(self.share_num, n_arguments)
            share_num_instruction = (
                f"- IMPORTANT: You must select EXACTLY {effective_share} arguments as relevant_argument_ids. "
                f"Pick the {effective_share} most relevant arguments for your predecessor question."
            )
        
        # Build future chain string
        future_chain_str = self._build_future_chain_str(
            future_functions, chain_entity_types
        )
        
        prompt = populate_prompt(
            self.predecessor_prompt_template,
            {
                "NEXT_GOAL": next_function,
                "NEXT_CONDITIONS": cond_str,
                "AVOID_GOALS": avoid_str,
                "SHARE_NUM_INSTRUCTION": share_num_instruction,
                "CHAIN_TYPE_INSTRUCTION": chain_type_instruction,
                "FUTURE_CHAIN": future_chain_str,
            }
        )
        
        model = self.model
        for attempt in range(self.max_attempts):
            # Escalate to fallback model after half the attempts
            if (self.fallback_model and attempt >= self.max_attempts // 2
                    and model != self.fallback_model):
                print(f"    Escalating to fallback model: {self.fallback_model}")
                model = self.fallback_model
            
            try:
                result = generate_json(
                    [{"role": "user", "content": prompt}],
                    model=model,
                    step="predecessor-function-generation",
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                )
                
                validated = self._validate_predecessor(
                    result=result,
                    next_arguments=next_arguments,
                    all_functions_in_chain=all_functions_in_chain,
                    existing_predecessors=existing_predecessors,
                    chain_entity_types=chain_entity_types,
                    attempt=attempt,
                    answer_keywords=answer_keywords,
                )
                if validated:
                    return validated
                    
            except Exception as e:
                print(f"    Error generating predecessor function: {e} (attempt {attempt + 1})")
                continue
        
        return None
    
    def _validate_predecessor(
        self,
        result: Dict[str, Any],
        next_arguments: List[Dict[str, Any]],
        all_functions_in_chain: List[str],
        existing_predecessors: List[Dict],
        chain_entity_types: List[str],
        attempt: int,
        answer_keywords: set = None,
    ) -> Optional[Dict[str, Any]]:
        """Validate and assemble a predecessor result. Returns None if invalid."""
        pred_function = result.get("predecessor_function", "").strip()
        relevant_ids = result.get("relevant_argument_ids", [])
        new_arguments = result.get("new_arguments", [])
        transition_reason = result.get("transition_reason", "")
        transition_type = result.get("transition_type", "")
        
        if not pred_function:
            return None
        
        # Check exact duplicate
        pred_lower = pred_function.strip().lower()
        all_check_functions = list(set(
            all_functions_in_chain +
            [p.get("predecessor_function", "") for p in existing_predecessors]
        ))
        if any(pred_lower == g.strip().lower() for g in all_check_functions if g):
            print(f"    Duplicate function, retrying... (attempt {attempt + 1})")
            return None
        
        # LLM-as-Judge semantic similarity check
        for g in all_check_functions:
            if not g:
                continue
            if self._llm_similarity_check(pred_function, g):
                print(f"    Too similar (LLM judge), retrying... (attempt {attempt + 1})")
                return None
        
        # Dangling reference check
        dangling_patterns = [
            r'\bthis\s+(author|person|individual|series|show|film|movie|player|team|book|article)\b',
            r'\bthe\s+same\s+(author|person|individual|series|show|film|movie|player|team|book|article)\b',
        ]
        if any(re.search(p, pred_lower) for p in dangling_patterns):
            print(f"    Dangling reference, retrying... (attempt {attempt + 1})")
            return None
        
        # Entity type dedup (same entity type = likely too similar)
        entity_sought = result.get("entity_sought", "").strip().lower()
        if entity_sought and chain_entity_types:
            entity_norm = entity_sought.replace("name", "").replace("title", "").strip()
            for existing_et in chain_entity_types:
                existing_norm = existing_et.lower().replace("name", "").replace("title", "").strip()
                if entity_norm and existing_norm and (
                    entity_norm == existing_norm or
                    entity_norm in existing_norm or
                    existing_norm in entity_norm
                ):
                    print(f"    Same entity type '{entity_sought}', retrying... (attempt {attempt + 1})")
                    return None
        
        # Question length check
        if len(pred_function.split()) > 35:
            print(f"    Question too long ({len(pred_function.split())} words), retrying... (attempt {attempt + 1})")
            return None
        
        # Answer leakage check — predecessor function and fabricated arguments
        # must not contain distinctive keywords from the original answer
        if answer_keywords:
            # Check predecessor function
            leaked = self._check_answer_leakage(pred_function, answer_keywords)
            if leaked:
                print(f"    Answer leakage in function ({leaked}), retrying... (attempt {attempt + 1})")
                return None
            # Check fabricated arguments
            for nc in new_arguments:
                nc_text = nc if isinstance(nc, str) else nc.get("argument", "")
                leaked = self._check_answer_leakage(nc_text, answer_keywords)
                if leaked:
                    print(f"    Answer leakage in argument ({leaked}), retrying... (attempt {attempt + 1})")
                    return None
        
        # Validate and assemble arguments
        all_ids = [c["argument_id"] for c in next_arguments]
        relevant_ids = [cid for cid in relevant_ids if cid in all_ids]
        if not relevant_ids:
            relevant_ids = all_ids[:max(1, len(all_ids) // 2)]
        
        # Enforce share_num
        if self.share_num is not None:
            effective_share = min(self.share_num, len(all_ids))
            if len(relevant_ids) > effective_share:
                relevant_ids = relevant_ids[:effective_share]
            elif len(relevant_ids) < effective_share:
                remaining = [cid for cid in all_ids if cid not in relevant_ids]
                relevant_ids.extend(remaining[:effective_share - len(relevant_ids)])
        
        # Build shared arguments
        shared_arguments = []
        for cond in next_arguments:
            if cond["argument_id"] in relevant_ids:
                shared_arguments.append({
                    "argument_id": cond["argument_id"],
                    "argument": cond["argument"],
                    "is_shared": True,
                })
        
        # Build new arguments with fresh IDs
        max_existing_id = max(all_ids) if all_ids else 0
        new_argument_objects = []
        for i, nc_text in enumerate(new_arguments):
            if isinstance(nc_text, str) and nc_text.strip():
                new_argument_objects.append({
                    "argument_id": max_existing_id + 100 + i + 1,
                    "argument": nc_text.strip(),
                    "is_shared": False,
                })
        
        full_arguments = shared_arguments + new_argument_objects
        
        return {
            "predecessor_function": pred_function,
            "entity_sought": result.get("entity_sought", ""),
            "full_arguments": full_arguments,
            "shared_arguments": [c["argument"] for c in shared_arguments],
            "shared_argument_ids": relevant_ids,
            "new_arguments": [c["argument"] for c in new_argument_objects],
            "new_argument_ids": [c["argument_id"] for c in new_argument_objects],
            "transition_reason": transition_reason,
            "transition_type": transition_type,
            "taxonomy_type": CHAIN_ARCHETYPES.get(transition_type, {}).get("taxonomy_type", ""),
            "causal_link": result.get("causal_link", ""),
            "reasoning": result.get("reasoning", ""),
        }
    
    # =========================================================================
    # Private: Verification
    # =========================================================================
    
    def _verify_chain(
        self,
        predecessors: List[Dict[str, Any]],
        original_function: str,
        original_arguments: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Verify cross-turn independence of fabricated arguments."""
        if not self.verify_prompt_template or not predecessors:
            return None
        
        # Collect new arguments per turn
        turn_new_arguments = {}
        for i, turn in enumerate(predecessors):
            new_conds = [
                c for c in turn.get("full_arguments", [])
                if not c.get("is_shared", False)
            ]
            turn_new_arguments[i] = {
                "function": turn["predecessor_function"],
                "new_arguments": new_conds,
            }
        
        results = {"passed": True, "details": [], "problematic_pairs": []}
        
        # Check each turn against others' new arguments
        for target_idx in range(len(predecessors)):
            target = predecessors[target_idx]
            target_function = target["predecessor_function"]
            target_conds = target.get("full_arguments", [])
            
            other_new = []
            for other_idx in range(len(predecessors)):
                if other_idx == target_idx:
                    continue
                for c in turn_new_arguments[other_idx]["new_arguments"]:
                    other_new.append({
                        "source_turn": other_idx + 1,
                        "argument": c["argument"],
                        "argument_id": c["argument_id"],
                    })
            
            if not other_new:
                results["details"].append({
                    "target_turn": target_idx + 1,
                    "passed": True,
                    "reason": "No other-turn new arguments to check",
                })
                continue
            
            detail = self._check_single_turn_independence(
                target_function=target_function,
                target_arguments=target_conds,
                other_new_arguments=other_new,
                turn_label=target_idx + 1,
            )
            results["details"].append(detail)
            if not detail["passed"]:
                results["passed"] = False
                results["problematic_pairs"].append({
                    "target_turn": target_idx + 1,
                    "problematic": detail.get("problematic_arguments", []),
                })
        
        # Check all predecessors' new arguments against final function
        all_other_new = []
        for idx in range(len(predecessors)):
            for c in turn_new_arguments[idx]["new_arguments"]:
                all_other_new.append({
                    "source_turn": idx + 1,
                    "argument": c["argument"],
                    "argument_id": c["argument_id"],
                })
        
        if all_other_new:
            detail = self._check_single_turn_independence(
                target_function=original_function,
                target_arguments=original_arguments,
                other_new_arguments=all_other_new,
                turn_label="final",
            )
            results["details"].append(detail)
            if not detail["passed"]:
                results["passed"] = False
                results["problematic_pairs"].append({
                    "target_turn": "final",
                    "problematic": detail.get("problematic_arguments", []),
                })
        
        return results
    
    def _check_single_turn_independence(
        self,
        target_function: str,
        target_arguments: List[Dict[str, Any]],
        other_new_arguments: List[Dict[str, Any]],
        turn_label,
    ) -> Dict[str, Any]:
        """Check if other turns' new arguments leak info for this turn."""
        target_cond_str = "\n".join(
            f"  - [{c['argument_id']}] {c['argument']}"
            for c in target_arguments
        )
        other_cond_str = "\n".join(
            f"  - [Turn {o['source_turn']}, ID {o['argument_id']}] {o['argument']}"
            for o in other_new_arguments
        )
        
        prompt = populate_prompt(self.verify_prompt_template, {
            "TARGET_GOAL": target_function,
            "TARGET_CONDITIONS": target_cond_str,
            "OTHER_NEW_CONDITIONS": other_cond_str,
        })
        
        try:
            result = generate_json(
                [{"role": "user", "content": prompt}],
                model=self.judge_model,
                step="cross-turn-relevance-check",
                temperature=self.temperature,
            )
            
            return {
                "target_turn": turn_label,
                "target_function": target_function[:80] if isinstance(target_function, str) else str(target_function),
                "passed": result.get("overall_pass", True),
                "problematic_arguments": result.get("problematic_arguments", []),
                "judgments": result.get("judgments", []),
            }
        except Exception as e:
            print(f"    Warning: relevance check failed for turn {turn_label}: {e}")
            return {
                "target_turn": turn_label,
                "passed": True,
                "reason": f"Check failed (assuming pass): {e}",
            }
    
    # =========================================================================
    # Private: Helpers
    # =========================================================================
    
    def _build_future_chain_str(
        self,
        future_functions: List[str],
        chain_entity_types: List[str],
    ) -> str:
        """Build the future chain context string for the prompt."""
        if not future_functions:
            return ""
        
        future_chain_str = (
            "\n=== FULL FUTURE CHAIN (CRITICAL — read carefully) ===\n"
            "After your predecessor question, the user will proceed through these functions IN ORDER:\n"
        )
        for i, fg in enumerate(future_functions):
            step_label = f"Step {i+1}" if i < len(future_functions) - 1 else "FINAL"
            future_chain_str += f"  {step_label}: \"{fg}\"\n"
        future_chain_str += (
            "\nCRITICAL RULES:\n"
            "- Your predecessor must ask about something FUNDAMENTALLY DIFFERENT from ALL steps above.\n"
            "- If the future chain already contains an identification question (Who/What is X?), "
            "your predecessor must NOT be another identification question. Instead, ask about "
            "CONTEXT, BACKGROUND, or a RELATED ENTITY that sets the stage.\n"
            "- If the future chain already asks about a specific entity or topic, your predecessor "
            "should explore a DIFFERENT aspect or a CONNECTED entity — not the same thing with different clues.\n"
            "- Think: What EARLIER, DIFFERENT question would a user naturally ask before this sequence?"
        )
        
        if chain_entity_types:
            future_chain_str += (
                f"\n\nEntity types already in the chain: {', '.join(chain_entity_types)}. "
                "Your predecessor's entity_sought MUST be a DIFFERENT type (e.g., if chain has 'person name', "
                "ask about an organization, event, place, work title, etc.)."
            )
        
        return future_chain_str
    
    def _to_predecessor_functions_format(
        self,
        predecessors: List[Dict[str, Any]],
        original_arguments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert predecessor predecessors to predecessor_functions format for user_simulation.py.
        
        Each predecessor becomes a predecessor_function entry with:
        - predecessor_function: the predecessor's question
        - counterfactual_arguments: list of arguments with is_shared flag
        - is_predecessor: True (marker for the simulator to use follow-up prefixes)
        
        is_shared is True if the argument_id appears in ANY other turn
        (another predecessor or the final original arguments). This lets
        the simulator skip re-revealing arguments already shown in earlier turns.
        """
        predecessor_functions = []
        original_ids = {c["argument_id"] for c in original_arguments}
        
        # Build a set of argument_ids for each predecessor (by index)
        pred_cond_ids = []
        for pred in predecessors:
            ids = {c["argument_id"] for c in pred.get("full_arguments", [])}
            pred_cond_ids.append(ids)
        
        for pred_idx, pred in enumerate(predecessors):
            # Argument is shared if it appears in any OTHER predecessor or in original
            other_ids = set(original_ids)
            for other_idx in range(len(predecessors)):
                if other_idx != pred_idx:
                    other_ids |= pred_cond_ids[other_idx]
            
            pg_arguments = []
            for c in pred.get("full_arguments", []):
                pg_arguments.append({
                    "argument_id": c["argument_id"],
                    "argument": c["argument"],
                    "is_shared": c["argument_id"] in other_ids,
                })
            
            predecessor_functions.append({
                "predecessor_function": pred["predecessor_function"],
                "counterfactual_arguments": pg_arguments,
                "is_predecessor": True,
                "transition_type": pred.get("transition_type", ""),
                "taxonomy_type": pred.get("taxonomy_type", ""),
                "transition_reason": pred.get("transition_reason", ""),
                "entity_sought": pred.get("entity_sought", ""),
            })
        
        return predecessor_functions


# =============================================================================
# CLI Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Predecessor inference function predecessor (production)"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input JSON file with function and arguments"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--dataset_type", type=str, default="browsecomp",
        choices=["browsecomp", "gsm8k", "default"],
        help="Type of dataset for prompt selection (default: browsecomp)"
    )
    parser.add_argument(
        "--num_predecessors", type=int, default=2,
        help="Number of predecessor functions per sample (default: 2)"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-5.1",
        help="Model to use (default: gpt-5.1)"
    )
    parser.add_argument(
        "--fallback_model", type=str, default=None,
        help="Stronger model for escalation on failure"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--judge_model", type=str, default="gpt-5.1",
        help="Model for all LLM-as-Judge tasks: similarity, cross-turn, independence (default: gpt-5.1)"
    )
    parser.add_argument(
        "--share_num", type=int, default=None,
        help="Exact number of shared arguments (None = archetype default)"
    )
    parser.add_argument(
        "--chain_types", nargs="+", default=None,
        choices=list(CHAIN_ARCHETYPES.keys()),
        help="Chain archetypes to cycle through (default: domain-appropriate)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Number of samples to process (default: all)"
    )
    parser.add_argument(
        "--max_attempts", type=int, default=5,
        help="Max retries per single predecessor generation (default: 5)"
    )
    parser.add_argument(
        "--max_verify_attempts", type=int, default=2,
        help="Max retries for cross-turn verification (default: 2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=50,
        help="Save checkpoint every N samples (default: 50)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if exists"
    )
    parser.add_argument(
        "--reasoning_effort", type=str, default=None,
        choices=["low", "medium", "high"],
        help="Thinking effort for reasoning models"
    )
    parser.add_argument(
        "--no_independence_test", action="store_true",
        help="Skip the functional independence test (g(C∪C_new)==g(C))"
    )
    parser.add_argument(
        "--independence_runs", type=int, default=3,
        help="Number of LLM runs for functional independence majority voting (default: 3)"
    )
    parser.add_argument(
        "--max_independence_retries", type=int, default=2,
        help="Max feedback-based argument regeneration retries (default: 2)"
    )
    parser.add_argument(
        "--corpus_dataset", type=str, default=None,
        help="HuggingFace corpus dataset for BM25 retrieval in BrowseComp independence check "
             "(default: Tevatron/browsecomp-plus-corpus)"
    )
    args = parser.parse_args()
    
    # Auto-append share_num to output filename if set
    if args.share_num is not None:
        base, ext = os.path.splitext(args.output)
        args.output = f"{base}_sn{args.share_num}{ext}"
    
    random.seed(args.seed)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = args.output.replace(".json", "_checkpoint.json")
    
    # Load input data
    print(f"Loading input data from: {args.input}")
    with open(args.input, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    
    if args.num_samples is not None:
        data = data[:args.num_samples]
        print(f"Processing first {len(data)} samples")
    
    print(f"\nConfiguration:")
    print(f"  Dataset type: {args.dataset_type}")
    print(f"  Predecessors per sample: {args.num_predecessors}")
    print(f"  Chain types: {args.chain_types or 'all (domain default)'} (random per step)")
    print(f"  Share num: {args.share_num}")
    print(f"  Model: {args.model}")
    print(f"  Fallback model: {args.fallback_model}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Reasoning effort: {args.reasoning_effort}")
    print(f"  Parallel workers: {args.parallel}")
    print(f"  Functional independence test: {'disabled' if args.no_independence_test else 'enabled'}")
    if not args.no_independence_test:
        print(f"  Independence runs: {args.independence_runs}")
        print(f"  Max independence retries: {args.max_independence_retries}")
    
    # Resume from checkpoint
    results = []
    failed = 0
    start_idx = 0
    processed_ids = set()
    
    if args.resume and os.path.exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)
        results = checkpoint_data.get("results", [])
        failed = checkpoint_data.get("failed", 0)
        start_idx = checkpoint_data.get("next_idx", 0)
        processed_ids = set(checkpoint_data.get("processed_ids", []))
        print(f"  Loaded {len(results)} completed results, starting from index {start_idx}")
    
    # Initialize generator
    script_dir = Path(__file__).parent
    generator = PredecessorGenerator(
        model=args.model,
        prompts_dir=str(script_dir / "prompts"),
        dataset_type=args.dataset_type,
        num_predecessors=args.num_predecessors,
        chain_types=args.chain_types,
        max_attempts=args.max_attempts,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        fallback_model=args.fallback_model,
        share_num=args.share_num,
        judge_model=args.judge_model,
        max_verify_attempts=args.max_verify_attempts,
        verify_independence=not args.no_independence_test,
        independence_runs=args.independence_runs,
        max_independence_retries=args.max_independence_retries,
        corpus_dataset=args.corpus_dataset,
    )
    
    # Thread-safe lock for shared state
    results_lock = threading.Lock()
    actual_idx = start_idx
    
    def process_sample(idx_sample):
        """Process a single sample."""
        idx, sample = idx_sample
        sample_id = sample.get("task_id", f"sample-{idx}")
        
        if sample_id in processed_ids:
            return None, sample_id, "skipped"
        
        result = generator.generate_predecessors(sample)
        
        if result is not None:
            return result, sample_id, "success"
        else:
            return None, sample_id, "failed"
    
    def save_checkpoint(next_idx):
        checkpoint_data = {
            "results": results,
            "failed": failed,
            "next_idx": next_idx,
            "processed_ids": list(processed_ids),
            "total_samples": len(data),
            "num_predecessors": args.num_predecessors,
            "dataset_type": args.dataset_type,
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)
    
    print(f"\nGenerating predecessor inference chains...")
    
    try:
        if args.parallel > 1:
            print(f"  Using {args.parallel} parallel workers...")
            samples_to_process = [(start_idx + i, s) for i, s in enumerate(data[start_idx:])]
            
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = {executor.submit(process_sample, item): item for item in samples_to_process}
                
                with tqdm(total=len(samples_to_process), desc="Predecessor inference") as pbar:
                    for future in as_completed(futures):
                        result, sample_id, status = future.result()
                        
                        with results_lock:
                            if status == "success":
                                results.append(result)
                                processed_ids.add(sample_id)
                            elif status == "failed":
                                failed += 1
                        
                        pbar.update(1)
        else:
            for idx, sample in enumerate(tqdm(
                data[start_idx:],
                desc="Predecessor inference",
                initial=start_idx,
                total=len(data),
            )):
                actual_idx = start_idx + idx
                sample_id = sample.get("task_id", f"sample-{actual_idx}")
                
                if sample_id in processed_ids:
                    continue
                
                result = generator.generate_predecessors(sample)
                
                if result is not None:
                    results.append(result)
                    processed_ids.add(sample_id)
                else:
                    failed += 1
                
                # Checkpoint
                if (actual_idx + 1) % args.checkpoint_interval == 0:
                    save_checkpoint(actual_idx + 1)
                    print(f"\n  💾 Checkpoint saved at index {actual_idx + 1} ({len(results)} results)")
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted! Saving checkpoint...")
        save_checkpoint(actual_idx + 1)
        print(f"  💾 Checkpoint saved to: {checkpoint_path}")
        print(f"  To resume, run with --resume flag")
        return
    
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        save_checkpoint(actual_idx + 1)
        print(f"  💾 Emergency checkpoint saved to: {checkpoint_path}")
        raise
    
    # Save final results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully processed: {len(results)}/{len(data)} samples")
    print(f"  Failed: {failed}")
    print(f"  Output: {args.output}")
    
    # Summary stats
    verified = sum(1 for r in results if r.get("verification_passed") is True)
    unverified = sum(1 for r in results if r.get("verification_passed") is None)
    v_failed = sum(1 for r in results if r.get("verification_passed") is False)
    print(f"  Cross-turn verification: {verified} passed, {v_failed} failed, {unverified} skipped")
    
    # Independence stats
    indep_passed = sum(1 for r in results if r.get("independence_passed") is True)
    indep_failed = sum(1 for r in results if r.get("independence_passed") is False)
    indep_skipped = sum(1 for r in results if r.get("independence_passed") is None)
    print(f"  Functional independence: {indep_passed} passed, {indep_failed} failed, {indep_skipped} skipped")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
