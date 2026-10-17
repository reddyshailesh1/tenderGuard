"""Natural Language Inference (NLI) engine for contradiction detection."""
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tenderguard.config import DEFAULT_CONFIG, ModelConfig
from tenderguard.models.embeddings import CandidatePair

logger = logging.getLogger(__name__)


@dataclass
class NLIPrediction:
    """Represents the multi-class NLI prediction for a candidate clause pair."""
    clause_a_id: str
    clause_b_id: str
    predicted_label: str             # "contradiction", "entailment", or "neutral"
    contradiction_prob: float
    entailment_prob: float
    neutral_prob: float
    is_contradiction: bool
    model_identifier: str
    details: Optional[Dict[str, float]] = None


class NLIEngine:
    """Pretrained NLI inference engine evaluating logical compatibility between clauses."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or DEFAULT_CONFIG.models
        self.model_name = self.config.nli_model_name
        self.device = "cuda" if (self.config.device == "cuda" or (self.config.device == "auto" and torch.cuda.is_available())) else "cpu"
        self._tokenizer = None
        self._model = None
        self._label_map: Dict[int, str] = {}

    def _load_model(self):
        """Lazy load tokenizer and sequence classification model."""
        if self._model is None or self._tokenizer is None:
            logger.info("Loading Pretrained NLI Model: %s on %s", self.model_name, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

            # Normalize model id2label mapping to standard lowercase keys
            # MNLI standard: 0 -> entailment, 1 -> neutral, 2 -> contradiction (or model config)
            raw_id2label = getattr(self._model.config, "id2label", {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"})
            self._label_map = {idx: label.lower() for idx, label in raw_id2label.items()}
            logger.info("NLI Label mapping: %s", self._label_map)

    def predict_pair(
        self,
        premise_text: str,
        hypothesis_text: str,
        clause_a_id: str = "A",
        clause_b_id: str = "B",
        bidirectional: bool = True
    ) -> NLIPrediction:
        """Evaluate logical entailment, neutral, and contradiction between two clauses."""
        self._load_model()

        # Forward prediction: (premise, hypothesis)
        forward_probs = self._run_single_inference(premise_text, hypothesis_text)

        if bidirectional:
            # Reverse prediction: (hypothesis, premise) to handle non-symmetric formulations
            reverse_probs = self._run_single_inference(hypothesis_text, premise_text)
            # Max pooling for contradiction risk to avoid missing bidirectional conflicts
            contra_prob = max(forward_probs.get("contradiction", 0.0), reverse_probs.get("contradiction", 0.0))
            entail_prob = max(forward_probs.get("entailment", 0.0), reverse_probs.get("entailment", 0.0))
            neutral_prob = (forward_probs.get("neutral", 0.0) + reverse_probs.get("neutral", 0.0)) / 2.0
            
            # Normalize probabilities so they sum to 1.0
            total = contra_prob + entail_prob + neutral_prob
            if total > 0:
                contra_prob /= total
                entail_prob /= total
                neutral_prob /= total
        else:
            contra_prob = forward_probs.get("contradiction", 0.0)
            entail_prob = forward_probs.get("entailment", 0.0)
            neutral_prob = forward_probs.get("neutral", 0.0)

        probs_by_label = {
            "contradiction": contra_prob,
            "entailment": entail_prob,
            "neutral": neutral_prob
        }
        top_label = max(probs_by_label.items(), key=lambda x: x[1])[0]
        is_contradiction = contra_prob >= self.config.contradiction_threshold

        return NLIPrediction(
            clause_a_id=clause_a_id,
            clause_b_id=clause_b_id,
            predicted_label=top_label,
            contradiction_prob=contra_prob,
            entailment_prob=entail_prob,
            neutral_prob=neutral_prob,
            is_contradiction=is_contradiction,
            model_identifier=self.model_name,
            details=probs_by_label
        )

    def _run_single_inference(self, text_a: str, text_b: str) -> Dict[str, float]:
        """Execute forward pass for a single pair and return softmax probabilities."""
        inputs = self._tokenizer(
            text_a,
            text_b,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        result = {}
        for idx, p in enumerate(probs):
            label_name = self._label_map.get(idx, f"label_{idx}")
            result[label_name] = float(p)
        return result

    def predict_candidate_pairs(self, candidate_pairs: List[CandidatePair], batch_size: int = 16) -> List[NLIPrediction]:
        """Run NLI prediction across a list of candidate pairs."""
        logger.info("Evaluating NLI inference for %d candidate pairs...", len(candidate_pairs))
        predictions: List[NLIPrediction] = []
        for pair in candidate_pairs:
            pred = self.predict_pair(
                premise_text=pair.clause_a_text,
                hypothesis_text=pair.clause_b_text,
                clause_a_id=pair.clause_a_id,
                clause_b_id=pair.clause_b_id,
                bidirectional=True
            )
            predictions.append(pred)
        return predictions
