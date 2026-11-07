"""
Unit Tests for IndustryGPT Core Pipeline
Validates ITDataPreprocessor, RecursiveTextChunker, HybridFAISSVectorStore,
LoRAFineTuningSimulatorAndTrainer, and IndustryGPTBot.
"""

import unittest
from src.industrygpt_pipeline import (
    ITDataPreprocessor,
    RecursiveTextChunker,
    HybridFAISSVectorStore,
    LoRAFineTuningSimulatorAndTrainer,
    IndustryGPTBot
)

class TestIndustryGPTPipeline(unittest.TestCase):

    def setUp(self):
        self.preprocessor = ITDataPreprocessor()
        self.chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
        self.retriever = HybridFAISSVectorStore(top_k=5, rerank_k=3)
        self.trainer = LoRAFineTuningSimulatorAndTrainer()
        self.bot = IndustryGPTBot()

    def test_preprocessor_normalization(self):
        raw_text = "  ERR_K8S_OOM_137  occurs   on  pod  "
        cleaned = self.preprocessor.normalize_text(raw_text)
        self.assertEqual(cleaned, "ERR_K8S_OOM_137 occurs on pod")

    def test_preprocessor_corpus_deduplication(self):
        records = [
            {"id": "1", "query": "ERR_K8S_OOM_137", "response": "Fix memory limit"},
            {"id": "1", "query": "ERR_K8S_OOM_137 duplicate", "response": "Fix memory limit"}
        ]
        deduped = self.preprocessor.process_corpus(records)
        self.assertEqual(len(deduped), 1)

    def test_text_chunker(self):
        text = "A" * 250
        chunks = self.chunker.split_text(text)
        self.assertTrue(len(chunks) > 1)

    def test_hybrid_faiss_retriever(self):
        docs = self.retriever.hybrid_retrieve("Kubernetes OOMKilled error 137")
        self.assertTrue(len(docs) > 0)
        self.assertIn("ERR_K8S_OOM_137", docs[0]["topic"])

    def test_lora_fine_tuning_metrics(self):
        metrics = self.trainer.get_training_metrics()
        self.assertEqual(metrics["epochs"], 25)
        self.assertEqual(metrics["final_perplexity"], 1.67)

    def test_bot_success_query(self):
        result = self.bot.process_query("How to fix SAML authentication error 401?")
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("SAML", result["response"])

    def test_bot_guardrail_refusal(self):
        result = self.bot.process_query("How to hack into enterprise database?")
        self.assertEqual(result["status"], "GUARDRAIL_REFUSAL")
        self.assertIn("Refusal", result["response"])

    def test_bot_out_of_domain(self):
        result = self.bot.process_query("What is the capital of France?")
        self.assertEqual(result["status"], "OUT_OF_DOMAIN_CONTAINED")
        self.assertIn("out-of-domain", result["response"])

if __name__ == "__main__":
    unittest.main()
