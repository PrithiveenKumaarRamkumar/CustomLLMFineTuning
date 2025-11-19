# pipeline/07_evaluator.py
# ================================================================================
# Module 7: Evaluate fine-tuned model
# ================================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from typing import Dict, List
import json
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate fine-tuned model using code-specific metrics"""
    
    def __init__(
        self,
        model_path: str,
        test_dataset_path: str,
        output_dir: str,
        max_samples: int = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to fine-tuned model
            test_dataset_path: Path to test dataset
            output_dir: Output directory for results
            max_samples: Maximum samples to evaluate (None for all)
            device: Device for inference
        """
        self.model_path = Path(model_path)
        self.test_dataset_path = test_dataset_path
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples
        self.device = device
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ModelEvaluator initialized")
        logger.info(f"Model: {model_path}")
        logger.info(f"Device: {device}")
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        logger.info("Loading model and tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        logger.info("✓ Model and tokenizer loaded")
    
    def load_test_dataset(self):
        """Load test dataset"""
        logger.info(f"Loading test dataset from: {self.test_dataset_path}")
        
        self.test_dataset = load_from_disk(self.test_dataset_path)
        
        if self.max_samples:
            self.test_dataset = self.test_dataset.select(range(min(self.max_samples, len(self.test_dataset))))
        
        logger.info(f"✓ Loaded {len(self.test_dataset)} test samples")
    
    def generate_predictions(self, max_new_tokens: int = 256) -> List[str]:
        """
        Generate predictions for test dataset
        
        Args:
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            List of generated predictions
        """
        logger.info("Generating predictions...")
        
        predictions = []
        
        for i, example in enumerate(tqdm(self.test_dataset, desc="Generating")):
            # Get input_ids
            input_ids = torch.tensor([example['input_ids']]).to(self.device)
            attention_mask = torch.tensor([example['attention_mask']]).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(generated_text)
        
        logger.info(f"✓ Generated {len(predictions)} predictions")
        return predictions
    
    def calculate_syntax_validity(self, predictions: List[str], language: str = 'python') -> float:
        """
        Calculate syntax validity rate
        
        Args:
            predictions: List of generated code
            language: Programming language
        
        Returns:
            Syntax validity rate (0-1)
        """
        if language != 'python':
            logger.warning(f"Syntax checking only supported for Python, got {language}")
            return 0.0
        
        valid_count = 0
        
        for pred in predictions:
            try:
                compile(pred, '<string>', 'exec')
                valid_count += 1
            except SyntaxError:
                pass
        
        validity_rate = valid_count / len(predictions) if predictions else 0.0
        logger.info(f"Syntax validity: {validity_rate:.2%}")
        
        return validity_rate
    
    def calculate_exact_match(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate exact match accuracy
        
        Args:
            predictions: List of predictions
            references: List of references
        
        Returns:
            Exact match accuracy (0-1)
        """
        if len(predictions) != len(references):
            logger.warning("Prediction and reference counts don't match")
            return 0.0
        
        matches = sum(1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip())
        accuracy = matches / len(predictions) if predictions else 0.0
        
        logger.info(f"Exact match: {accuracy:.2%}")
        return accuracy
    
    def calculate_codebleu(self, predictions: List[str], references: List[str], language: str = 'python') -> Dict:
        """
        Calculate CodeBLEU scores
        
        Args:
            predictions: List of predictions
            references: List of references
            language: Programming language
        
        Returns:
            Dictionary with CodeBLEU scores
        """
        try:
            from codebleu import calc_codebleu
            
            # CodeBLEU expects list of references for each prediction
            refs = [[ref] for ref in references]
            
            result = calc_codebleu(
                references=refs,
                predictions=predictions,
                lang=language,
                weights=(0.25, 0.25, 0.25, 0.25)
            )
            
            logger.info(f"CodeBLEU: {result['codebleu']:.4f}")
            
            return {
                'codebleu': result['codebleu'],
                'ngram_match_score': result.get('ngram_match_score', 0.0),
                'weighted_ngram_match_score': result.get('weighted_ngram_match_score', 0.0),
                'syntax_match_score': result.get('syntax_match_score', 0.0),
                'dataflow_match_score': result.get('dataflow_match_score', 0.0)
            }
        
        except Exception as e:
            logger.error(f"Error calculating CodeBLEU: {e}")
            return {
                'codebleu': 0.0,
                'ngram_match_score': 0.0,
                'weighted_ngram_match_score': 0.0,
                'syntax_match_score': 0.0,
                'dataflow_match_score': 0.0
            }
    
    def calculate_perplexity(self) -> float:
        """
        Calculate model perplexity on test set
        
        Returns:
            Perplexity score
        """
        logger.info("Calculating perplexity...")
        
        total_loss = 0.0
        total_tokens = 0
        
        for example in tqdm(self.test_dataset, desc="Calculating perplexity"):
            input_ids = torch.tensor([example['input_ids']]).to(self.device)
            attention_mask = torch.tensor([example['attention_mask']]).to(self.device)
            labels = torch.tensor([example['labels']]).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        logger.info(f"Perplexity: {perplexity:.2f}")
        return perplexity
    
    def evaluate(self, language: str = 'python') -> Dict:
        """
        Run full evaluation
        
        Args:
            language: Programming language
        
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("="*80)
        logger.info("STEP 7: MODEL EVALUATION")
        logger.info("="*80)
        
        # Load model and dataset
        self.load_model_and_tokenizer()
        self.load_test_dataset()
        
        # Generate predictions
        predictions = self.generate_predictions()
        
        # Get references (decode labels from test dataset)
        references = []
        for example in self.test_dataset:
            # Decode labels
            labels = example['labels']
            # Remove padding (-100)
            labels = [l for l in labels if l != -100]
            reference = self.tokenizer.decode(labels, skip_special_tokens=True)
            references.append(reference)
        
        # Calculate metrics
        results = {}
        
        # Syntax validity
        results['syntax_validity'] = self.calculate_syntax_validity(predictions, language)
        
        # Exact match
        results['exact_match'] = self.calculate_exact_match(predictions, references)
        
        # CodeBLEU
        codebleu_results = self.calculate_codebleu(predictions, references, language)
        results.update(codebleu_results)
        
        # Perplexity
        results['perplexity'] = self.calculate_perplexity()
        
        # Additional statistics
        results['num_test_samples'] = len(self.test_dataset)
        results['avg_prediction_length'] = np.mean([len(p.split()) for p in predictions])
        results['avg_reference_length'] = np.mean([len(r.split()) for r in references])
        
        # Save results
        self._save_results(results, predictions, references)
        
        logger.info("="*80)
        logger.info("EVALUATION RESULTS")
        logger.info("="*80)
        for key, value in results.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
        logger.info("="*80)
        logger.info("✓ STEP 7 COMPLETE")
        logger.info("="*80)
        
        return results
    
    def _save_results(self, results: Dict, predictions: List[str], references: List[str]):
        """Save evaluation results"""
        # Save metrics
        metrics_path = self.output_dir / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"✓ Metrics saved to: {metrics_path}")
        
        # Save predictions
        predictions_path = self.output_dir / "predictions.jsonl"
        with open(predictions_path, 'w') as f:
            for pred, ref in zip(predictions, references):
                entry = {
                    'prediction': pred,
                    'reference': ref
                }
                f.write(json.dumps(entry) + '\n')
        logger.info(f"✓ Predictions saved to: {predictions_path}")


def run_evaluation(config: dict) -> Dict:
    """
    Convenience function to run evaluation
    
    Args:
        config: Evaluation configuration
    
    Returns:
        Dictionary with evaluation results
    """
    evaluator = ModelEvaluator(
        model_path=config['model_path'],
        test_dataset_path=config['test_dataset_path'],
        output_dir=config['output_dir'],
        max_samples=config.get('max_samples', None),
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    return evaluator.evaluate(language=config.get('language', 'python'))


if __name__ == "__main__":
    # Example usage
    config = {
        'model_path': './output/final_model/full_model',
        'test_dataset_path': './data/tokenized/test_tokenized',
        'output_dir': './output/evaluation_results',
        'max_samples': 100,
        'language': 'python'
    }
    
    results = run_evaluation(config)
    print(json.dumps(results, indent=2))