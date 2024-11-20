import numpy as np
import random
import string
from copy import deepcopy
from collections import defaultdict
from typing import List, Dict
import torch
from torch.utils.data import DataLoader
from pet.utils import InputExample
import log

logger = log.get_logger('root')
class RobustnessEvaluator:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.model = wrapper.model
        self.tokenizer = wrapper.tokenizer
        self.device = next(wrapper.model.parameters()).device

    def _apply_perturbation(self, text: str, perturbation_type: str) -> str:
        if perturbation_type == 'character_noise':
            return self._add_character_noise(text)
        elif perturbation_type == 'word_dropout':
            return self._word_dropout(text)
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")

    def _add_character_noise(self, text: str, noise_prob: float = 0.1) -> str:
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < noise_prob and chars[i].isalpha():
                chars[i] = random.choice(string.ascii_letters)
        return ''.join(chars)

    def _word_dropout(self, text: str, dropout_prob: float = 0.1) -> str:
        words = text.split()
        kept_words = [w for w in words if random.random() > dropout_prob]
        if not kept_words:
            kept_words = [random.choice(words)]
        return ' '.join(kept_words)

    def evaluate_robustness(self, examples: List[InputExample], perturbation_types: List[str] = None) -> Dict[str, float]:
        if perturbation_types is None:
            perturbation_types = ['character_noise', 'word_dropout']
            
        orig_acc = self._evaluate_examples(examples)
        perturbed_results = defaultdict(list)
        
        for example in examples:
            for pert_type in perturbation_types:
                perturbed_text = self._apply_perturbation(example.text_a, pert_type)
                pert_example = deepcopy(example)
                pert_example.text_a = perturbed_text
                pred = self._get_prediction(pert_example)
                correct = (pred == example.label)
                perturbed_results[pert_type].append(correct)
                
        metrics = {
            'original_accuracy': orig_acc
        }
        
        for pert_type, results in perturbed_results.items():
            metrics[f'{pert_type}_accuracy'] = np.mean(results)
            metrics[f'{pert_type}_relative'] = metrics[f'{pert_type}_accuracy'] / orig_acc
            
        return metrics

    def _evaluate_examples(self, examples: List[InputExample]) -> float:
        dataset = self.wrapper._generate_dataset(examples)
        dataloader = DataLoader(dataset, batch_size=8)
        
        correct = 0
        total = 0
        
        self.wrapper.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_batch = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }
                if 'token_type_ids' in batch:
                    input_batch['token_type_ids'] = batch['token_type_ids'].to(self.device)
                    
                outputs = self.wrapper.model(**input_batch)
                logits = outputs[0]
                predictions = torch.argmax(logits, dim=1)
                labels = batch['labels'].to(self.device)
                correct += (predictions == labels).sum().item()
                total += len(labels)
        
        return correct / total if total > 0 else 0.0

    def _get_prediction(self, example: InputExample) -> str:
        dataset = self.wrapper._generate_dataset([example])
        batch = next(iter(DataLoader(dataset, batch_size=1)))
        
        input_batch = {
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device)
        }
        if 'token_type_ids' in batch:
            input_batch['token_type_ids'] = batch['token_type_ids'].to(self.device)
        
        self.wrapper.model.eval()
        with torch.no_grad():
            outputs = self.wrapper.model(**input_batch)
            logits = outputs[0]
            pred = torch.argmax(logits, dim=1).item()
            return str(pred)
    def __init__(self, wrapper):
        """Initialize evaluator with wrapper instance"""
        self.wrapper = wrapper
        self.model = wrapper.model
        self.tokenizer = wrapper.tokenizer
        self.device = next(wrapper.model.parameters()).device if hasattr(wrapper.model, 'parameters') else 'cpu'

    def _get_prediction(self, example: InputExample) -> str:
        features = self.wrapper._convert_examples_to_features([example])
        dataset = self.wrapper._generate_dataset([example])
        batch = next(iter(DataLoader(dataset, batch_size=1)))
        
        input_batch = {
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
            'token_type_ids': batch['token_type_ids'].to(self.device)
        }
        
        self.wrapper.model.eval()
        with torch.no_grad():
            outputs = self.wrapper.model(**input_batch)
            logits = outputs[0]
            pred = torch.argmax(logits, dim=1).item()
            return str(pred)

    def evaluate_robustness(self, examples: List[InputExample], perturbation_types: List[str] = None) -> Dict[str, float]:
        if perturbation_types is None:
            perturbation_types = ['character_noise', 'word_dropout']
            
        orig_acc = self._evaluate_examples(examples)
        perturbed_results = defaultdict(list)
        
        for example in examples:
            for pert_type in perturbation_types:
                perturbed_text = self._apply_perturbation(example.text_a, pert_type)
                pert_example = deepcopy(example)
                pert_example.text_a = perturbed_text
                pred = self._get_prediction(pert_example)
                correct = (pred == example.label)
                perturbed_results[pert_type].append(correct)
                
        metrics = {
            'original_accuracy': orig_acc
        }
        
        for pert_type, results in perturbed_results.items():
            metrics[f'{pert_type}_accuracy'] = np.mean(results)
            metrics[f'{pert_type}_relative'] = metrics[f'{pert_type}_accuracy'] / orig_acc
            
        return metrics

    def _evaluate_examples(self, examples: List[InputExample]) -> float:
        dataset = self.wrapper._generate_dataset(examples)
        dataloader = DataLoader(dataset, batch_size=8)
        
        correct = 0
        total = 0
        
        self.wrapper.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                # Remove mlm_labels from inputs
                input_batch = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                    'token_type_ids': batch['token_type_ids'].to(self.device)
                }
                outputs = self.wrapper.model(**input_batch)
                logits = outputs[0]
                predictions = torch.argmax(logits, dim=1)
                labels = batch['labels'].to(self.device)
                correct += (predictions == labels).sum().item()
                total += len(labels)
        
        return correct / total if total > 0 else 0.0