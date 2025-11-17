"""
LLM Backend for Content Generation
Supports local inference with Llama-3.1-8B and other open-weights models.
"""

import torch
from typing import Optional, List
import re


class LlamaBackend:
    """Backend for Llama model inference."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", 
                 device: str = "auto", max_length: int = 512):
        """
        Initialize Llama backend.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cuda', 'cpu', or 'auto')
            max_length: Maximum generation length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.model = None
        self.tokenizer = None
        
        print(f"Initializing {model_name}...")
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            print(f"Model loaded successfully on {self.model.device}")
            
        except ImportError:
            print("Error: transformers library not installed.")
            print("Install with: pip install transformers torch accelerate --break-system-packages")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Note: You may need to authenticate with HuggingFace for Llama models.")
            print("Run: huggingface-cli login")
            raise
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                 max_new_tokens: int = 150, top_p: float = 0.9) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
        
        Returns:
            Generated text
        """
        if self.model is None:
            return "[Model not loaded]"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new text (after prompt)
        response = generated_text[len(prompt):].strip()
        
        # Clean up response
        response = self._clean_response(response)
        
        return response
    
    def _clean_response(self, text: str) -> str:
        """Clean up generated text."""
        # Remove any remaining special tokens
        text = re.sub(r'<\|.*?\|>', '', text)
        
        # Take only first few sentences if too long
        sentences = text.split('.')
        if len(sentences) > 4:
            text = '.'.join(sentences[:4]) + '.'
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def batch_generate(self, prompts: List[str], temperature: float = 0.7, 
                      max_new_tokens: int = 150) -> List[str]:
        """Generate responses for multiple prompts (more efficient)."""
        if self.model is None:
            return ["[Model not loaded]"] * len(prompts)
        
        # Tokenize all prompts
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode all outputs
        responses = []
        for i, output in enumerate(outputs):
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Extract new text
            response = generated_text[len(prompts[i]):].strip()
            response = self._clean_response(response)
            responses.append(response)
        
        return responses


class MockLLMBackend:
    """Mock LLM backend for testing without actual model loading."""
    
    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
        self.call_count = 0
        print(f"Using mock LLM backend (for testing)")
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                 max_new_tokens: int = 150, top_p: float = 0.9) -> str:
        """Generate mock responses."""
        self.call_count += 1
        
        # Extract key context from prompt
        if "climate change" in prompt.lower():
            responses = [
                "Climate change is one of the most pressing challenges of our time, requiring urgent collective action.",
                "We need to transition to renewable energy sources to combat climate change effectively.",
                "The scientific consensus on climate change is clear, and we must act now."
            ]
        elif "technology" in prompt.lower() or "ai" in prompt.lower():
            responses = [
                "AI technology is advancing rapidly and transforming many aspects of our society.",
                "We must carefully consider the ethical implications of artificial intelligence.",
                "Technology can be a powerful tool for solving complex problems."
            ]
        elif "healthcare" in prompt.lower():
            responses = [
                "Access to quality healthcare should be a fundamental right for everyone.",
                "Healthcare systems need reform to better serve all communities.",
                "Medical innovation is improving outcomes for many patients."
            ]
        else:
            responses = [
                "This is an important topic that deserves thoughtful discussion.",
                "There are multiple perspectives to consider on this issue.",
                "We should base our views on evidence and careful reasoning."
            ]
        
        # Add variation based on temperature
        import random
        response = random.choice(responses)
        
        if temperature > 0.7:
            response += " " + random.choice([
                "What do others think?",
                "I'm curious to hear different viewpoints.",
                "This requires more investigation."
            ])
        
        return response
    
    def batch_generate(self, prompts: List[str], temperature: float = 0.7, 
                      max_new_tokens: int = 150) -> List[str]:
        """Generate mock responses for batch."""
        return [self.generate(p, temperature) for p in prompts]


def get_llm_backend(use_mock: bool = False, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """
    Factory function to get appropriate LLM backend.
    
    Args:
        use_mock: If True, use mock backend for testing
        model_name: Model to load (if not using mock)
    
    Returns:
        LLM backend instance
    """
    if use_mock:
        return MockLLMBackend()
    else:
        return LlamaBackend(model_name=model_name)