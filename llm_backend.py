"""
Multi-Architecture LLM Backend for Content Generation
Supports multiple LLM architectures coexisting in the same simulation.

Architecture types:
- Llama family (Llama-3.1, Llama-3.2, etc.)
- GPT-style models
- Gemini models (via API)
- Smaller models (e.g., Phi, TinyLlama)
- Mock backend for testing
"""

from typing import Optional, List, Dict
import re
from abc import ABC, abstractmethod

# Make torch import optional (only needed for real LLM backends)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class BaseLLMBackend(ABC):
    """Abstract base class for all LLM backends."""

    def __init__(self, model_name: str, architecture_type: str):
        """
        Initialize LLM backend.

        Args:
            model_name: Model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            architecture_type: Type of architecture (e.g., "llama", "gpt", "gemini")
        """
        self.model_name = model_name
        self.architecture_type = architecture_type
        self.generation_count = 0
        self.total_tokens_generated = 0

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7,
                 max_new_tokens: int = 150, top_p: float = 0.9) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def batch_generate(self, prompts: List[str], temperature: float = 0.7,
                      max_new_tokens: int = 150) -> List[str]:
        """Generate responses for multiple prompts."""
        pass

    def get_stats(self) -> Dict:
        """Return statistics about this backend's usage."""
        return {
            'model_name': self.model_name,
            'architecture_type': self.architecture_type,
            'generation_count': self.generation_count,
            'total_tokens_generated': self.total_tokens_generated
        }

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


class LlamaBackend(BaseLLMBackend):
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
        super().__init__(model_name, architecture_type="llama")
        self.max_length = max_length
        self.device = device
        self.model = None
        self.tokenizer = None

        print(f"[Llama Backend] Initializing {model_name}...")
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Llama backend. Install with: pip install torch transformers")

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

        except ImportError as e:
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

        # Update statistics
        self.generation_count += 1
        self.total_tokens_generated += len(self.tokenizer.encode(response))

        return response

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

            # Update statistics
            self.generation_count += 1
            self.total_tokens_generated += len(self.tokenizer.encode(response))

        return responses


class GPTStyleBackend(BaseLLMBackend):
    """Backend for GPT-style models (GPT-2, GPT-Neo, GPT-J, etc.)."""

    def __init__(self, model_name: str = "gpt2", device: str = "auto"):
        """
        Initialize GPT-style backend.

        Args:
            model_name: HuggingFace model identifier (e.g., "gpt2", "EleutherAI/gpt-neo-2.7B")
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        super().__init__(model_name, architecture_type="gpt")
        self.device = device
        self.model = None
        self.tokenizer = None

        print(f"[GPT Backend] Initializing {model_name}...")
        self._load_model()

    def _load_model(self):
        """Load GPT model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device,
                low_cpu_mem_usage=True
            )

            self.model.eval()
            print(f"[GPT Backend] Model loaded successfully")

        except Exception as e:
            print(f"[GPT Backend] Error loading model: {e}")
            raise

    def generate(self, prompt: str, temperature: float = 0.7,
                 max_new_tokens: int = 150, top_p: float = 0.9) -> str:
        """Generate text from prompt."""
        if self.model is None:
            return "[Model not loaded]"

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        response = self._clean_response(response)

        # Update statistics
        self.generation_count += 1
        self.total_tokens_generated += len(self.tokenizer.encode(response))

        return response

    def batch_generate(self, prompts: List[str], temperature: float = 0.7,
                      max_new_tokens: int = 150) -> List[str]:
        """Generate responses for multiple prompts."""
        return [self.generate(p, temperature, max_new_tokens) for p in prompts]


class MistralBackend(BaseLLMBackend):
    """Backend for Mistral models."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", device: str = "auto"):
        """
        Initialize Mistral backend.

        Args:
            model_name: HuggingFace model identifier for Mistral models
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        super().__init__(model_name, architecture_type="mistral")
        self.device = device
        self.model = None
        self.tokenizer = None

        print(f"[Mistral Backend] Initializing {model_name}...")
        self._load_model()

    def _load_model(self):
        """Load Mistral model and tokenizer."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Mistral backend. Install with: pip install torch transformers")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device,
                low_cpu_mem_usage=True
            )

            self.model.eval()
            print(f"[Mistral Backend] Model loaded successfully")

        except Exception as e:
            print(f"[Mistral Backend] Error loading model: {e}")
            raise

    def generate(self, prompt: str, temperature: float = 0.7,
                 max_new_tokens: int = 150, top_p: float = 0.9) -> str:
        """Generate text from prompt."""
        if self.model is None:
            return "[Model not loaded]"

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        response = self._clean_response(response)

        # Update statistics
        self.generation_count += 1
        self.total_tokens_generated += len(self.tokenizer.encode(response))

        return response

    def batch_generate(self, prompts: List[str], temperature: float = 0.7,
                      max_new_tokens: int = 150) -> List[str]:
        """Generate responses for multiple prompts."""
        return [self.generate(p, temperature, max_new_tokens) for p in prompts]


class SmallModelBackend(BaseLLMBackend):
    """Backend for smaller, faster models (Phi, TinyLlama, etc.)."""

    def __init__(self, model_name: str = "microsoft/phi-2", device: str = "auto"):
        """
        Initialize small model backend.

        Args:
            model_name: HuggingFace model identifier (e.g., "microsoft/phi-2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            device: Device to run on
        """
        super().__init__(model_name, architecture_type="small")
        self.device = device
        self.model = None
        self.tokenizer = None

        print(f"[Small Model Backend] Initializing {model_name}...")
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device,
                low_cpu_mem_usage=True,
                trust_remote_code=True  # Some small models require this
            )

            self.model.eval()
            print(f"[Small Model Backend] Model loaded successfully")

        except Exception as e:
            print(f"[Small Model Backend] Error loading model: {e}")
            raise

    def generate(self, prompt: str, temperature: float = 0.7,
                 max_new_tokens: int = 150, top_p: float = 0.9) -> str:
        """Generate text from prompt."""
        if self.model is None:
            return "[Model not loaded]"

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        response = self._clean_response(response)

        # Update statistics
        self.generation_count += 1
        self.total_tokens_generated += len(self.tokenizer.encode(response))

        return response

    def batch_generate(self, prompts: List[str], temperature: float = 0.7,
                      max_new_tokens: int = 150) -> List[str]:
        """Generate responses for multiple prompts."""
        return [self.generate(p, temperature, max_new_tokens) for p in prompts]


class MockLLMBackend(BaseLLMBackend):
    """Mock LLM backend for testing without actual model loading."""

    def __init__(self, model_name: str = "mock", architecture_type: str = "mock"):
        super().__init__(model_name, architecture_type)
        print(f"[Mock Backend] Using mock LLM backend (for testing)")
    
    def generate(self, prompt: str, temperature: float = 0.7,
                 max_new_tokens: int = 150, top_p: float = 0.9) -> str:
        """Generate mock responses."""
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

        # Update statistics
        self.generation_count += 1
        self.total_tokens_generated += len(response.split())

        return response

    def batch_generate(self, prompts: List[str], temperature: float = 0.7,
                      max_new_tokens: int = 150) -> List[str]:
        """Generate mock responses for batch."""
        return [self.generate(p, temperature) for p in prompts]


class MultiLLMManager:
    """
    Manages multiple LLM backends for a simulation.
    Allows different agents to use different model architectures.
    """

    def __init__(self):
        self.backends: Dict[str, BaseLLMBackend] = {}
        self.agent_backend_mapping: Dict[str, str] = {}

    def add_backend(self, backend_id: str, backend: BaseLLMBackend):
        """Add a new LLM backend to the manager."""
        self.backends[backend_id] = backend
        print(f"[MultiLLM Manager] Added backend '{backend_id}' ({backend.architecture_type})")

    def assign_backend_to_agent(self, agent_id: str, backend_id: str):
        """Assign a specific backend to an agent."""
        if backend_id not in self.backends:
            raise ValueError(f"Backend '{backend_id}' not found. Available: {list(self.backends.keys())}")
        self.agent_backend_mapping[agent_id] = backend_id

    def get_backend_for_agent(self, agent_id: str) -> BaseLLMBackend:
        """Get the backend assigned to a specific agent."""
        backend_id = self.agent_backend_mapping.get(agent_id)
        if backend_id is None:
            # Return first available backend as default
            if self.backends:
                return list(self.backends.values())[0]
            raise ValueError("No backends available")
        return self.backends[backend_id]

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics from all backends."""
        return {
            backend_id: backend.get_stats()
            for backend_id, backend in self.backends.items()
        }

    def print_stats(self):
        """Print statistics for all backends."""
        print("\n" + "="*60)
        print("LLM BACKEND STATISTICS")
        print("="*60)
        for backend_id, stats in self.get_all_stats().items():
            print(f"\n{backend_id}:")
            print(f"  Model: {stats['model_name']}")
            print(f"  Architecture: {stats['architecture_type']}")
            print(f"  Generations: {stats['generation_count']}")
            print(f"  Total tokens: {stats['total_tokens_generated']}")
        print("="*60 + "\n")


def get_llm_backend(use_mock: bool = False, model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                   backend_type: str = "llama"):
    """
    Factory function to get appropriate LLM backend.

    Args:
        use_mock: If True, use mock backend for testing
        model_name: Model to load (if not using mock)
        backend_type: Type of backend ("llama", "mistral", "gpt", "small", "mock")

    Returns:
        LLM backend instance
    """
    if use_mock:
        return MockLLMBackend()

    if backend_type == "llama":
        return LlamaBackend(model_name=model_name)
    elif backend_type == "mistral":
        return MistralBackend(model_name=model_name)
    elif backend_type == "gpt":
        return GPTStyleBackend(model_name=model_name)
    elif backend_type == "small":
        return SmallModelBackend(model_name=model_name)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def create_multi_llm_setup(config: Dict[str, Dict]) -> MultiLLMManager:
    """
    Create a multi-LLM setup from configuration.

    Args:
        config: Dictionary mapping backend_id to backend configuration
            Example:
            {
                "llama": {"type": "llama", "model": "meta-llama/Llama-3.1-8B-Instruct"},
                "gpt": {"type": "gpt", "model": "gpt2"},
                "small": {"type": "small", "model": "microsoft/phi-2"}
            }

    Returns:
        MultiLLMManager with all backends loaded
    """
    manager = MultiLLMManager()

    for backend_id, backend_config in config.items():
        backend_type = backend_config.get("type", "mock")
        model_name = backend_config.get("model", "mock")
        use_mock = backend_config.get("use_mock", False)

        backend = get_llm_backend(
            use_mock=use_mock,
            model_name=model_name,
            backend_type=backend_type
        )
        manager.add_backend(backend_id, backend)

    return manager