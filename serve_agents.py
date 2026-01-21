#!/usr/bin/env python3
"""
AFS Agents Inference Server
Serves all 5 trained agents via a simple API using LoRA adapters directly.
No merge/convert needed - loads adapters on-demand.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import argparse
from pathlib import Path

class AgentInference:
    def __init__(self, base_model="Qwen/Qwen2.5-Coder-3B-Instruct"):
        print(f"Loading base model: {base_model}")

        # Load in 4-bit for efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        self.current_adapter = None
        self.model = self.base_model

    def load_adapter(self, adapter_path: str, agent_name: str):
        """Load a LoRA adapter"""
        if self.current_adapter == agent_name:
            print(f"✓ {agent_name} already loaded")
            return

        print(f"Loading {agent_name} adapter from {adapter_path}...")
        try:
            # Remove previous adapter if any
            if self.current_adapter:
                # Reset to base model
                self.model = self.base_model

            # Load new adapter
            self.model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                adapter_name=agent_name
            )
            self.current_adapter = agent_name
            print(f"✓ {agent_name} loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load {agent_name}: {e}")
            raise

    def generate(self, prompt: str, max_new_tokens=512, temperature=0.7):
        """Generate response"""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant's response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]

        return response.strip()

import sys
sys.path.append(str(Path(__file__).parent / "src"))
try:
    from afs.config_loader import get_chat_registry
except ImportError:
    def get_chat_registry(): return {"models": []}

def main():
    # Load available agents from registry
    registry = get_chat_registry()
    available_agents = [
        m["name"] for m in registry.get("models", []) 
        if "oracle" in m.get("tags", []) or m.get("name") in ["majora"]
    ]
    
    # Fallback if registry is empty
    if not available_agents:
        available_agents = ["nayru", "majora", "din", "farore", "veran"]

    parser = argparse.ArgumentParser(description="AFS Agents Inference Server")
    parser.add_argument("--adapters-dir", type=str,
                       default=str(Path.home() / "Downloads/afs_models"),
                       help="Directory containing LoRA adapters")
    parser.add_argument("--agent", type=str, required=True,
                       help=f"Agent to use (available: {', '.join(available_agents)})")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Prompt to send to the agent")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="Max tokens to generate")
    args = parser.parse_args()

    if args.agent not in available_agents:
        print(f"Error: Agent '{args.agent}' not found. Available: {available_agents}")
        sys.exit(1)

    # Initialize inference engine
    engine = AgentInference()

    # Load specific agent
    adapter_path = Path(args.adapters_dir) / f"{args.agent}-lora"
    engine.load_adapter(str(adapter_path), args.agent)

    # Generate response
    print(f"\n{'='*60}")
    print(f"Agent: {args.agent.upper()}")
    print(f"{'='*60}")
    print(f"Prompt: {args.prompt}\n")
    print(f"Response:")
    print(f"{'='*60}")

    # Get temperature from registry if available
    temperature = 0.7
    for m in registry.get("models", []):
        if m["name"] == args.agent:
            temperature = m.get("parameters", {}).get("temperature", 0.7)
            break

    response = engine.generate(args.prompt, max_new_tokens=args.max_tokens, temperature=temperature)
    print(response)
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
