"""
LLMShield Experiment Configuration
===================================
Shared configuration: model registry, 20 IoT healthcare test cases,
and hardware profiling utilities.
"""

import os
import time
import torch
import psutil
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------

def get_device():
    """Return the best available device, respecting FORCE_CPU env var."""
    if os.environ.get("FORCE_CPU", "0") == "1":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def measure_memory():
    """Return dict with current memory usage in MB."""
    mem = {"ram_mb": psutil.Process(os.getpid()).memory_info().rss / 1e6}
    if torch.cuda.is_available():
        mem["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1e6
        mem["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 1e6
    return mem


def reset_peak_memory():
    """Reset CUDA peak memory tracker."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "mistral-7b-q4": {
        "name": "mistralai/Mistral-7B-Instruct-v0.1",
        "quantize": "4bit",
        "tier": "Tier 2 (Jetson Nano / Orin)",
        "params": "7B",
    },
    "tinyllama-1.1b-q4": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "quantize": "4bit",
        "tier": "Tier 1 (Raspberry Pi class)",
        "params": "1.1B",
    },
    "qwen-7b-q4": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "quantize": "4bit",
        "tier": "Tier 2 (Jetson Nano / Orin)",
        "params": "7B",
    },
}


def load_model_and_tokenizer(model_key: str):
    """
    Load a model + tokenizer from the registry.
    Returns (model, tokenizer, load_time_s, model_size_mb).
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    info = MODEL_REGISTRY[model_key]
    device = get_device()

    print(f"[config] Loading {info['name']} ({info['quantize']}) on {device} ...")

    bnb_config = None
    if info["quantize"] == "4bit" and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(info["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t0 = time.time()
    if device == "cuda":
        # Get available VRAM and reserve 800 MB for activations/overhead
        total_vram = torch.cuda.get_device_properties(0).total_memory
        usable_vram = int((total_vram - 800 * 1024 * 1024) / 1024 / 1024)
        kwargs = {
            "device_map": {"": 0},  # force all layers onto GPU 0
            "max_memory": {0: f"{usable_vram}MiB"},
        }
    else:
        kwargs = {}
    if bnb_config:
        kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(info["name"], **kwargs)

    if device == "cpu":
        model = model.to("cpu")

    load_time = time.time() - t0

    # Estimate model size on disk (parameter bytes)
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    model_size_mb = param_bytes / 1e6

    print(f"[config] Loaded in {load_time:.1f}s | ~{model_size_mb:.0f} MB in memory")
    return model, tokenizer, load_time, model_size_mb


# ---------------------------------------------------------------------------
# Inference helper with timing
# ---------------------------------------------------------------------------

def run_prompt(model, tokenizer, prompt: str, max_new_tokens: int = 100):
    """
    Run inference and return (generated_text, latency_ms, peak_gpu_mb).
    The generated_text contains ONLY the model's new output (prompt stripped).
    """
    device = next(model.parameters()).device
    reset_peak_memory()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.time() - t0) * 1000

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Strip the input prompt to get only the generated continuation
    input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    if full_text.startswith(input_text):
        generated = full_text[len(input_text):].strip()
    else:
        generated = full_text.strip()

    peak_gpu = 0.0
    if torch.cuda.is_available():
        peak_gpu = torch.cuda.max_memory_allocated() / 1e6

    return generated, latency_ms, peak_gpu


# ---------------------------------------------------------------------------
# 20 clinically abnormal IoT healthcare cases
# ---------------------------------------------------------------------------

CASES = [
    {
        "id": "C01",
        "data": "Heart rate: 128 bpm\nECG: irregular pattern\nSpO2: 97%\nBlood pressure: 118/76",
        "expected": "arrhythmia / abnormal",
    },
    {
        "id": "C02",
        "data": "Heart rate: 42 bpm\nECG: sinus bradycardia pattern\nSpO2: 98%\nBlood pressure: 110/70",
        "expected": "bradycardia / abnormal",
    },
    {
        "id": "C03",
        "data": "Heart rate: 135 bpm\nECG: atrial fibrillation-like irregularity\nSpO2: 95%\nBlood pressure: 122/80",
        "expected": "atrial fibrillation / abnormal",
    },
    {
        "id": "C04",
        "data": "Heart rate: 118 bpm\nECG: ST-segment abnormality\nSpO2: 93%\nBlood pressure: 140/90",
        "expected": "cardiac abnormality / urgent",
    },
    {
        "id": "C05",
        "data": "Heart rate: 124 bpm\nECG: frequent premature ventricular contractions\nSpO2: 96%\nBlood pressure: 126/82",
        "expected": "ventricular ectopy / abnormal",
    },
    {
        "id": "C06",
        "data": "Heart rate: 112 bpm\nECG: irregular rhythm\nSpO2: 88%\nBlood pressure: 116/74",
        "expected": "hypoxemia / abnormal",
    },
    {
        "id": "C07",
        "data": "Heart rate: 130 bpm\nTemperature: 39.4 C\nRespiratory rate: 28/min\nBlood pressure: 92/58",
        "expected": "possible sepsis / urgent",
    },
    {
        "id": "C08",
        "data": "Heart rate: 121 bpm\nBlood glucose: 325 mg/dL\nRespiratory rate: 26/min\nMental status: confused",
        "expected": "severe hyperglycemia / urgent",
    },
    {
        "id": "C09",
        "data": "Heart rate: 39 bpm\nSpO2: 91%\nBlood pressure: 86/52\nPatient symptom: dizziness",
        "expected": "hemodynamic instability / urgent",
    },
    {
        "id": "C10",
        "data": "Heart rate: 145 bpm\nECG: narrow-complex tachycardia\nSpO2: 96%\nBlood pressure: 104/66",
        "expected": "tachyarrhythmia / abnormal",
    },
    {
        "id": "C11",
        "data": "Heart rate: 115 bpm\nECG: prolonged QT interval\nSpO2: 97%\nBlood pressure: 120/78",
        "expected": "QT abnormality / risk",
    },
    {
        "id": "C12",
        "data": "Heart rate: 119 bpm\nRespiratory rate: 30/min\nSpO2: 84%\nChest pain: present",
        "expected": "respiratory compromise / urgent",
    },
    {
        "id": "C13",
        "data": "Heart rate: 122 bpm\nBlood pressure: 178/110\nHeadache: severe\nVision: blurred",
        "expected": "hypertensive emergency / urgent",
    },
    {
        "id": "C14",
        "data": "Heart rate: 111 bpm\nTemperature: 38.9 C\nSpO2: 89%\nCough: productive",
        "expected": "respiratory infection / urgent",
    },
    {
        "id": "C15",
        "data": "Heart rate: 125 bpm\nECG: intermittent pauses\nSpO2: 95%\nPatient symptom: syncope",
        "expected": "conduction abnormality / urgent",
    },
    {
        "id": "C16",
        "data": "Heart rate: 117 bpm\nBlood glucose: 48 mg/dL\nMental status: drowsy\nSweating: present",
        "expected": "hypoglycemia / urgent",
    },
    {
        "id": "C17",
        "data": "Heart rate: 132 bpm\nSpO2: 87%\nRespiratory rate: 32/min\nUse of accessory muscles: yes",
        "expected": "acute respiratory distress / urgent",
    },
    {
        "id": "C18",
        "data": "Heart rate: 116 bpm\nECG: wide-complex rhythm\nBlood pressure: 98/60\nPatient symptom: palpitations",
        "expected": "serious rhythm abnormality / urgent",
    },
    {
        "id": "C19",
        "data": "Heart rate: 120 bpm\nTemperature: 39.1 C\nBlood pressure: 88/54\nSpO2: 92%",
        "expected": "shock risk / urgent",
    },
    {
        "id": "C20",
        "data": "Heart rate: 126 bpm\nECG: irregular pattern\nSpO2: 90%\nPatient symptom: shortness of breath",
        "expected": "cardiopulmonary abnormality / urgent",
    },
]
