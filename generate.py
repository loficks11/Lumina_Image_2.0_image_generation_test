import argparse
import json
import os
import tempfile
import time
from pathlib import Path


MODEL_ID = "Alpha-VLLM/Lumina-Image-2.0"
HEIGHT = 1024
WIDTH = 1024
GUIDANCE_SCALE = 4.0
NUM_INFERENCE_STEPS = 50
CFG_TRUNC_RATIO = 0.25
CFG_NORMALIZATION = True
SEEDS = [0]
DIRECT_GPU_MIN_VRAM_GIB = 24.0

PROMPTS = [
    "A black colored banana.",
    "One cat and two dogs sitting on the grass.",
    "A horse riding an astronaut.",
    "A storefront with 'Hello World' written on it.",
    "Paying for a quarter-sized pizza with a pizza-sized quarter.",
    "An emoji of a baby panda wearing a red hat, green gloves, red shirt, and green pants.",
    "A portrait photo of a kangaroo wearing an orange hoodie and blue sunglasses standing on the grass in front of the Sydney Opera House holding a sign on the chest that says Welcome Friends!",
    "A raccoon wearing formal clothes, wearing a tophat and holding a cane. The raccoon is holding a garbage bag. Oil painting in the style of Vincent Van Gogh.",
    "a photo of a cake left of a bus",
    "a metallic spoon and a wooden bowl",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the Lumina-Image-2.0 prompt comparison set."
    )
    parser.add_argument(
        "--limit",
        type=int,
        choices=range(1, len(PROMPTS) + 1),
        metavar=f"1-{len(PROMPTS)}",
        help="Generate only the first N prompts (use --limit 1 for a smoke test).",
    )
    return parser.parse_args()


def load_metadata(metadata_path):
    if not metadata_path.exists():
        return {"model": MODEL_ID, "generations": []}

    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if metadata.get("model") != MODEL_ID or not isinstance(
        metadata.get("generations"), list
    ):
        raise ValueError(f"Unexpected metadata format in {metadata_path}")
    return metadata


def write_metadata(metadata_path, metadata):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=metadata_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(metadata, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
        os.replace(temporary_path, metadata_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def next_output_path(output_dir, prompt_id, seed):
    base_name = f"prompt_{prompt_id:02d}_seed_{seed}"
    output_path = output_dir / f"{base_name}.png"
    suffix = 2
    while output_path.exists():
        output_path = output_dir / f"{base_name}_{suffix}.png"
        suffix += 1
    return output_path


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "outputs" / "lumina_image_2"
    metadata_path = project_dir / "metadata" / "lumina_image_2.json"
    metadata = load_metadata(metadata_path)

    try:
        import torch
        from diffusers import Lumina2Pipeline
    except ImportError as error:
        raise SystemExit(
            "Missing model dependencies. Install requirements.txt and a CUDA-enabled "
            "PyTorch build selected for this machine's driver and CUDA runtime."
        ) from error

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU not detected. Lumina-Image-2.0 cannot be run efficiently on "
            "this CPU/integrated-GPU environment. Use a CUDA-enabled AIKU server "
            "or another NVIDIA GPU host, then run this script there."
        )

    device_properties = torch.cuda.get_device_properties(0)
    vram_gib = device_properties.total_memory / (1024**3)
    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported(including_emulation=False)
        else torch.float16
    )
    use_cpu_offload = vram_gib < DIRECT_GPU_MIN_VRAM_GIB

    print(f"Model: {MODEL_ID}")
    print(f"GPU: {device_properties.name} ({vram_gib:.1f} GiB VRAM)")
    print(f"dtype: {dtype}; CPU offload: {'enabled' if use_cpu_offload else 'disabled'}")
    if use_cpu_offload:
        print(
            "VRAM is below the conservative 24 GiB direct-load threshold; "
            "enabling model CPU offload."
        )

    total_started = time.perf_counter()
    load_started = time.perf_counter()
    try:
        pipe = Lumina2Pipeline.from_pretrained(MODEL_ID, dtype=dtype)
        if use_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as error:
        raise SystemExit(
            "CUDA ran out of VRAM while loading Lumina-Image-2.0. Model CPU offload "
            "is enabled below 24 GiB, but this GPU still has too little available "
            "memory. Use a GPU with more VRAM or close other GPU workloads."
        ) from error
    load_time = time.perf_counter() - load_started
    print(f"Model load time: {load_time:.1f} sec")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_count = args.limit or len(PROMPTS)
    generation_times = []

    try:
        for prompt_index, prompt in enumerate(PROMPTS[:prompt_count], start=1):
            for seed in SEEDS:
                output_path = next_output_path(output_dir, prompt_index, seed)
                torch.cuda.synchronize()
                generation_started = time.perf_counter()
                result = pipe(
                    prompt,
                    height=HEIGHT,
                    width=WIDTH,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    cfg_trunc_ratio=CFG_TRUNC_RATIO,
                    cfg_normalization=CFG_NORMALIZATION,
                    generator=torch.Generator("cpu").manual_seed(seed),
                )
                torch.cuda.synchronize()
                generation_time = time.perf_counter() - generation_started
                image = result.images[0]

                temporary_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        dir=output_dir,
                        suffix=".png",
                        delete=False,
                    ) as temporary_file:
                        temporary_path = Path(temporary_file.name)
                    image.save(temporary_path, format="PNG")
                    os.replace(temporary_path, output_path)
                finally:
                    if temporary_path is not None and temporary_path.exists():
                        temporary_path.unlink()

                record = {
                    "model": MODEL_ID,
                    "prompt_id": prompt_index,
                    "prompt": prompt,
                    "seed": seed,
                    "height": HEIGHT,
                    "width": WIDTH,
                    "guidance_scale": GUIDANCE_SCALE,
                    "num_inference_steps": NUM_INFERENCE_STEPS,
                    "cfg_trunc_ratio": CFG_TRUNC_RATIO,
                    "cfg_normalization": CFG_NORMALIZATION,
                    "generation_time_sec": round(generation_time, 3),
                    "output_path": output_path.relative_to(project_dir).as_posix(),
                }
                metadata["generations"].append(record)
                write_metadata(metadata_path, metadata)
                generation_times.append(generation_time)
                print(
                    f"Prompt {prompt_index:02d}, seed {seed}: "
                    f"{generation_time:.1f} sec -> {record['output_path']}"
                )
    except torch.cuda.OutOfMemoryError as error:
        raise SystemExit(
            "CUDA ran out of VRAM during image generation. Previously completed "
            "images and metadata have been preserved. Use a GPU with more VRAM; "
            "CPU offload is already enabled when VRAM is below 24 GiB."
        ) from error

    total_time = time.perf_counter() - total_started
    average_time = sum(generation_times) / len(generation_times)
    print(f"Metadata: {metadata_path.relative_to(project_dir).as_posix()}")
    print(f"Completed: {len(generation_times)} image(s)")
    print(f"Total elapsed time (including model load): {total_time:.1f} sec")
    print(f"Average image generation time: {average_time:.1f} sec")


if __name__ == "__main__":
    main()