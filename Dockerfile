# Use specific version of nvidia cuda image
FROM wlsdml1114/engui_genai-base_blackwell:1.1 AS runtime

# ---------------------------------------------------------
# SYSTEM TOOLS
# ---------------------------------------------------------

RUN apt-get update && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -U "huggingface_hub[hf_transfer]"
RUN pip install runpod websocket-client librosa

WORKDIR /

# ---------------------------------------------------------
# COMFYUI
# ---------------------------------------------------------

RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    pip install -r requirements.txt

# ---------------------------------------------------------
# COMFYUI MANAGER
# ---------------------------------------------------------

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    pip install -r requirements.txt

# ---------------------------------------------------------
# GGUF
# ---------------------------------------------------------

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    cd ComfyUI-GGUF && \
    pip install -r requirements.txt

# ---------------------------------------------------------
# KJ NODES
# ---------------------------------------------------------

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && \
    pip install -r requirements.txt

# ---------------------------------------------------------
# VIDEO HELPER SUITE
# ---------------------------------------------------------

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt

# ---------------------------------------------------------
# WAN BLOCKSWAP
# ---------------------------------------------------------

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/orssorbit/ComfyUI-wanBlockswap

# ---------------------------------------------------------
# MELBAND ROFORMER
# ---------------------------------------------------------

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-MelBandRoFormer && \
    cd ComfyUI-MelBandRoFormer && \
    pip install -r requirements.txt

# ---------------------------------------------------------
# WAN VIDEO WRAPPER / INFINITETALK
# ---------------------------------------------------------

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-WanVideoWrapper && \
    pip install -r requirements.txt

# ---------------------------------------------------------
# MULTITALK WAV2VEC PATCH
# ---------------------------------------------------------
#
# Problem observed in our RunPod worker:
#
# embeddings.hidden_states == None
#
# Patch only the two semantic statements we need instead of
# matching a large formatting-sensitive source block.
#
# ---------------------------------------------------------

RUN python - <<'PY'
from pathlib import Path

path = Path(
    "/ComfyUI/custom_nodes/"
    "ComfyUI-WanVideoWrapper/multitalk/nodes.py"
)

if not path.exists():
    raise RuntimeError(f"Cannot find MultiTalk nodes.py: {path}")

text = path.read_text(encoding="utf-8")

call_marker = (
    "embeddings = wav2vec2("
    "audio_feature.to(dtype), "
    "seq_len=int(video_length), "
    "output_hidden_states=True)"
)

stack_marker = (
    "audio_emb = torch.stack("
    "embeddings.hidden_states[1:], dim=1).squeeze(0)"
)

if call_marker not in text:
    raise RuntimeError(
        "Could not find MultiTalk Wav2Vec invocation. "
        "Upstream WanVideoWrapper changed."
    )

if stack_marker not in text:
    raise RuntimeError(
        "Could not find MultiTalk hidden_states consumption. "
        "Upstream WanVideoWrapper changed."
    )

patched_call = """embeddings = wav2vec2(
                audio_feature.to(dtype),
                seq_len=int(video_length),
                output_hidden_states=True,
                return_dict=True,
            )"""

patched_stack = """hidden_states = getattr(
                embeddings,
                "hidden_states",
                None,
            )

            if hidden_states is None:
                try:
                    hidden_states = embeddings["hidden_states"]
                except (TypeError, KeyError):
                    hidden_states = None

            if hidden_states is None:
                raise RuntimeError(
                    "MultiTalk Wav2Vec returned hidden_states=None "
                    "even with output_hidden_states=True and return_dict=True"
                )

            if len(hidden_states) <= 1:
                raise RuntimeError(
                    "MultiTalk Wav2Vec returned insufficient hidden states: "
                    f"{len(hidden_states)}"
                )

            audio_emb = torch.stack(
                hidden_states[1:],
                dim=1,
            ).squeeze(0)"""

text = text.replace(call_marker, patched_call, 1)
text = text.replace(stack_marker, patched_stack, 1)

path.write_text(text, encoding="utf-8")

print("MultiTalk Wav2Vec patch applied.")
PY

# ---------------------------------------------------------
# VERIFY MULTITALK PATCH
# ---------------------------------------------------------

RUN python - <<'PY'
from pathlib import Path

path = Path(
    "/ComfyUI/custom_nodes/"
    "ComfyUI-WanVideoWrapper/multitalk/nodes.py"
)

text = path.read_text(encoding="utf-8")

required = [
    "return_dict=True",
    'hidden_states = getattr(',
    '"hidden_states"',
    "MultiTalk Wav2Vec returned hidden_states=None",
    "hidden_states[1:]",
]

for marker in required:
    if marker not in text:
        raise RuntimeError(
            f"MultiTalk patch verification failed: {marker}"
        )

if "embeddings.hidden_states[1:]" in text:
    raise RuntimeError(
        "Old unsafe embeddings.hidden_states access still exists."
    )

compile(text, str(path), "exec")

print("MultiTalk Wav2Vec patch verified.")
print("nodes.py syntax OK.")
PY

# ---------------------------------------------------------
# ENVIRONMENT DIAGNOSTICS
# ---------------------------------------------------------

RUN python - <<'PY'
import transformers
import diffusers
import huggingface_hub

print("Transformers:", transformers.__version__)
print("Diffusers:", diffusers.__version__)
print("HuggingFace Hub:", huggingface_hub.__version__)
PY

# ---------------------------------------------------------
# FINAL DEPENDENCY CHECK
# ---------------------------------------------------------

RUN pip check

# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

RUN wget -q \
    https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors \
    -O /ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors

RUN wget -q \
    https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors \
    -O /ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors

RUN wget -q \
    https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors \
    -O /ComfyUI/models/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors

RUN wget -q \
    https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors \
    -O /ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors

RUN wget -q \
    https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors \
    -O /ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors

RUN wget -q \
    https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors \
    -O /ComfyUI/models/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors

RUN wget -q \
    https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors \
    -O /ComfyUI/models/clip_vision/clip_vision_h.safetensors

RUN wget -q \
    https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors \
    -O /ComfyUI/models/diffusion_models/MelBandRoformer_fp16.safetensors

# ---------------------------------------------------------
# RUNPOD APPLICATION
# ---------------------------------------------------------

COPY . .

RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
