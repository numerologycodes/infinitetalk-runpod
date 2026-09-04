# Use specific version of nvidia cuda image
FROM wlsdml1114/engui_genai-base_blackwell:1.1 as runtime

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
# INFINITETALK / MULTITALK TRANSFORMERS COMPATIBILITY
# ---------------------------------------------------------
#
# The base/current dependency chain installs Transformers 5.x.
# InfiniteTalk MultiTalk Wav2Vec currently fails there with:
#
# embeddings.hidden_states == None
#
# Pin the runtime to the compatible Transformers 4.x branch.
# Do this AFTER all custom-node requirements have been installed
# so a previous installation cannot overwrite this version.
# ---------------------------------------------------------

RUN pip install --no-cache-dir --force-reinstall \
    "transformers==4.53.2"

# ---------------------------------------------------------
# VERIFY TRANSFORMERS VERSION
# ---------------------------------------------------------

RUN python - <<'PY'
import transformers

expected = "4.53.2"
actual = transformers.__version__

print(f"Transformers installed version: {actual}")

if actual != expected:
    raise RuntimeError(
        f"Wrong Transformers version. "
        f"Expected {expected}, got {actual}"
    )

print("Transformers compatibility pin verified.")
PY

# ---------------------------------------------------------
# VERIFY MULTITALK WAV2VEC CODE
# ---------------------------------------------------------
#
# Do NOT patch the source here.
# Current WanVideoWrapper should explicitly request
# output_hidden_states=True.
#
# If upstream changes and this disappears, stop the build
# instead of silently deploying an unknown configuration.
# ---------------------------------------------------------

RUN python - <<'PY'
from pathlib import Path

path = Path(
    "/ComfyUI/custom_nodes/"
    "ComfyUI-WanVideoWrapper/multitalk/nodes.py"
)

if not path.exists():
    raise RuntimeError(
        f"Cannot find MultiTalk nodes.py: {path}"
    )

text = path.read_text(encoding="utf-8")

if "output_hidden_states=True" not in text:
    raise RuntimeError(
        "WanVideoWrapper MultiTalk no longer explicitly "
        "requests Wav2Vec hidden states. Build stopped."
    )

print("MultiTalk Wav2Vec hidden-state request verified.")
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
