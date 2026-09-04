# Use specific version of nvidia cuda image
FROM wlsdml1114/engui_genai-base_blackwell:1.1 as runtime

# Install system tools
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
# FIX: INFINITETALK WAV2VEC HIDDEN STATES
# ---------------------------------------------------------
#
# InfiniteTalk later accesses embeddings.hidden_states.
# Force the Wav2Vec model to return hidden states.
#
# If WanVideoWrapper changes upstream and this patch can no
# longer be applied safely, STOP THE BUILD instead of deploying
# a worker with an unknown configuration.
# ---------------------------------------------------------

RUN python - <<'PY'
from pathlib import Path
import re

path = Path(
    "/ComfyUI/custom_nodes/"
    "ComfyUI-WanVideoWrapper/multitalk/nodes.py"
)

if not path.exists():
    raise RuntimeError(
        f"Cannot find WanVideoWrapper multitalk nodes.py: {path}"
    )

text = path.read_text(encoding="utf-8")

# If upstream already explicitly requests hidden states,
# there is nothing to change.
if re.search(
    r"output_hidden_states\s*=\s*True",
    text
):
    print("Wav2Vec hidden states are already enabled upstream.")

else:
    # Locate the model call that creates `embeddings`.
    #
    # Example:
    # embeddings = self.wav2vec2(...)
    #
    # and add:
    # output_hidden_states=True, return_dict=True

    pattern = re.compile(
        r"(embeddings\s*=\s*"
        r"self\.[A-Za-z0-9_]*wav2vec[A-Za-z0-9_]*"
        r"\s*\()",
        re.IGNORECASE
    )

    match = pattern.search(text)

    if not match:
        raise RuntimeError(
            "Could not locate the Wav2Vec embeddings call. "
            "Build stopped intentionally."
        )

    start = match.end()

    # Find the matching closing parenthesis of the call.
    depth = 1
    i = start

    while i < len(text) and depth:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        i += 1

    if depth != 0:
        raise RuntimeError(
            "Could not determine the end of the Wav2Vec call."
        )

    closing_paren = i - 1

    existing_args = text[start:closing_paren].rstrip()

    if existing_args.endswith(","):
        addition = (
            "\n"
            "            output_hidden_states=True,\n"
            "            return_dict=True"
        )
    else:
        addition = (
            ",\n"
            "            output_hidden_states=True,\n"
            "            return_dict=True"
        )

    text = (
        text[:closing_paren]
        + addition
        + text[closing_paren:]
    )

    path.write_text(text, encoding="utf-8")

    print("Applied InfiniteTalk Wav2Vec hidden-states patch.")

# ---------------------------------------------------------
# VERIFY PATCH
# ---------------------------------------------------------

verify = path.read_text(encoding="utf-8")

if not re.search(
    r"output_hidden_states\s*=\s*True",
    verify
):
    raise RuntimeError(
        "Patch verification failed: "
        "output_hidden_states=True is missing."
    )

print("InfiniteTalk Wav2Vec patch verified successfully.")
PY

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
