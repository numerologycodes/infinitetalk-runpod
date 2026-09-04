#!/bin/bash

set -e

echo "=========================================="
echo "InfiniteTalk RunPod starting..."
echo "=========================================="

VOLUME_MODELS="/runpod-volume/infinitetalk-models"
COMFY_MODELS="/ComfyUI/models"

# ---------------------------------------------------------
# VERIFY NETWORK VOLUME
# ---------------------------------------------------------

echo "Checking Network Volume..."

if [ ! -d "$VOLUME_MODELS" ]; then
    echo "ERROR: Network Volume models directory not found:"
    echo "$VOLUME_MODELS"
    exit 1
fi

echo "Network Volume found."

# ---------------------------------------------------------
# CONNECT PERSISTENT MODELS TO COMFYUI
# ---------------------------------------------------------

echo "Connecting persistent models to ComfyUI..."

mkdir -p "$COMFY_MODELS/diffusion_models"
mkdir -p "$COMFY_MODELS/loras"
mkdir -p "$COMFY_MODELS/vae"
mkdir -p "$COMFY_MODELS/text_encoders"
mkdir -p "$COMFY_MODELS/clip_vision"

ln -sfn \
"$VOLUME_MODELS/diffusion_models/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors" \
"$COMFY_MODELS/diffusion_models/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors"

ln -sfn \
"$VOLUME_MODELS/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors" \
"$COMFY_MODELS/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors"

ln -sfn \
"$VOLUME_MODELS/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors" \
"$COMFY_MODELS/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"

ln -sfn \
"$VOLUME_MODELS/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" \
"$COMFY_MODELS/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"

ln -sfn \
"$VOLUME_MODELS/vae/Wan2_1_VAE_bf16.safetensors" \
"$COMFY_MODELS/vae/Wan2_1_VAE_bf16.safetensors"

ln -sfn \
"$VOLUME_MODELS/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors" \
"$COMFY_MODELS/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors"

ln -sfn \
"$VOLUME_MODELS/clip_vision/clip_vision_h.safetensors" \
"$COMFY_MODELS/clip_vision/clip_vision_h.safetensors"

ln -sfn \
"$VOLUME_MODELS/diffusion_models/MelBandRoformer_fp16.safetensors" \
"$COMFY_MODELS/diffusion_models/MelBandRoformer_fp16.safetensors"

echo "Persistent models connected."

# ---------------------------------------------------------
# VERIFY REQUIRED MODELS
# ---------------------------------------------------------

echo "Verifying required models..."

REQUIRED_MODELS=(
"$COMFY_MODELS/diffusion_models/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors"
"$COMFY_MODELS/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors"
"$COMFY_MODELS/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"
"$COMFY_MODELS/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
"$COMFY_MODELS/vae/Wan2_1_VAE_bf16.safetensors"
"$COMFY_MODELS/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors"
"$COMFY_MODELS/clip_vision/clip_vision_h.safetensors"
"$COMFY_MODELS/diffusion_models/MelBandRoformer_fp16.safetensors"
)

for model in "${REQUIRED_MODELS[@]}"; do
    if [ ! -s "$model" ]; then
        echo "ERROR: Required model missing or empty:"
        echo "$model"
        exit 1
    fi
done

echo "All required models verified."

# ---------------------------------------------------------
# START COMFYUI
# ---------------------------------------------------------

echo "Starting ComfyUI..."

python /ComfyUI/main.py \
    --listen \
    --use-sage-attention &

# ---------------------------------------------------------
# WAIT FOR COMFYUI
# ---------------------------------------------------------

echo "Waiting for ComfyUI..."

max_wait=120
wait_count=0

while [ "$wait_count" -lt "$max_wait" ]; do

    if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
        echo "ComfyUI is ready."
        break
    fi

    echo "Waiting for ComfyUI... ($wait_count/$max_wait)"

    sleep 2
    wait_count=$((wait_count + 2))

done

if [ "$wait_count" -ge "$max_wait" ]; then
    echo "ERROR: ComfyUI failed to start within $max_wait seconds."
    exit 1
fi

# ---------------------------------------------------------
# START RUNPOD HANDLER
# ---------------------------------------------------------

echo "Starting RunPod handler..."

exec python /handler.py
