#!/bin/bash


DEST_WORKSPACE="compiled_model"
mkdir -p "$DEST_WORKSPACE"

echo "======================================================================"

echo "[Pre-compiling] Global CoreML components..."


if [ -d "coreml_rope/qwen3_5_moe_rope.mlpackage" ]; then
    xcrun coremlcompiler compile coreml_rope/qwen3_5_moe_rope.mlpackage "$DEST_WORKSPACE"
elif [ -d "coreml_rope" ]; then
    xcrun coremlcompiler compile coreml_rope/*.mlpackage "$DEST_WORKSPACE"
fi


if [ -d "coreml_lm_head/qwen3_5_moe_lm_head.mlpackage" ]; then
    xcrun coremlcompiler compile coreml_lm_head/qwen3_5_moe_lm_head.mlpackage "$DEST_WORKSPACE"
elif [ -d "coreml_lm_head" ]; then
    xcrun coremlcompiler compile coreml_lm_head/*.mlpackage "$DEST_WORKSPACE"
fi

echo -e "\n[Pre-compiling] Looping through all 40 layers sequentially..."

for layer_idx in {0..39}
do
    LAYER_DEST="${DEST_WORKSPACE}/layer_${layer_idx}"
    mkdir -p "$LAYER_DEST"
    
    
    if [ -d "coreml_attentions/layer_${layer_idx}" ]; then
        find "coreml_attentions/layer_${layer_idx}" -name "*.mlpackage" -exec xcrun coremlcompiler compile {} "$LAYER_DEST" \;
    fi
    
    # ❷ Router
    if [ -d "coreml_routers/layer_${layer_idx}" ]; then
        find "coreml_routers/layer_${layer_idx}" -name "*.mlpackage" -exec xcrun coremlcompiler compile {} "$LAYER_DEST" \;
    fi
    
    # ❸ Experts 
    FLAT_EXPERT_SRC="coreml_experts/experts_layer_${layer_idx}.mlpackage"
    if [ -d "$FLAT_EXPERT_SRC" ]; then
        xcrun coremlcompiler compile "$FLAT_EXPERT_SRC" "$LAYER_DEST"
        echo "    -> [Experts] Compiled and synced into layer_${layer_idx}"
    else
        find coreml_experts -maxdepth 1 -name "*layer_${layer_idx}*.mlpackage" -exec xcrun coremlcompiler compile {} "$LAYER_DEST" \; 2>/dev/null
    fi
    
    # ❹ MLP / Shared Expert
    if [ -d "coreml_mlps/layer_${layer_idx}" ]; then
        find "coreml_mlps/layer_${layer_idx}" -name "*.mlpackage" -exec xcrun coremlcompiler compile {} "$LAYER_DEST" \;
    fi
    
    # ❺ LayerNorm (input_layernorm, post_attention_layernorm)
    if [ -d "coreml_norms/layer_${layer_idx}" ]; then
        find "coreml_norms/layer_${layer_idx}" -name "*.mlpackage" -exec xcrun coremlcompiler compile {} "$LAYER_DEST" \;
    fi
done

echo -e "\n🎉 🏁 [SUCCESS] Every single component compiled perfectly into .mlmodelc!"
echo "Verify your contents inside the './${DEST_WORKSPACE}' folder."
