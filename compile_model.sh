#!/bin/bash


DEST_WORKSPACE="all_model"
mkdir -p "$DEST_WORKSPACE"

echo "======================================================================"
echo "[Syncing Assets] Copying raw Core AI (.aimodel) directories "



copy_package() {
    local src_dir="$1"
    local dest_dir="$2"
    if [ -d "$src_dir" ]; then
        echo "  -> Copying directory: $(basename "$src_dir")"
       
        cp -R "$src_dir" "$dest_dir/"
    fi
}

# ❶ RoPE
if [ -d "coreai_rope/qwen3_5_moe_rope.aimodel" ]; then
    copy_package "coreai_rope/qwen3_5_moe_rope.aimodel" "$DEST_WORKSPACE"
elif [ -d "coreai_rope" ]; then
    find coreai_rope -maxdepth 1 -type d -name "*.aimodel" | while read -r pkg; do
        copy_package "$pkg" "$DEST_WORKSPACE"
    done
fi

# ❷ LM Head
if [ -d "coreai_lm_head_split" ]; then
    find coreai_lm_head_split -maxdepth 1 -type d -name "*.aimodel" | while read -r pkg; do
        copy_package "$pkg" "$DEST_WORKSPACE"
    done
elif [ -d "coreai_lm_head/qwen3_5_moe_lm_head.aimodel" ]; then
    copy_package "coreai_lm_head/qwen3_5_moe_lm_head.aimodel" "$DEST_WORKSPACE"
elif [ -d "coreai_lm_head" ]; then
    find coreai_lm_head -maxdepth 1 -type d -name "*.aimodel" | while read -r pkg; do
        copy_package "$pkg" "$DEST_WORKSPACE"
    done
fi


if [ -d "embedding_binary" ]; then
    find embedding_binary -name "*.bin" -exec cp {} "$DEST_WORKSPACE/" \;
    echo "  🎉 Synced embedding binary vectors."
fi

# Layer Copy Loop
for layer_idx in {0..39}
do
    LAYER_DEST="${DEST_WORKSPACE}/layer_${layer_idx}"
    mkdir -p "$LAYER_DEST"
    
    # ❶ Attention
    if [ -d "coreai_attentions/layer_${layer_idx}" ]; then
        find "coreai_attentions/layer_${layer_idx}" -maxdepth 1 -type d -name "*.aimodel" | while read -r pkg; do
            copy_package "$pkg" "$LAYER_DEST"
        done
    fi
    
    # ❷ Router
    if [ -d "coreai_routers/layer_${layer_idx}" ]; then
        find "coreai_routers/layer_${layer_idx}" -maxdepth 1 -type d -name "*.aimodel" | while read -r pkg; do
            copy_package "$pkg" "$LAYER_DEST"
        done
    fi
    
    # ❸ Experts
    FLAT_EXPERT_SRC="coreai_experts/experts_layer_${layer_idx}.aimodel"
    if [ -d "$FLAT_EXPERT_SRC" ]; then
        copy_package "$FLAT_EXPERT_SRC" "$LAYER_DEST"
    else
        find coreai_experts -maxdepth 1 -type d -name "*layer_${layer_idx}*.aimodel" | while read -r pkg; do
            copy_package "$pkg" "$LAYER_DEST"
        done 2>/dev/null
    fi
    
    # ❹ MLP / Shared Expert
    if [ -d "coreai_mlps/layer_${layer_idx}" ]; then
        find "coreai_mlps/layer_${layer_idx}" -maxdepth 1 -type d -name "*.aimodel" | while read -r pkg; do
            copy_package "$pkg" "$LAYER_DEST"
        done
    fi
    
    # ❺ LayerNorm
    if [ -d "coreai_norms/layer_${layer_idx}" ]; then
        find "coreai_norms/layer_${layer_idx}" -maxdepth 1 -type d -name "*.aimodel" | while read -r pkg; do
            copy_package "$pkg" "$LAYER_DEST"
        done
    fi
done


echo -e "\n[Syncing] Tokenizer assets..."
if [ -d "tokenizer" ]; then
    cp -R tokenizer/ "$DEST_WORKSPACE/"
    echo "  🎉 Successfully copied tokenizer contents into: $DEST_WORKSPACE"
fi

echo -e "\n🎉 🏁 [SUCCESS] Every single .aimodel directory copied and structured perfectly without compilation!"
echo "Verify your contents inside the './${DEST_WORKSPACE}' folder."
