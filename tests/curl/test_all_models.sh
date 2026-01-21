#!/bin/bash

# Test all deployed models with sample queries

echo "Testing all models..."

# Test queries by model specialty
declare -A QUERIES=(
    ["zelda-majora"]="Pitch a weird side-quest for Oracle of Secrets."
    ["zelda-din"]="Optimize this 65816 loop for speed:\\nLDX #$00\\n.loop\\nLDA $7E1234,X\\nCLC\\nADC #$01\\nSTA $7E1234,X\\nINX\\nCPX #$10\\nBNE .loop"
    ["zelda-farore"]="Complete this routine header with a brief purpose and inputs: `Routine_CopyTiles`."
    ["zelda-veran"]="Design a state machine for a dungeon progression system."
    ["zelda-hylia"]="Where can I find documentation on the memory map?"
    ["zelda-scribe"]="Write a clean docstring for a function that compresses tilemaps."
    ["scawful-echo"]="give me your vibe in one short paragraph."
    ["scawful-memory"]="Summarize these notes into 4 crisp bullets with dates kept: 2026-01-19: echo eval; 2026-01-20: LMStudio cleanup."
    ["scawful-muse"]="Brainstorm 5 names for a ROM hacking assistant tool."
)

declare -A PORTS=(
    ["zelda-majora"]=5000
    ["zelda-din"]=5001
    ["zelda-farore"]=5002
    ["zelda-veran"]=5003
    ["zelda-hylia"]=5004
    ["zelda-scribe"]=5005
    ["scawful-echo"]=5006
    ["scawful-memory"]=5007
    ["scawful-muse"]=5008
)

for model_name in "${!PORTS[@]}"; do
    port=${PORTS[$model_name]}
    query="${QUERIES[$model_name]}"

    echo ""
    echo "Testing $model_name on port $port..."

    curl -X POST "http://localhost:$port/chat" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$query\"}" \
        --connect-timeout 5 \
        --max-time 30 \
        -s | jq . || echo "✗ Connection failed"
done
