#!/bin/bash

# Configuration variables
SRC_SCRIPT="project.py"
DATA_OUT="results.txt"

echo "=== Initiating Core Batch Processing Pipeline ==="

if [ ! -f "$SRC_SCRIPT" ]; then
    echo "Error: Source script '$SRC_SCRIPT' not found."
    exit 1
fi

# Run the vision processing pipeline explicitly via the local virtual environment interpreter
python "$SRC_SCRIPT"

if [ ! -f "$DATA_OUT" ] || [ ! -s "$DATA_OUT" ]; then
    echo "Error: Metrics generation failed or '$DATA_OUT' is empty."
    exit 1
fi

echo "--------------------------------------------------"
echo "Analyzing Processing Output Data Metrics:"

# Use sed to isolate the raw pixel coordinates from the logs
echo "🔍 Sample of Extracted Detections (Raw Box Coordinates):"
sed -n 's/.*bbox=\([^)]*\).*/\1/p' "$DATA_OUT" | head -n 3

# Use awk, pipes, and filters to dynamically parse data telemetry and calculate analytics
echo ""
awk -F'conf=' '{print $2}' "$DATA_OUT" | awk -F',' '{print $1}' | awk '
    { sum += $1; count++ } 
    END { 
        if (count > 0) 
            printf "Average Object Confidence: %.2f%% (Total Detections: %d)\n", (sum/count)*100, count; 
        else 
            print "No active metrics logged."; 
    }'