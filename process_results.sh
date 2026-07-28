#!/bin/bash

# Configuration variables
SRC_SCRIPT="project.py"
DATA_OUT="results.csv"

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

# Use csv-aware field extraction for the spreadsheet output
echo "🔍 Sample of Extracted Detections (Raw Box Coordinates):"
awk -F',' 'NR > 1 { printf "%s,%s,%s,%s\n", $5, $6, $7, $8; if (++count == 3) exit }' "$DATA_OUT"

# Use awk, pipes, and filters to dynamically parse data telemetry and calculate analytics
echo ""
awk -F',' 'NR > 1 { sum += $3; count++ } END { if (count > 0) printf "Average Object Confidence: %.2f%% (Total Detections: %d)\n", (sum/count)*100, count; else print "No active metrics logged."; }' "$DATA_OUT"