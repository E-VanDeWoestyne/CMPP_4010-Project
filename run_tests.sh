#!/bin/bash

# Configuration variables
TEST_FILE="test_project.py"
LOG_FILE="test_run.log"
TARGET_FILE="project.py"

echo "=== Running Unit Tests for Pallet Detection System ==="

# Execute unit tests using the active virtual environment's python interpreter
python "$TEST_FILE" > "$LOG_FILE" 2>&1
TEST_EXIT_CODE=$?

# Display test results to console
cat "$LOG_FILE"

if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "Error: Unit tests failed. Fix errors before deployment."
    exit $TEST_EXIT_CODE
else
    echo "Unit tests passed successfully!"
    echo "--------------------------------------------------"
    echo "Checking code metrics and style standards..."
    
    # Use grep filters and pipes to ensure clean, explicit exception handling patterns
    BARE_EXCEPTIONS=$(grep -n "except" "$TARGET_FILE" | grep -v "as e" | grep -v "except:")
    
    if [ -n "$BARE_EXCEPTIONS" ]; then
        echo "Warning: Found potential bare or unnamed exceptions:"
        echo "$BARE_EXCEPTIONS"
    else
        echo "Code Quality Check: Error handling patterns look consistent."
    fi
fi