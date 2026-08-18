#!/bin/bash
# PRIDE Metadata Extraction Framework - Pipeline Runner
#
# This script runs the full extraction pipeline:
# 1. Runs all agents (biological, technical, experimental)
# 2. Enables validation (The Critic)
# 3. Uses default input/output paths (unless overridden)
#
# First-run checks are performed to ensure dependencies are ready.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check for Python
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}Error: Python not found${NC}"
        echo "Please install Python 3.8+ and try again."
        exit 1
    fi
}

# Check for virtual environment
check_venv() {
    if [ -d "venv" ]; then
        source venv/bin/activate 2>/dev/null || true
    fi
}

# Check if dependencies are installed
check_dependencies() {
    $PYTHON_CMD -c "import openai; import transformers; import yaml; import faiss" 2>/dev/null
    return $?
}

# Check if ontologies exist (for --normalize mode)
check_ontologies() {
    if [ -d "ontologies" ] && [ "$(ls -A ontologies/*.obo 2>/dev/null)" ]; then
        return 0
    else
        return 1
    fi
}

# Main checks
check_python
check_venv

# Check if this is first run
if ! check_dependencies; then
    echo -e "${YELLOW}First-time setup detected!${NC}"
    echo ""
    echo "Dependencies are not installed. Please run setup first:"
    echo ""
    echo "    ./setup.sh"
    echo ""
    echo "Or install manually:"
    echo "    pip install -r requirements.txt"
    echo ""
    exit 1
fi

# Check for normalize flag and ontologies
if [[ "$*" == *"--normalize"* ]]; then
    if ! check_ontologies; then
        echo -e "${YELLOW}Warning: Ontologies not found${NC}"
        echo ""
        echo "The --normalize flag requires ontology files."
        echo "Download them with:"
        echo ""
        echo "    python -m normalization.download"
        echo ""
        echo "Or run setup:"
        echo "    ./setup.sh"
        echo ""
        exit 1
    fi
fi

# Run the pipeline
echo "Starting Unified Scientific Metadata Extraction Pipeline..."
echo "Mode: ALL"
echo "Validation: ENABLED"
echo ""

$PYTHON_CMD main.py all --validate "$@"

echo ""
echo "Pipeline Complete."
