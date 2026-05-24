#!/bin/bash
# Fix missing git submodules in VisoMaster
# Run this if packages/streamrelay is empty after git pull

set -e

echo "============================================"
echo "  VisoMaster Submodule Fix"
echo "============================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if we're in the VisoMaster directory
if [ ! -f "main.py" ]; then
    print_error "Not in VisoMaster directory!"
    echo "Please run this script from the VisoMaster root directory:"
    echo "  cd /workspace/VisoMaster"
    echo "  bash scripts/fix_submodules.sh"
    exit 1
fi

print_status "Found VisoMaster directory"

# Check if .gitmodules exists
if [ ! -f ".gitmodules" ]; then
    print_error "No .gitmodules file found!"
    echo "This repository doesn't have submodules configured."
    exit 1
fi

print_status "Found .gitmodules file"

# Check current submodule status
echo ""
echo "Current submodule status:"
git submodule status

# Initialize and update submodules
echo ""
echo "Initializing and updating submodules..."
git submodule update --init --recursive

if [ $? -eq 0 ]; then
    print_status "Submodules updated successfully"
else
    print_error "Failed to update submodules"
    echo ""
    echo "Troubleshooting steps:"
    echo "1. Check your internet connection"
    echo "2. Verify git is installed: git --version"
    echo "3. Try manually:"
    echo "   git submodule init"
    echo "   git submodule update --recursive"
    exit 1
fi

# Verify streamrelay submodule
echo ""
echo "Verifying streamrelay submodule..."
if [ -d "packages/streamrelay" ]; then
    FILE_COUNT=$(find packages/streamrelay -type f -name "*.py" | wc -l)
    if [ "$FILE_COUNT" -gt 0 ]; then
        print_status "streamrelay submodule is populated ($FILE_COUNT Python files found)"
        echo ""
        echo "Files in packages/streamrelay:"
        ls -lh packages/streamrelay/*.py 2>/dev/null || ls -lh packages/streamrelay/src/streamrelay/*.py 2>/dev/null || echo "  (files in subdirectories)"
    else
        print_warning "streamrelay directory exists but appears empty"
        echo "Contents:"
        ls -la packages/streamrelay/
    fi
else
    print_error "packages/streamrelay directory not found!"
    exit 1
fi

# Final status
echo ""
echo "============================================"
echo "  ✅ Submodule Fix Complete"
echo "============================================"
echo ""
echo "Submodule status:"
git submodule status
echo ""
print_status "All submodules are now initialized and updated"
echo ""
