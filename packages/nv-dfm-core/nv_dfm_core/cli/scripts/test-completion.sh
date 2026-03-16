#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Test script for DFM CLI shell completion
# This script demonstrates how the completion works
# Located in dfm/cli/scripts/ for easy access

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_info "DFM CLI Shell Completion Test"
print_info "============================="

# Check if dfm is available
if ! command -v dfm &> /dev/null; then
    print_warning "DFM CLI tool not found in PATH. Please install DFM first."
    exit 1
fi

print_success "DFM CLI tool found: $(which dfm)"

# Test completion script generation
print_info "Testing completion script generation..."

# Test bash completion (if bash version supports it)
if bash --version | grep -q "version 4\." || bash --version | grep -q "version [5-9]"; then
    print_info "Testing bash completion generation..."
    if _DFM_COMPLETE=bash_source dfm > /tmp/dfm-test-bash.bash 2>/dev/null; then
        print_success "Bash completion script generated successfully"
        print_info "Bash completion script size: $(wc -l < /tmp/dfm-test-bash.bash) lines"
    else
        print_warning "Bash completion generation failed (may be due to bash version)"
    fi
else
    print_warning "Bash version too old for completion support (requires 4.4+)"
fi

# Test zsh completion
print_info "Testing zsh completion generation..."
if _DFM_COMPLETE=zsh_source dfm > /tmp/dfm-test-zsh.zsh 2>/dev/null; then
    print_success "Zsh completion script generated successfully"
    print_info "Zsh completion script size: $(wc -l < /tmp/dfm-test-zsh.zsh) lines"
else
    print_warning "Zsh completion generation failed"
fi

# Test fish completion
print_info "Testing fish completion generation..."
if _DFM_COMPLETE=fish_source dfm > /tmp/dfm-test-fish.fish 2>/dev/null; then
    print_success "Fish completion script generated successfully"
    print_info "Fish completion script size: $(wc -l < /tmp/dfm-test-fish.fish) lines"
else
    print_warning "Fish completion generation failed"
fi

# Test the completion command
print_info "Testing the built-in completion command..."
if dfm completion --help > /dev/null 2>&1; then
    print_success "Built-in completion command is available"
    
    # Test generating completion script via the command
    if dfm completion --shell zsh --output /tmp/dfm-test-command.zsh > /dev/null 2>&1; then
        print_success "Completion command generated script successfully"
    else
        print_warning "Completion command failed to generate script"
    fi
else
    print_warning "Built-in completion command not available"
fi

print_info ""
print_success "Completion test completed!"
print_info ""
print_info "To manually test completion:"
print_info "1. Generate completion script: _DFM_COMPLETE=zsh_source dfm > ~/.dfm-complete.zsh"
print_info "2. Source the script: source ~/.dfm-complete.zsh"
print_info "3. Test completion: dfm <TAB>"
print_info ""
print_info "Or use the setup script: ./scripts/setup-completion.sh"

# Clean up test files
rm -f /tmp/dfm-test-*.bash /tmp/dfm-test-*.zsh /tmp/dfm-test-*.fish
