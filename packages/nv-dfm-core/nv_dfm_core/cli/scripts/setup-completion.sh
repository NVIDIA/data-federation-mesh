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

# Setup script for DFM CLI shell completion
# This script automatically installs bash and zsh completion for the DFM CLI tool
# Located in dfm/cli/scripts/ for easy access

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if dfm command is available
check_dfm_available() {
    if ! command -v dfm &> /dev/null; then
        print_error "DFM CLI tool not found in PATH. Please install DFM first."
        print_status "You can install DFM by running: pip install -e ."
        exit 1
    fi
    print_success "DFM CLI tool found"
}

# Function to detect shell
detect_shell() {
    if [ -n "$ZSH_VERSION" ]; then
        echo "zsh"
    elif [ -n "$BASH_VERSION" ]; then
        echo "bash"
    else
        # Try to detect from parent process
        parent_shell=$(ps -p $PPID -o comm= 2>/dev/null || echo "")
        case "$parent_shell" in
            *zsh*) echo "zsh" ;;
            *bash*) echo "bash" ;;
            *) echo "unknown" ;;
        esac
    fi
}

# Function to setup bash completion
setup_bash_completion() {
    local completion_file="$HOME/.dfm-complete.bash"
    local bashrc_file="$HOME/.bashrc"
    
    print_status "Setting up bash completion..."
    
    # Generate completion script
    if _DFM_COMPLETE=bash_source dfm > "$completion_file" 2>/dev/null; then
        print_success "Generated bash completion script: $completion_file"
    else
        print_error "Failed to generate bash completion script"
        return 1
    fi
    
    # Add to bashrc if not already present
    if [ -f "$bashrc_file" ]; then
        if ! grep -q "source $completion_file" "$bashrc_file"; then
            echo "" >> "$bashrc_file"
            echo "# DFM CLI completion" >> "$bashrc_file"
            echo "source $completion_file" >> "$bashrc_file"
            print_success "Added completion to $bashrc_file"
        else
            print_warning "Completion already configured in $bashrc_file"
        fi
    else
        print_warning "$bashrc_file not found. Please manually add the following line:"
        echo "  source $completion_file"
    fi
}

# Function to setup zsh completion
setup_zsh_completion() {
    local completion_file="$HOME/.dfm-complete.zsh"
    local zshrc_file="$HOME/.zshrc"
    
    print_status "Setting up zsh completion..."
    
    # Generate completion script
    if _DFM_COMPLETE=zsh_source dfm > "$completion_file" 2>/dev/null; then
        print_success "Generated zsh completion script: $completion_file"
    else
        print_error "Failed to generate zsh completion script"
        return 1
    fi
    
    # Add to zshrc if not already present
    if [ -f "$zshrc_file" ]; then
        if ! grep -q "source $completion_file" "$zshrc_file"; then
            echo "" >> "$zshrc_file"
            echo "# DFM CLI completion" >> "$zshrc_file"
            echo "source $completion_file" >> "$zshrc_file"
            print_success "Added completion to $zshrc_file"
        else
            print_warning "Completion already configured in $zshrc_file"
        fi
    else
        print_warning "$zshrc_file not found. Please manually add the following line:"
        echo "  source $completion_file"
    fi
}

# Function to test completion
test_completion() {
    local shell_type="$1"
    print_status "Testing completion setup..."
    
    case "$shell_type" in
        "bash")
            if [ -f "$HOME/.dfm-complete.bash" ]; then
                print_success "Bash completion script is ready"
                print_status "To test, restart your shell or run: source ~/.bashrc"
                print_status "Then try: dfm <TAB> to see available commands"
            fi
            ;;
        "zsh")
            if [ -f "$HOME/.dfm-complete.zsh" ]; then
                print_success "Zsh completion script is ready"
                print_status "To test, restart your shell or run: source ~/.zshrc"
                print_status "Then try: dfm <TAB> to see available commands"
            fi
            ;;
    esac
}

# Main function
main() {
    print_status "DFM CLI Shell Completion Setup"
    print_status "================================"
    
    # Check if dfm is available
    check_dfm_available
    
    # Detect current shell
    current_shell=$(detect_shell)
    print_status "Detected shell: $current_shell"
    
    # Setup completion based on shell
    case "$current_shell" in
        "bash")
            setup_bash_completion
            test_completion "bash"
            ;;
        "zsh")
            setup_zsh_completion
            test_completion "zsh"
            ;;
        *)
            print_warning "Unknown shell: $current_shell"
            print_status "Setting up both bash and zsh completion..."
            setup_bash_completion
            setup_zsh_completion
            print_status "Please manually source the appropriate completion file for your shell"
            ;;
    esac
    
    print_success "Setup complete!"
    print_status ""
    print_status "Usage examples:"
    print_status "  dfm <TAB>                    # Show available commands"
    print_status "  dfm poc <TAB>               # Show poc subcommands"
    print_status "  dfm fed submit <TAB>        # Show submit options"
    print_status "  dfm --help                  # Show help"
}

# Run main function
main "$@"
