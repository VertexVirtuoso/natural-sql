#!/bin/bash

# Natural SQL User-Local Installation Script
# This script installs natural-sql command in user's local bin directory

# Configuration
PROJECT_DIR="/home/charlie/git/natural_sql/natural-sql"
SCRIPT_NAME="natural-sql-launcher.sh"
USER_BIN_DIR="$HOME/.local/bin"
COMMAND_NAME="natural-sql"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
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

# Check if launcher script exists
if [[ ! -f "$PROJECT_DIR/$SCRIPT_NAME" ]]; then
    print_error "Launcher script not found: $PROJECT_DIR/$SCRIPT_NAME"
    exit 1
fi

print_info "Installing Natural SQL CLI for current user..."

# Create ~/.local/bin if it doesn't exist
if [[ ! -d "$USER_BIN_DIR" ]]; then
    print_info "Creating directory: $USER_BIN_DIR"
    mkdir -p "$USER_BIN_DIR"
fi

# Create symlink in ~/.local/bin
if ln -sf "$PROJECT_DIR/$SCRIPT_NAME" "$USER_BIN_DIR/$COMMAND_NAME"; then
    print_success "Created symlink: $USER_BIN_DIR/$COMMAND_NAME -> $PROJECT_DIR/$SCRIPT_NAME"
else
    print_error "Failed to create symlink"
    exit 1
fi

# Make sure the symlink is executable
chmod +x "$USER_BIN_DIR/$COMMAND_NAME"

# Check if ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$USER_BIN_DIR:"* ]]; then
    print_warning "~/.local/bin is not in your PATH"
    print_info "Add the following line to your ~/.bashrc or ~/.zshrc:"
    print_info "export PATH=\"\$HOME/.local/bin:\$PATH\""
    print_info ""
    print_info "Then run: source ~/.bashrc (or ~/.zshrc)"
    print_info "Or simply restart your terminal"
else
    print_success "~/.local/bin is already in your PATH"
fi

# Verify installation
if command -v "$COMMAND_NAME" >/dev/null 2>&1; then
    print_success "Installation completed successfully!"
    print_info "You can now run 'natural-sql' from anywhere on your system"
else
    print_warning "Command not immediately available (PATH issue)"
    print_info "Try running: export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

print_info ""
print_info "Examples:"
print_info "  natural-sql query 'How many users are there?'"
print_info "  natural-sql modify 'Add a new user named John'"
print_info "  natural-sql interactive"
print_info "  natural-sql test-connection"

print_info ""
print_warning "Note: Make sure your .env file is properly configured in:"
print_info "$PROJECT_DIR/.env"