#!/bin/bash

# Natural SQL System-Wide Installation Script for Arch Linux
# This script installs natural-sql command system-wide

# Configuration
PROJECT_DIR="/home/charlie/git/natural_sql/natural-sql"
SCRIPT_NAME="natural-sql-launcher.sh"
INSTALL_DIR="/usr/local/bin"
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

# Check if running with sufficient privileges
if [[ $EUID -ne 0 ]]; then
    print_error "This script must be run as root (use sudo)"
    exit 1
fi

# Check if launcher script exists
if [[ ! -f "$PROJECT_DIR/$SCRIPT_NAME" ]]; then
    print_error "Launcher script not found: $PROJECT_DIR/$SCRIPT_NAME"
    exit 1
fi

print_info "Installing Natural SQL CLI system-wide..."

# Create symlink in /usr/local/bin
if ln -sf "$PROJECT_DIR/$SCRIPT_NAME" "$INSTALL_DIR/$COMMAND_NAME"; then
    print_success "Created symlink: $INSTALL_DIR/$COMMAND_NAME -> $PROJECT_DIR/$SCRIPT_NAME"
else
    print_error "Failed to create symlink"
    exit 1
fi

# Make sure the symlink is executable
chmod +x "$INSTALL_DIR/$COMMAND_NAME"

# Verify installation
if command -v "$COMMAND_NAME" >/dev/null 2>&1; then
    print_success "Installation completed successfully!"
    print_info "You can now run 'natural-sql' from anywhere on your system"
    print_info ""
    print_info "Examples:"
    print_info "  natural-sql query 'How many users are there?'"
    print_info "  natural-sql modify 'Add a new user named John'"
    print_info "  natural-sql interactive"
    print_info "  natural-sql test-connection"
else
    print_error "Installation verification failed"
    exit 1
fi

print_info ""
print_warning "Note: Make sure your .env file is properly configured in:"
print_info "$PROJECT_DIR/.env"