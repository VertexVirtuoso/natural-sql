#!/bin/bash

# Natural SQL CLI Launcher Script
# This script allows you to run natural-sql from anywhere on your system

# Configuration
PROJECT_DIR="/home/charlie/git/natural_sql/natural-sql"
UV_PATH="/home/charlie/.local/bin/uv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if project directory exists
if [[ ! -d "$PROJECT_DIR" ]]; then
    print_error "Project directory not found: $PROJECT_DIR"
    print_info "Please update the PROJECT_DIR variable in this script"
    exit 1
fi

# Check if uv is installed
if [[ ! -x "$UV_PATH" ]]; then
    print_error "uv not found at: $UV_PATH"
    print_info "Please install uv or update the UV_PATH variable in this script"
    exit 1
fi

# Change to project directory
cd "$PROJECT_DIR" || {
    print_error "Failed to navigate to project directory"
    exit 1
}

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    print_warning ".env file not found in project directory"
    print_info "You may need to configure your database and API settings"
    print_info "Copy .env.example to .env and edit with your settings"
fi

# Run the natural-sql command with all passed arguments
print_info "Running natural-sql from: $PROJECT_DIR"
exec "$UV_PATH" run natural-sql "$@"