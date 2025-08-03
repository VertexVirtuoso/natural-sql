# Natural SQL - System Installation Guide

This guide explains how to install Natural SQL CLI system-wide so you can run it from anywhere on your Arch Linux system.

## Quick Installation

### Option 1: User-Local Installation (Recommended)

Install for the current user only (no sudo required):

```bash
./install-user-local.sh
```

This installs the command to `~/.local/bin/natural-sql` and is immediately available if `~/.local/bin` is in your PATH.

### Option 2: System-Wide Installation

Install for all users (requires sudo):

```bash
sudo ./install-system-wide.sh
```

This installs the command to `/usr/local/bin/natural-sql` and is available system-wide.

## Manual Installation

If you prefer to install manually:

1. **Make the launcher executable:**
   ```bash
   chmod +x natural-sql-launcher.sh
   ```

2. **Create a symlink in your preferred location:**

   For user-local installation:
   ```bash
   mkdir -p ~/.local/bin
   ln -sf "$(pwd)/natural-sql-launcher.sh" ~/.local/bin/natural-sql
   ```

   For system-wide installation:
   ```bash
   sudo ln -sf "$(pwd)/natural-sql-launcher.sh" /usr/local/bin/natural-sql
   ```

3. **Ensure the directory is in your PATH:**

   For `~/.local/bin`, add to your `~/.bashrc` or `~/.zshrc`:
   ```bash
   export PATH="$HOME/.local/bin:$PATH"
   ```

## Verification

After installation, test that it works:

```bash
# Test from any directory
natural-sql --help

# Test database connection
natural-sql test-connection

# Test a simple query
natural-sql query "How many kpop groups are there?"
```

## Usage Examples

Once installed system-wide, you can use Natural SQL from anywhere:

### Read-Only Queries
```bash
# Simple queries
natural-sql query "Show me all users"
natural-sql query "How many orders were placed last month?"

# With different output formats
natural-sql query --format json "List all products"
natural-sql query --format csv "Show user emails"

# With explanations
natural-sql query --explain "What are the top 5 products by sales?"
```

### Data Modifications (Admin Mode)
```bash
# INSERT operations
natural-sql modify "Add a new user named John"

# UPDATE operations  
natural-sql modify "Update user age to 25 where name is John"

# DELETE operations (shows preview before deletion)
natural-sql modify "Delete users older than 65"

# Dry run mode (preview only)
natural-sql modify --dry-run "Delete all test data"
```

### Interactive Mode
```bash
# Start interactive session
natural-sql interactive

# In interactive mode, use these commands:
# /help     - Show help
# /schema   - Show database schema  
# /tables   - List all tables
# /modify   - Execute modifications
# /history  - Show query history
# /quit     - Exit
```

## Configuration

Make sure your `.env` file is properly configured with:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=3306
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password

# OpenRouter Configuration (Recommended)
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=qwen/qwen3-coder:free
```

## Troubleshooting

### Command Not Found

If you get "command not found" after installation:

1. **Check if the symlink exists:**
   ```bash
   ls -la ~/.local/bin/natural-sql
   # or
   ls -la /usr/local/bin/natural-sql
   ```

2. **Check your PATH:**
   ```bash
   echo $PATH
   ```

3. **Reload your shell configuration:**
   ```bash
   source ~/.bashrc  # or ~/.zshrc
   ```

### Permission Issues

If you get permission errors:

1. **Make sure the launcher script is executable:**
   ```bash
   chmod +x natural-sql-launcher.sh
   ```

2. **Check ownership of the symlink:**
   ```bash
   ls -la ~/.local/bin/natural-sql
   ```

### Database Connection Issues

1. **Test the database connection:**
   ```bash
   natural-sql test-connection
   ```

2. **Check your `.env` file:**
   ```bash
   cat .env
   ```

3. **Verify database credentials and accessibility**

## Uninstallation

To remove the system-wide installation:

```bash
# For user-local installation
rm ~/.local/bin/natural-sql

# For system-wide installation
sudo rm /usr/local/bin/natural-sql
```

## Unicode Character Support

Natural SQL fully supports Japanese, Korean, Chinese, and other Unicode characters:

- ✅ Database properly configured with `utf8mb4` character set
- ✅ Connection uses `utf8mb4_unicode_ci` collation  
- ✅ All text fields support full Unicode range
- ✅ CLI preserves Unicode characters in queries and results

Example:
```bash
natural-sql modify "Add idol with name 'Honochi' and native name 'ほのち'"
```

## Features

- 🧠 **Natural Language Processing**: Uses OpenRouter + Qwen3 Coder model
- 🛡️ **Security**: Safe by default, admin mode for modifications
- 🔍 **Preview Mode**: Shows what DELETE/UPDATE operations would affect
- 📊 **Multiple Formats**: Table, JSON, CSV output
- 🌐 **Unicode Support**: Full Japanese/Korean character support
- 💬 **Interactive Mode**: Conversational database querying
- 🚀 **System-Wide Access**: Run from anywhere on your system

---

For more information, see the main [README.md](README.md) file.