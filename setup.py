#!/usr/bin/env python3
"""
Project Synapse MCP Server Setup Script

This script helps users set up Project Synapse with all required dependencies
and initial configuration.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True, capture_output=False):
    """Run a command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, capture_output=capture_output, text=True)
    return result


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    if sys.version_info < (3, 12):
        print("❌ Python 3.12+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True


def check_uv_installation():
    """Check if uv is installed."""
    print("Checking uv installation...")
    if not shutil.which('uv'):
        print("❌ uv is not installed. Please install it from: https://docs.astral.sh/uv/")
        return False
    print("✅ uv is installed")
    return True


def setup_virtual_environment():
    """Set up virtual environment with uv."""
    print("Setting up virtual environment...")
    try:
        # Create virtual environment
        run_command(['uv', 'venv', '--python', '3.12', '--seed'])

        # Install project dependencies (editable mode)
        run_command(['uv', 'pip', 'install', '-e', '.'])

        print("✅ Virtual environment created and dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to set up virtual environment: {e}")
        return False


def download_spacy_model():
    """Download required spaCy model."""
    print("Downloading spaCy model...")
    try:
        run_command(['uv', 'run', 'python', '-m', 'spacy', 'download', 'en_core_web_sm'])
        print("✅ spaCy model downloaded")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download spaCy model: {e}")
        return False


def setup_environment_file():
    """Set up environment configuration file."""
    print("Setting up environment configuration...")

    env_example = Path('.env.example')
    env_file = Path('.env')

    if env_example.exists() and not env_file.exists():
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file with your specific configuration")
        return True
    elif env_file.exists():
        print("✅ .env file already exists")
        return True
    else:
        print("❌ .env.example file not found")
        return False


def check_neo4j():
    """Check if Neo4j is available."""
    print("Checking Neo4j availability...")

    # Try to connect to default Neo4j instance
    try:
        import neo4j
        from dotenv import load_dotenv
        load_dotenv()
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "synapse_password")
        driver = neo4j.GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        print("✅ Neo4j is running and accessible")
        return True
    except Exception as e:
        print(f"⚠️  Neo4j connection failed: {e}")
        print("Please ensure Neo4j is installed and running with the correct credentials from your .env file")
        return False


def run_basic_tests():
    """Run basic tests to verify installation."""
    print("Running basic tests...")
    try:
        result = run_command(['uv', 'run', 'python', '-m', 'pytest', 'tests/test_basic.py', '-v'],
                           capture_output=True)
        if result.returncode == 0:
            print("✅ Basic tests passed")
            return True
        else:
            print(f"⚠️  Some tests failed:\n{result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run tests: {e}")
        return False


def create_example_config():
    """Create example MCP configuration for Claude Desktop."""
    print("Creating example MCP configuration...")

    config_content = {
        "mcpServers": {
            "project-synapse": {
                "command": "uv",
                "args": [
                    "--directory",
                    str(Path.cwd().absolute()),
                    "run",
                    "python",
                    "-m",
                    "synapse_mcp.server"
                ],
                "env": {
                    "NEO4J_URI": "bolt://localhost:7687",
                    "NEO4J_USER": "neo4j",
                    "NEO4J_PASSWORD": "synapse_password",
                    "NEO4J_DATABASE": "synapse",
                    "LOG_LEVEL": "INFO"
                }
            }
        }
    }

    # Write to current directory
    import json
    with open('claude_desktop_config.json', 'w') as f:
        json.dump(config_content, f, indent=2)

    print("✅ Created claude_desktop_config.json")
    print("Copy this configuration to your Claude Desktop config file")
    return True


def main():
    """Main setup function."""
    print("🧠 Project Synapse MCP Server Setup")
    print("=" * 40)

    # Check prerequisites
    if not check_python_version():
        sys.exit(1)

    if not check_uv_installation():
        sys.exit(1)

    # Setup steps
    success = True

    success &= setup_virtual_environment()
    success &= download_spacy_model()
    success &= setup_environment_file()

    # Optional checks
    check_neo4j()  # Don't fail on Neo4j issues
    run_basic_tests()  # Don't fail on test issues

    # Create configuration
    create_example_config()

    print("\n" + "=" * 40)
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file with your configuration")
        print("2. Start Neo4j database")
        print("3. Add claude_desktop_config.json to Claude Desktop")
        print("4. Test the server: uv run python -m synapse_mcp.server")
    else:
        print("⚠️  Setup completed with some issues")
        print("Please review the errors above and fix them manually")

    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
