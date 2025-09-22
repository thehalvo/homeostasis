#!/bin/bash

# Script to help users set up Docker for testing
# This ensures GitHub Actions tests can be replicated locally

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Docker Setup Helper ===${NC}"
echo -e "${BLUE}This script helps ensure your local environment matches GitHub Actions${NC}\n"

# Check OS
OS="$(uname -s)"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker is not installed${NC}"
    echo -e "${YELLOW}To install Docker Desktop:${NC}"

    case "$OS" in
        Darwin)
            echo -e "${BLUE}macOS Installation Options:${NC}"
            echo -e "1. Download from: https://docker.com/products/docker-desktop/"
            echo -e "2. Or use Homebrew:"
            echo -e "   ${GREEN}brew install --cask docker${NC}"
            ;;
        Linux)
            echo -e "${BLUE}Linux Installation:${NC}"
            echo -e "Visit: https://docs.docker.com/engine/install/"
            ;;
        *)
            echo -e "Visit: https://docker.com/products/docker-desktop/"
            ;;
    esac
    exit 1
fi

# Check if Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}✗ Docker is installed but not running${NC}"

    # Check for Colima
    if command -v colima &> /dev/null; then
        echo -e "${YELLOW}Detected Colima installation${NC}"

        # Check Colima status
        if ! colima status >/dev/null 2>&1; then
            echo -e "${YELLOW}Starting Colima...${NC}"
            colima start

            # Set up Docker context for Colima
            echo -e "${YELLOW}Configuring Docker to use Colima...${NC}"
            docker context create colima-default --docker "host=unix://${HOME}/.colima/default/docker.sock" 2>/dev/null || true
            docker context use colima-default
        else
            echo -e "${GREEN}Colima is running${NC}"
            # Ensure we're using the right context
            docker context use colima-default 2>/dev/null || true
        fi
    else
        case "$OS" in
            Darwin)
                # Check for Docker Desktop
                if [ -f "/Applications/Docker.app/Contents/MacOS/Docker" ]; then
                    echo -e "${YELLOW}Starting Docker Desktop...${NC}"
                    open -a Docker
                else
                    echo -e "${YELLOW}No Docker runtime found. You can use either:${NC}"
                    echo -e "1. Colima (lightweight): brew install colima"
                    echo -e "2. Docker Desktop: https://docker.com/products/docker-desktop/"
                    exit 1
                fi
                ;;
            Linux)
                echo -e "${YELLOW}Starting Docker service...${NC}"
                sudo systemctl start docker || sudo service docker start
                ;;
        esac

        # Wait for Docker to start
        echo -e "${YELLOW}Waiting for Docker to start...${NC}"
        for i in {1..60}; do
            if docker info >/dev/null 2>&1; then
                echo -e "\n${GREEN}✓ Docker is now running!${NC}"
                break
            fi
            printf "."
            sleep 1
        done

        if ! docker info >/dev/null 2>&1; then
            echo -e "\n${RED}Docker failed to start. Please start it manually.${NC}"
            exit 1
        fi
    fi
fi

# Check Docker Compose
if ! command -v docker compose &> /dev/null 2>&1; then
    echo -e "${RED}✗ Docker Compose is not available${NC}"
    echo -e "${YELLOW}Docker Compose should be included with Docker Desktop${NC}"
    echo -e "${YELLOW}Try: docker compose version${NC}"
    exit 1
fi

# Final verification
echo -e "\n${GREEN}=== Docker Status ===${NC}"
docker --version
docker compose version
echo ""
docker info | grep -E "(Server Version|Operating System|CPUs|Total Memory)" || true

echo -e "\n${GREEN}✓ Docker is ready for testing!${NC}"
echo -e "${BLUE}You can now run: git push origin main${NC}"