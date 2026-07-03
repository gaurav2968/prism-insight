# PRISM-INSIGHT Docker Image
# Ubuntu 24.04 based AI Stock Analysis System (India / NSE)

FROM ubuntu:24.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Kolkata \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHON_VERSION=3.12 \
    ENABLE_CRON=true

# Working directory
WORKDIR /app

# System packages update and basic tools installation (cron included)
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    python3=${PYTHON_VERSION}* \
    python3-pip \
    python3-venv \
    python3-full \
    git \
    curl \
    wget \
    ca-certificates \
    gnupg \
    locales \
    tzdata \
    vim \
    nano \
    cron \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22.x LTS
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g npm@latest && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install UV (Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> /root/.bashrc

# Add UV to PATH
ENV PATH="/root/.cargo/bin:$PATH"

# Create Python virtual environment
RUN python3 -m venv /app/venv

# Activate virtual environment
ENV PATH="/app/venv/bin:$PATH"

# Clone Git repository
RUN git clone -b main https://github.com/dragon1086/prism-insight.git /app/prism-insight

# Change working directory
WORKDIR /app/prism-insight

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Install Playwright browser (Chromium only)
RUN playwright install --with-deps chromium

# Install Perplexity MCP server (official npm package)
RUN npm install -g @perplexity-ai/mcp-server

# Refresh font cache
RUN fc-cache -fv && \
    python3 -c "import matplotlib.font_manager as fm; fm.fontManager.rebuild()" || true

# Copy config files (example files)
RUN cp .env.example .env && \
    cp mcp_agent.config.yaml.example mcp_agent.config.yaml && \
    cp mcp_agent.secrets.yaml.example mcp_agent.secrets.yaml

# Create SQLite database directory
RUN mkdir -p /app/prism-insight/sqlite && \
    touch /app/prism-insight/stock_tracking_db.sqlite

# Create log and output directories
RUN mkdir -p /app/prism-insight/reports \
             /app/prism-insight/pdf_reports \
             /app/prism-insight/html_reports \
             /app/prism-insight/charts \
             /app/prism-insight/logs \
             /app/prism-insight/telegram_messages/sent \
             /app/prism-insight/prism-in/reports \
             /app/prism-insight/prism-in/pdf_reports

# Create Docker config directory
RUN mkdir -p /app/prism-insight/docker

# Copy Crontab and Entrypoint scripts
COPY docker/crontab /app/prism-insight/docker/crontab
COPY docker/entrypoint.sh /app/prism-insight/docker/entrypoint.sh

# Grant execute permission to entrypoint
RUN chmod +x /app/prism-insight/docker/entrypoint.sh && \
    chmod 644 /app/prism-insight/docker/crontab

# Set permissions
RUN chmod -R 755 /app/prism-insight

# Create cron log file
RUN touch /var/log/cron.log

# Health check - verify DB tables and cron status
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "\
import sqlite3; \
conn = sqlite3.connect('/app/prism-insight/stock_tracking_db.sqlite'); \
c = conn.cursor(); \
c.execute(\"SELECT COUNT(*) FROM sqlite_master WHERE type='table'\"); \
tables = c.fetchone()[0]; \
assert tables >= 3, f'Only {tables} tables found'; \
" && service cron status || exit 1

# Default shell
SHELL ["/bin/bash", "-c"]

# Entrypoint
ENTRYPOINT ["/app/prism-insight/docker/entrypoint.sh"]

# Default command (if none, entrypoint keeps container alive)
CMD []
