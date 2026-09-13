# Sandbox execution image for cognitive-router exec_sandbox.ts
# Provides node 22 + python3, no network, minimal footprint
FROM node:22-slim

# Install python3 for Python probe support
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        python3 \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Verify both runtimes
RUN node --version && python3 --version

# Create non-root user for additional safety
RUN groupadd -r sandbox && useradd -r -g sandbox sandbox

# No entrypoint: the sandbox driver passes full `sh -c <script>` commands.
# (A leftover `ENTRYPOINT ["python3","--version"]` here silently made every
# probe run print the version banner and exit 0 — DO NOT restore it.)
ENTRYPOINT []
