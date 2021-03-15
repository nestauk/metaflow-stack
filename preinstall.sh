#!/bin/bash
set -euo pipefail

# Fetch research daps key
aws s3 cp s3://nesta-production-config/research_daps.key .

# Unencrypt research daps
git-crypt unlock research_daps.key &> /dev/null || true

# Clean up
rm research_daps.key

