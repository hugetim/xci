#!/bin/bash
# Fetch Epoch AI's Bayesian ECI code (eci-public) at the commit XCI v0.1 was checked against and install it.
# Run from the repo root inside the xci environment:  bash setup.sh
set -e
[ -d eci-public ] || git clone -q https://github.com/epoch-research/eci-public.git eci-public
git -C eci-public checkout -q e04929ab751531e558b1a61e86f436487c710ef8
pip install -q -e eci-public --no-deps
echo "eci-public at $(git -C eci-public rev-parse --short HEAD)"
