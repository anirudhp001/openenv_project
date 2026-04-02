#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import urllib.request
import urllib.error
import urllib.parse
import json

def print_step(title):
    print(f"\n{'='*40}")
    print(f"\033[1m{title}\033[0m")
    print(f"{'='*40}")

def print_pass(msg):
    print(f"\033[92m[PASSED]\033[0m {msg}")

def print_fail(msg, hint=None):
    print(f"\033[91m[FAILED]\033[0m {msg}")
    if hint:
        print(f"\033[93m  Hint:\033[0m {hint}")

def main():
    parser = argparse.ArgumentParser(description="Cross-platform Submission Validator")
    parser.add_argument("ping_url", help="Your HuggingFace Space URL (e.g. https://your-space.hf.space)")
    parser.add_argument("--repo_dir", default=".", help="Path to your repo (default: current directory)")
    
    args = parser.parse_args()
    
    ping_url = args.ping_url.rstrip("/")
    repo_dir = os.path.abspath(args.repo_dir)
    
    if not os.path.isdir(repo_dir):
        print_fail(f"Directory '{repo_dir}' not found.")
        sys.exit(1)

    # Step 1: Ping HF Space
    print_step("Step 1/3: Pinging HF Space (/reset) ...")
    reset_url = f"{ping_url}/reset"
    
    try:
        req = urllib.request.Request(reset_url, data=b"{}", headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status == 200:
                print_pass("HF Space is live and responds to /reset")
            else:
                print_fail(f"HF Space /reset returned HTTP {response.status} (expected 200)", 
                           "Make sure your Space is running and the URL is correct.")
                sys.exit(1)
    except urllib.error.HTTPError as e:
        print_fail(f"HF Space /reset returned HTTP {e.code} (expected 200)", 
                   "View the URL in your browser or check logs.")
        sys.exit(1)
    except Exception as e:
        print_fail("HF Space not reachable (connection failed or timed out)", 
                   "Check your network connection and verify Space is running.")
        sys.exit(1)

    # Step 2: Docker Build
    print_step("Step 2/3: Running docker build ...")
    docker_context = repo_dir
    if not os.path.isfile(os.path.join(docker_context, "Dockerfile")):
        alt_context = os.path.join(repo_dir, "server")
        if os.path.isfile(os.path.join(alt_context, "Dockerfile")):
            docker_context = alt_context
        else:
            print_fail("No Dockerfile found in repo root or server/ directory")
            sys.exit(1)
            
    print(f"Found Dockerfile in {docker_context}. Building...")
    
    try:
        # Build image natively
        res = subprocess.run(["docker", "build", docker_context], capture_output=True, text=True, timeout=600)
        if res.returncode == 0:
            print_pass("Docker build succeeded")
        else:
            print_fail("Docker build failed")
            print(res.stderr[-1000:])
            sys.exit(1)
    except FileNotFoundError:
        print_fail("docker command not found", "Install Docker: https://docs.docker.com/get-docker/")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print_fail("Docker build timed out after 600s")
        sys.exit(1)

    # Step 3: OpenEnv Validate
    print_step("Step 3/3: Running openenv validate ...")
    try:
        res = subprocess.run(["openenv", "validate"], cwd=repo_dir, capture_output=True, text=True)
        if res.returncode == 0:
            print_pass("openenv validate passed")
            print("  " + "\n  ".join(res.stdout.splitlines()[:5])) # Optional output truncation
        else:
            print_fail("openenv validate failed")
            print(res.stdout)
            print(res.stderr)
            sys.exit(1)
    except FileNotFoundError:
        print_fail("openenv command not found", "Install it: pip install openenv-core")
        sys.exit(1)

    print("\n\033[92m\033[1m========================================\n  All 3/3 checks passed!\n  Your submission is ready to submit.\n========================================\033[0m")

if __name__ == "__main__":
    main()
