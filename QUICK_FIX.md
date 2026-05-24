# Quick Fix: Missing Submodule After Git Pull

## Problem
After running `git pull`, the `packages/streamrelay` folder is empty or missing.

## Solution (Copy & Paste)

```bash
# Fix missing submodules
git submodule update --init --recursive

# Verify it worked
ls -la packages/streamrelay/
```

## Expected Output
You should see files like:
- `__init__.py`
- `server.py`
- `reader.py`
- `protocol.py`
- etc.

## Why This Happens
Git submodules are separate repositories. When you clone or pull, you need to explicitly initialize them.

## Prevention
Next time, clone with submodules from the start:
```bash
git clone --recurse-submodules https://github.com/crazidev/VisoMaster.git
```

## Automated Fix Script
```bash
bash scripts/fix_submodules.sh
```

## Still Not Working?
1. Check internet connection
2. Verify git is installed: `git --version`
3. Try manual initialization:
   ```bash
   git submodule init
   git submodule update --recursive
   ```

## For RunPod Users
The automated setup script now handles this automatically:
```bash
bash scripts/runpod_quick_setup.sh
```
