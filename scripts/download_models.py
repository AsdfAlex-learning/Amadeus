#!/usr/bin/env python3
"""Pre-download all model weights for offline use.

Downloads and caches all encoder weights that FullDuplexDiT needs:
  - Hubert (facebook/hubert-base-ls960) — ~378MB
  - MobileNetV3-Small (torchvision) — ~10MB
  - BERT-tiny (google/bert_uncased_L-2_H-128_A-2) — ~17MB

Run once before first use, or after clearing caches.
"""

import sys
import time


def download_hubert():
    print("[1/3] Downloading Hubert (facebook/hubert-base-ls960)...")
    t0 = time.time()
    try:
        from transformers import HubertModel
        HubertModel.from_pretrained("facebook/hubert-base-ls960")
        print(f"      ✓ Hubert cached ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"      ✗ Failed: {e}")
        return False
    return True


def download_mobilenet():
    print("[2/3] Downloading MobileNetV3-Small...")
    t0 = time.time()
    try:
        from torchvision.models import mobilenet_v3_small
        mobilenet_v3_small(weights="DEFAULT")
        print(f"      ✓ MobileNetV3 cached ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"      ✗ Failed: {e}")
        return False
    return True


def download_bert():
    print("[3/3] Downloading BERT-tiny (google/bert_uncased_L-2_H-128_A-2)...")
    t0 = time.time()
    try:
        from transformers import BertModel, BertTokenizer
        BertTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        print(f"      ✓ BERT-tiny cached ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"      ✗ Failed: {e}")
        return False
    return True


def verify_cache():
    print("\n[verify] Checking cached files...")
    import os
    cache_dirs = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/torch/hub/checkpoints"),
    ]
    for d in cache_dirs:
        if os.path.exists(d):
            size = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(d)
                for f in files
            )
            print(f"      {d}: {size / 1024 / 1024:.0f} MB")


def main():
    print("=" * 60)
    print("  Amadeus — Model Weight Downloader")
    print("=" * 60)
    print()

    results = [
        download_hubert(),
        download_mobilenet(),
        download_bert(),
    ]

    verify_cache()

    if all(results):
        print("\n✓ All models downloaded successfully.")
        print("  FullDuplexDiT will load from local cache with no network access.")
    else:
        failed = [["Hubert", "MobileNet", "BERT"][i] for i, ok in enumerate(results) if not ok]
        print(f"\n⚠ Some downloads failed: {', '.join(failed)}")
        print("  The model will use random initialization for these encoders.")
        sys.exit(1)


if __name__ == "__main__":
    main()
