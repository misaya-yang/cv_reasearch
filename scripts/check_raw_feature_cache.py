#!/usr/bin/env python3
"""Raw-vector cache identity, deduplication, coverage and corruption checks."""
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

from raw_feature_cache import RawFeatureCache
from cached_dino import CachedDINO


def main():
    rng=np.random.RandomState(4);inputs=rng.randn(3,16,16).astype(np.float32)
    raw=rng.randn(32,16).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        cache=RawFeatureCache(tmp,dict(model='fixture',weights_sha256='frozen',preprocess='fixed',dtype='float32'))
        key,first=cache.write(inputs,{'O/24':raw},dict(role='reference',rgb_hash='same',crop=[0,0,16,16]))
        second_key,_=cache.write(inputs,{'O/24':raw.copy()},dict(role='query',rgb_hash='same',crop=[0,0,16,16]))
        assert key==second_key and np.array_equal(cache.read(inputs,['O/24'])['O/24'],raw)
        try:
            cache.read(inputs,['Q/16']);raise AssertionError('Missing raw branch silently accepted')
        except KeyError:pass
        cache.write(inputs,{'Q/16':raw+1},dict(role='query'))
        read=cache.read(inputs,['O/24','Q/16']);assert np.array_equal(read['O/24'],raw)
        other=RawFeatureCache(tmp,dict(model='fixture',weights_sha256='changed',preprocess='fixed',dtype='float32'))
        assert other.key(inputs)!=key
        changed=inputs.copy();changed[0,0,0]+=1;assert cache.key(changed)!=key
        try:
            cache.write(inputs,{'O/24':raw+1},dict(role='query'));raise AssertionError('Different raw vectors overwritten')
        except ValueError:pass
        entry=cache.folder/key/'entry.json'
        import json
        path=cache.folder/key/json.loads(entry.read_text())['file'];path.write_bytes(b'corrupt')
        try:
            cache.read(inputs,['O/24']);raise AssertionError('Corrupt payload accepted')
        except ValueError:pass
        replay=CachedDINO(cache)
        try:
            replay.raw(torch.zeros(1,3,1024,1024),('O/24',))
            raise AssertionError('Missing input silently fell back to encoding')
        except KeyError:pass
        assert replay.encoder is None and replay.encoder_calls==0
    print('PASS: exact raw FP32 readback, image-role dedup, coverage/identity/corruption rejection and encoder-free missing-input failure')


if __name__=='__main__':main()
