#!/usr/bin/env python3
"""Raw-vector cache identity, deduplication, coverage and corruption checks."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event
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
        assert len(list((cache.folder/key).glob('*.npz')))==1
        ready,release=Event(),Event()
        def held_reader():
            with cache._locked(inputs,False):
                value=cache._read(inputs,['O/24']);ready.set()
                if not release.wait(5):raise AssertionError('Reader release timed out')
                return value
        with ThreadPoolExecutor(max_workers=2) as pool:
            reader=pool.submit(held_reader)
            assert ready.wait(5)
            writer=pool.submit(cache.write,inputs,{'K/24':raw+4},dict(role='while-reading'))
            from concurrent.futures import TimeoutError
            try:
                writer.result(timeout=.05);raise AssertionError('Writer replaced an active reader payload')
            except TimeoutError:pass
            finally:release.set()
            assert np.array_equal(reader.result()['O/24'],raw)
            writer.result()
        assert np.array_equal(cache.read(inputs,['K/24'])['K/24'],raw+4)
        assert len(list((cache.folder/key).glob('*.npz')))==1
        with ThreadPoolExecutor(max_workers=2) as pool:
            updates=[pool.submit(cache.write,inputs,{name:raw+offset},dict(role='parallel'))
                     for name,offset in (('K/16',2),('Q/24',3))]
            for update in updates:update.result()
        merged=cache.read(inputs,['O/24','Q/16','K/16','Q/24'])
        assert all(np.array_equal(merged[name],raw+offset) for name,offset in (('O/24',0),('Q/16',1),('K/16',2),('Q/24',3)))
        assert len(list((cache.folder/key).glob('*.npz')))==1
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
    print('PASS: exact raw readback, lossless concurrent branch extension, obsolete payload reclamation, coverage/identity/corruption rejection and no encoder fallback')


if __name__=='__main__':main()
