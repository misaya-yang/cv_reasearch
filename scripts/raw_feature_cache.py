"""Lossless, input-addressed raw DINO feature storage and encoder-free replay.

Features are unprojected model outputs, not masks, margins, graph fields or
mask-dependent prototypes. A cache entry records its exact available branches.
Missing coverage fails explicitly; this reader never initializes an encoder.
"""
import hashlib
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import tempfile

import numpy as np


class MissingBranches(KeyError):
    """Explicit lack of branch coverage, distinct from corrupt entry contents."""


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for part in iter(lambda:source.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def tensor_hash(value):
    x=np.ascontiguousarray(value)
    h=hashlib.sha256(json.dumps([list(x.shape),x.dtype.str]).encode())
    h.update(x.tobytes());return h.hexdigest()


class RawFeatureCache:
    def __init__(self, root, profile):
        self.root,self.profile=Path(root),profile
        self.profile_id=canonical_hash(profile)
        self.folder=self.root/self.profile_id
        self.folder.mkdir(parents=True,exist_ok=True)
        document=self.folder/'profile.json'
        if document.exists() and json.loads(document.read_text())!=profile:
            raise ValueError('Cache profile identity changed')
        if not document.exists():
            with tempfile.NamedTemporaryFile(dir=self.folder,mode='w',delete=False) as handle:
                handle.write(json.dumps(profile,indent=2)+'\n');temporary=Path(handle.name)
            temporary.replace(document)

    def key(self, model_input):
        return canonical_hash(dict(profile=self.profile_id,input_tensor_hash=tensor_hash(model_input)))

    def read(self, model_input, branches):
        with self._locked(model_input,False):
            return self._read(model_input,branches)

    @contextmanager
    def _locked(self, model_input, exclusive):
        folder=self.folder/self.key(model_input)
        if exclusive:folder.mkdir(parents=True,exist_ok=True)
        with (folder/'entry.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:yield
            finally:fcntl.flock(lock,fcntl.LOCK_UN)

    def _read(self, model_input, branches):
        key=self.key(model_input);folder=self.folder/key
        info=json.loads((folder/'entry.json').read_text())
        if info['profile_id']!=self.profile_id or info['input_tensor_hash']!=tensor_hash(model_input):
            raise ValueError('Raw cache input/model/preprocess identity differs')
        missing=set(branches)-set(info['features'])
        if missing:
            raise MissingBranches('Raw feature coverage missing: '+str(sorted(missing)))
        if info['file']!=info['file_sha256']+'.npz' or Path(info['file']).name!=info['file']:
            raise ValueError('Raw feature payload name differs from its content hash')
        payload=folder/info['file']
        if file_hash(payload)!=info['file_sha256']:
            raise ValueError('Raw feature payload changed')
        result={}
        with np.load(payload,allow_pickle=False) as source:
            for branch in branches:
                x=source[branch].copy();receipt=info['features'][branch]
                if list(x.shape)!=receipt['shape'] or str(x.dtype)!=receipt['dtype'] or tensor_hash(x)!=receipt['tensor_sha256']:
                    raise ValueError('Raw feature array failed readback identity')
                result[branch]=x
        return result

    def write(self, model_input, features, provenance):
        with self._locked(model_input,True):
            return self._write(model_input,features,provenance)

    def _write(self, model_input, features, provenance):
        key=self.key(model_input);folder=self.folder/key;folder.mkdir(parents=True,exist_ok=True)
        arrays={name:np.ascontiguousarray(value) for name,value in features.items()}
        if not arrays or any(x.dtype!=np.float32 or not np.isfinite(x).all() for x in arrays.values()):
            raise ValueError('Raw cache requires finite unmodified FP32 model outputs')
        old=folder/'entry.json'
        history=[];previous_payload=None
        if old.exists():
            info=json.loads(old.read_text());previous=self._read(model_input,list(info['features']))
            previous_payload=folder/info['file']
            for name in set(previous)&set(arrays):
                if not np.array_equal(previous[name],arrays[name]):
                    raise ValueError('Same input/profile produced different raw features')
            arrays={**previous,**arrays};history=info['provenance']
        handle=tempfile.NamedTemporaryFile(dir=folder,suffix='.npz',delete=False);temporary=Path(handle.name)
        try:
            with handle:
                np.savez_compressed(handle,**arrays)
            digest=file_hash(temporary);payload=folder/(digest+'.npz')
            temporary.replace(payload)
            info=dict(profile_id=self.profile_id,input_tensor_hash=tensor_hash(model_input),
                      file=payload.name,file_sha256=digest,storage='lossless NPZ ZIP, float32',
                      features={name:dict(shape=list(x.shape),dtype=str(x.dtype),tensor_sha256=tensor_hash(x)) for name,x in arrays.items()},
                      provenance=history+[provenance])
            pending=folder/'entry.tmp';pending.write_text(json.dumps(info,indent=2)+'\n');pending.replace(old)
            loaded=self._read(model_input,list(arrays))
            if any(not np.array_equal(arrays[k],loaded[k]) for k in arrays):
                raise ValueError('Raw cache write/readback differs')
            if previous_payload is not None and previous_payload!=payload:
                previous_payload.unlink()
            return key,info
        finally:
            temporary.unlink(missing_ok=True)
