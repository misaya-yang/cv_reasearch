#!/usr/bin/env python3
"""A failed C1 prerequisite must not read samples or construct a model."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import run_c1_candidate as runner


def main():
    for passed in (False,None,'false',1):
        assert runner.qualified_choice(dict(candidates={'C1':dict(passed=passed)})) is None
    assert runner.qualified_choice(dict(candidates={'C1':dict(passed=True,representation='Q/16/deb')})) == 'Q/16/deb'
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);decision=root/'decision.json';decision.write_text(json.dumps(dict(candidates={'C1':dict(passed=False)})))
        args=['C1','--decision',str(decision),'--manifest','/missing/manifest.json',
              '--assets','/missing/assets','--out',str(root/'candidate'),'--split','dev']
        with patch.object(sys,'argv',args),patch.object(runner,'read_rows',side_effect=AssertionError('sample read')), \
                patch.object(runner,'build_model',side_effect=AssertionError('model constructed')), \
                contextlib.redirect_stdout(io.StringIO()):
            runner.main()
        assert not (root/'candidate').exists()
    print('PASS: failed C1 prerequisite performs no sample read, model construction or candidate run')


if __name__ == '__main__':
    main()
