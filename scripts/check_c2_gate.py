#!/usr/bin/env python3
"""Check C2 qualification blocks any reference/model work when it fails."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import check_c2_self_reference as runner


def main():
    for value in (False, None, 'false', 1):
        assert runner.qualified_choice(dict(candidates={'C2':dict(passed=value)})) is None
    assert runner.qualified_choice(dict(candidates={'C2':dict(passed=True,representation='C2concat/deb')})) == 'C2concat/deb'
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);decision=root/'decision.json';decision.write_text(json.dumps(dict(candidates={'C2':dict(passed=False)})))
        args=['check','--decision',str(decision),'--manifest','/missing/manifest.json',
              '--basis','/missing/basis','--out',str(root/'run'),'--assets','/missing/assets']
        with patch.object(sys,'argv',args), patch.object(runner,'build_host',side_effect=AssertionError('model initialized')), \
                patch.object(runner,'evaluate_reference',side_effect=AssertionError('reference opened')), \
                contextlib.redirect_stdout(io.StringIO()):
            runner.main()
        assert not (root/'run').exists()
    print('PASS: failed/absent C2 qualification performs no reference encoding, label read, model construction or new run')


if __name__ == '__main__':
    main()
