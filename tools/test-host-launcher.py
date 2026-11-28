#!/usr/bin/env python3
"""Exercise the development shebang without a GPU, network or native build."""
import errno
import os
from pathlib import Path
import pty
import select
import signal
import sqlite3
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class HostLauncherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='cusco host ')
        self.addCleanup(temp.cleanup)
        self.data = Path(temp.name)
        self.service = self.data / 'test service'
        self.env = dict(os.environ, CUSCO_DATA_DIR=str(self.data), CUSCO_SERVICE_BIN=str(self.service))
        self.config = self.data / 'config.yaml'
        self.config.write_bytes((ROOT / 'config.host.example.yaml').read_bytes())

    def test_wrong_directory_rejected_before_data_lookup(self):
        env = dict(self.env, CUSCO_DATA_DIR=str(self.data / 'missing'))
        result = subprocess.run([ROOT / 'run-host.trn'], cwd=self.data, env=env, capture_output=True, timeout=180)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'Run this launcher from the Cusco repository root', result.stderr)

    def test_container_catalog_refused_before_start(self):
        db = self.data / 'db/catalog.sqlite'
        db.parent.mkdir()
        with sqlite3.connect(db) as conn:
            conn.execute('CREATE TABLE models(path TEXT)')
            conn.execute("INSERT INTO models VALUES('/data/models/model.gguf')")
        before = db.read_bytes()
        result = subprocess.run(['./run-host.trn'], cwd=ROOT, env=self.env, capture_output=True, timeout=180)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'/data', result.stderr)
        self.assertIn(b'catalog', result.stderr)
        self.assertEqual(db.read_bytes(), before)

    def test_terminal_interrupt_waits_for_service(self):
        self.service.write_text('''#!/usr/bin/env python3
import os, signal, sys, time
from pathlib import Path
data = Path(os.environ['CUSCO_DATA_DIR'])
assert Path.cwd() != data
assert sys.argv[1:] == ['--config', str(data / 'config.yaml'), '--http-debug', 'safe']
assert not sys.stdin.buffer.read(1)
def drain(*_):
    time.sleep(.25)
    (data / 'drained').touch()
    sys.stderr.write('drain complete\\n')
    sys.exit(23)
signal.signal(signal.SIGTERM, drain)
print('ready for interrupt', flush=True)
signal.pause()
''')
        self.service.chmod(0o755)
        pid, fd = pty.fork()
        if pid == 0:
            os.chdir(ROOT)
            os.execve('./run-host.trn', ['./run-host.trn', '--http-debug', 'safe'], self.env)
        output = bytearray()
        interrupted = reaped = False
        deadline = time.monotonic() + 180
        try:
            while time.monotonic() < deadline:
                if select.select([fd], [], [], .2)[0]:
                    try:
                        part = os.read(fd, 8192)
                    except OSError as error:
                        if error.errno != errno.EIO:
                            raise
                        part = b''
                    if not part:
                        break
                    output.extend(part)
                    if not interrupted and b'ready for interrupt' in output:
                        os.write(fd, b'\x03')
                        interrupted = True
            else:
                self.fail(f'timed out: {output[-2000:]!r}')
            _, status = os.waitpid(pid, 0)
            reaped = True
            self.assertEqual(os.waitstatus_to_exitcode(status), 23, output[-2000:])
            self.assertTrue((self.data / 'drained').is_file())
            self.assertIn(b'drain complete', output)
        finally:
            if not reaped:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            os.close(fd)


if __name__ == '__main__':
    unittest.main()
