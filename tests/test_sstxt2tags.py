import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import sstxt2tags


class RenameTests(unittest.TestCase):
    def test_image_pair_required_and_knowledge_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / 'sub'
            nested.mkdir()
            for name in ('a', 'notes', '_knowledge', 'a_knowledge', 'existing', 'otherdir'):
                (root / (name + '.txt')).write_text('original', encoding='utf-8')
            for name in ('a.PNG', '_knowledge.png', 'a_knowledge.jpg', 'existing.webp'):
                (root / name).touch()
            (nested / 'otherdir.png').touch()
            (nested / 'nested.JPG').touch()
            (nested / 'nested.TXT').write_text('nested', encoding='utf-8')
            (root / 'existing.tags').write_text('keep', encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sstxt2tags.main([str(root), '--dry_run']), 0)
                self.assertTrue((root / 'a.txt').exists())
                self.assertFalse((root / 'a.tags').exists())
                self.assertEqual(sstxt2tags.main([str(root), str(nested)]), 0)
            self.assertEqual((root / 'a.tags').read_text(encoding='utf-8'), 'original')
            self.assertEqual((nested / 'nested.tags').read_text(encoding='utf-8'), 'nested')
            for name in ('notes', '_knowledge', 'a_knowledge', 'existing', 'otherdir'):
                self.assertTrue((root / (name + '.txt')).exists(), name)
            self.assertEqual((root / 'existing.tags').read_text(encoding='utf-8'), 'keep')


if __name__ == '__main__':
    unittest.main()
