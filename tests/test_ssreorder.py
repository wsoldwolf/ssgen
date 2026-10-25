import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import ssreorder

try:
    from PIL import Image
except ImportError:
    Image = None


class ReorderTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / 'generated'
        scratch.mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def file(self, name, value, timestamp):
        path = self.root / name
        path.write_bytes(value)
        os.utime(path, ns=(timestamp, timestamp))
        return path

    def test_time_order_numbering_cycles_and_captions(self):
        old = self.file('0001.png', b'old', 1_000_000_000)
        new = self.file('0000.png', b'new', 3_000_000_000)
        self.file('middle.PNG', b'middle', 2_000_000_000)
        old.with_suffix('.txt').write_text('old caption', encoding='utf-8')
        new.with_suffix('.txt').write_text('new caption', encoding='utf-8')
        (self.root / 'notes.md').write_text('keep', encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ssreorder.reorder_pngs(self.root), 3)
        self.assertEqual([(self.root / f'{index:04d}.png').read_bytes() for index in range(3)],
                         [b'old', b'middle', b'new'])
        self.assertEqual((self.root / '0000.txt').read_text(encoding='utf-8'), 'old caption')
        self.assertEqual((self.root / '0002.txt').read_text(encoding='utf-8'), 'new caption')
        self.assertEqual((self.root / '0001.png').stat().st_mtime_ns, 2_000_000_000)
        self.assertTrue((self.root / 'notes.md').is_file())
        with contextlib.redirect_stdout(io.StringIO()):
            ssreorder.reorder_pngs(self.root)
        self.assertEqual((self.root / '0002.png').read_bytes(), b'new')

    def test_orphan_caption_conflict_preserves_files(self):
        self.file('photo.png', b'image', 1_000_000_000)
        self.file('0000.txt', b'orphan', 1_000_000_000)
        with self.assertRaises(FileExistsError):
            ssreorder.reorder_pngs(self.root)
        self.assertEqual((self.root / 'photo.png').read_bytes(), b'image')
        self.assertEqual((self.root / '0000.txt').read_bytes(), b'orphan')

    def test_rename_failure_restores_names(self):
        self.file('first.png', b'first', 1_000_000_000)
        self.file('second.png', b'second', 2_000_000_000)
        move = ssreorder.move_no_replace
        count = 0

        def fail_once(source, destination):
            nonlocal count
            count += 1
            if count == 4:
                raise PermissionError('simulated lock')
            return move(source, destination)

        with patch.object(ssreorder, 'move_no_replace', side_effect=fail_once), self.assertRaises(PermissionError):
            ssreorder.reorder_pngs(self.root)
        self.assertEqual({p.name: p.read_bytes() for p in self.root.iterdir()},
                         {'first.png': b'first', 'second.png': b'second'})

    def test_png_only_needs_no_pillow_and_skips_subdirectories(self):
        self.file('image.png', b'image', 1_000_000_000)
        nested = self.root / 'nested'
        nested.mkdir()
        (nested / 'keep.jpg').write_bytes(b'keep')
        with patch.object(sys, 'argv', ['ssreorder', str(self.root)]), \
                patch.object(ssreorder, 'pillow') as pillow, contextlib.redirect_stdout(io.StringIO()):
            ssreorder.main()
        pillow.assert_not_called()
        self.assertEqual((self.root / '0000.png').read_bytes(), b'image')
        self.assertEqual((nested / 'keep.jpg').read_bytes(), b'keep')


@unittest.skipIf(Image is None, 'Install ssgen[image] to run image conversion tests')
class ConversionTests(unittest.TestCase):
    setUp = ReorderTests.setUp

    def test_jpeg_webp_exif_and_original_deletion(self):
        jpeg = self.root / 'photo.JPG'
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (3, 5), 'red').save(jpeg, format='JPEG', exif=exif)
        os.utime(jpeg, ns=(2_000_000_000, 2_000_000_000))
        (self.root / 'photo.txt').write_text('caption', encoding='utf-8')
        webp = self.root / 'alpha.webp'
        Image.new('RGBA', (4, 4), (10, 20, 30, 100)).save(webp, format='WEBP', lossless=True)
        os.utime(webp, ns=(1_000_000_000, 1_000_000_000))
        with patch.object(sys, 'argv', ['ssreorder', str(self.root)]), contextlib.redirect_stdout(io.StringIO()):
            ssreorder.main()
        self.assertFalse(jpeg.exists())
        self.assertFalse(webp.exists())
        with Image.open(self.root / '0000.png') as image:
            self.assertEqual(image.getpixel((0, 0)), (10, 20, 30, 100))
        with Image.open(self.root / '0001.png') as image:
            self.assertEqual(image.size, (5, 3))
            self.assertNotEqual(image.getexif().get(274), 6)
        self.assertEqual((self.root / '0001.txt').read_text(encoding='utf-8'), 'caption')
        self.assertEqual((self.root / '0000.png').stat().st_mtime_ns, 1_000_000_000)

    def test_existing_png_collision_keeps_both_images_and_captions(self):
        original = self.root / 'same.png'
        Image.new('RGB', (2, 2), 'red').save(original)
        before = original.read_bytes()
        source = self.root / 'same.jpeg'
        Image.new('RGB', (2, 2), 'blue').save(source)
        (self.root / 'same.txt').write_text('shared caption', encoding='utf-8')
        destination = ssreorder.convert_to_png(source)
        self.assertEqual(destination.name, 'same_1.png')
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual(destination.with_suffix('.txt').read_text(encoding='utf-8'), 'shared caption')
        self.assertFalse(source.exists())

    def test_corrupt_source_not_deleted_and_no_reorder(self):
        corrupt = self.root / 'broken.jpg'
        corrupt.write_bytes(b'broken jpeg')
        Image.new('RGB', (2, 2)).save(self.root / 'existing.png')
        with patch.object(sys, 'argv', ['ssreorder', str(self.root)]), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                ssreorder.main()
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(corrupt.read_bytes(), b'broken jpeg')
        self.assertTrue((self.root / 'existing.png').is_file())
        self.assertFalse(list(self.root.glob('.ssreorder-*')))

    def test_animated_webp_preserves_frames(self):
        source = self.root / 'animation.webp'
        Image.new('RGBA', (2, 2), 'red').save(source, format='WEBP', save_all=True,
            append_images=[Image.new('RGBA', (2, 2), 'blue')], duration=[100, 200], loop=2, lossless=True)
        destination = ssreorder.convert_to_png(source)
        with Image.open(destination) as image:
            self.assertEqual(image.n_frames, 2)
            self.assertEqual(image.info['loop'], 2)
            image.seek(1)
            self.assertEqual(image.getpixel((0, 0))[:3], (0, 0, 255))
        self.assertFalse(source.exists())


if __name__ == '__main__':
    unittest.main()
