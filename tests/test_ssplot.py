import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import ssplot

try:
    import matplotlib
    from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, types_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
except ImportError:
    EventFileWriter = None


class RollingMeanTests(unittest.TestCase):
    def test_trailing_window(self):
        self.assertEqual(ssplot.rolling_mean([(1, 2), (2, 4), (3, 9)], 2),
                         [(2, 3), (3, 6.5)])


@unittest.skipIf(EventFileWriter is None, "Install ssgen[plot] to run plot integration tests")
class PlotTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[1] / "generated"
        self.scratch.mkdir(exist_ok=True)

    def test_scalar_tensor_and_partial_tail(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as folder:
            writer = EventFileWriter(folder)
            writer.add_event(event_pb2.Event(wall_time=1, step=1, summary=summary_pb2.Summary(value=[
                summary_pb2.Summary.Value(tag="loss/current", simple_value=0.5),
                summary_pb2.Summary.Value(tag="loss/epoch_average", simple_value=0.5),
                summary_pb2.Summary.Value(tag="tensor_scalar", tensor=tensor_pb2.TensorProto(
                    dtype=types_pb2.DT_FLOAT, float_val=[0.25])),
                summary_pb2.Summary.Value(tag="invalid", simple_value=float("nan")),
            ])))
            writer.close()
            path = next(Path(folder).glob("events.out.tfevents.*"))
            with path.open("ab") as stream:
                stream.write(b"\x01\x02")
            series = ssplot.read_scalars(path)
            self.assertEqual(series["loss/current"], [(1, 0.5)])
            self.assertEqual(series["tensor_scalar"], [(1, 0.25)])
            self.assertNotIn("invalid", series)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ssplot.main([str(path)]), 0)
                self.assertEqual(ssplot.main([str(path)]), 0)
            output = path.with_name(path.name + ".png")
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_empty_log_error(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as folder:
            writer = EventFileWriter(folder)
            writer.close()
            path = next(Path(folder).glob("events.out.tfevents.*"))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(ssplot.main([str(path)]), 1)
            self.assertFalse(path.with_name(path.name + ".png").exists())


if __name__ == "__main__":
    unittest.main()
