import threading
import unittest

import numpy as np

from homesense.camera import CameraWorker, Frame, LatestFrameBuffer


def frame(number: int) -> Frame:
    return Frame(number, float(number), np.full((2, 2), number, dtype=np.uint8))


class BufferTests(unittest.TestCase):
    def test_replacement_and_image_ownership(self) -> None:
        buffer = LatestFrameBuffer()
        first = frame(1)
        buffer.publish(first)
        first.image[:] = 9
        self.assertTrue(np.all(buffer.wait_for_newer().image == 1))
        buffer.publish(frame(2))
        result = buffer.wait_for_newer(0)
        self.assertEqual(result.frame_id, 2)
        result.image[:] = 8
        self.assertTrue(np.all(buffer.wait_for_newer().image == 2))
        self.assertIsNone(buffer.wait_for_newer(2, timeout=0))
        with self.assertRaises(ValueError):
            buffer.publish(frame(2))

    def test_wait_wakes_for_new_frame_and_close(self) -> None:
        buffer = LatestFrameBuffer()
        results = []
        started = threading.Event()

        def wait() -> None:
            started.set()
            results.append(buffer.wait_for_newer(1, timeout=1))

        thread = threading.Thread(target=wait)
        thread.start()
        self.assertTrue(started.wait(1))
        buffer.publish(frame(1))
        buffer.publish(frame(2))
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0].frame_id, 2)
        results.clear()
        thread = threading.Thread(target=lambda: results.append(buffer.wait_for_newer(2, timeout=1)))
        thread.start()
        buffer.close()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results, [None])
        self.assertFalse(buffer.publish(frame(3)))

    def test_concurrent_reads_never_expose_partial_images(self) -> None:
        buffer = LatestFrameBuffer()
        failures = []
        def producer() -> None:
            for i in range(1, 101):
                buffer.publish(Frame(i, float(i), np.full((100, 100), i, dtype=np.uint8)))
        thread = threading.Thread(target=producer)
        thread.start()
        last = 0
        while last < 100:
            result = buffer.wait_for_newer(last, timeout=1)
            if result is None:
                failures.append("timed out")
                break
            if not np.all(result.image == result.frame_id):
                failures.append("partial frame")
            last = result.frame_id
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])


class FakeSource:
    def __init__(self, opened=True) -> None:
        self.opened = opened
        self.entered = threading.Event()
        self.unblock = threading.Event()
        self.released = False
        self.count = 0

    def isOpened(self):
        return self.opened

    def read(self):
        self.count += 1
        if self.count > 2:
            self.entered.set()
            self.unblock.wait(2)
        return True, np.full((2, 2), self.count, dtype=np.uint8)

    def release(self):
        self.released = True


class WorkerTests(unittest.TestCase):
    def test_capture_and_bounded_stop_releases_after_read(self) -> None:
        source = FakeSource()
        buffer = LatestFrameBuffer()
        worker = CameraWorker(buffer, source_factory=lambda _: source, clock=lambda: 12.0)
        worker.start()
        self.assertTrue(source.entered.wait(1))
        result = buffer.wait_for_newer(timeout=1)
        self.assertEqual((result.frame_id, result.captured_at), (2, 12.0))
        self.assertFalse(worker.stop(timeout=0.01))
        self.assertIsNone(buffer.wait_for_newer())
        source.unblock.set()
        self.assertTrue(worker.stop(timeout=1))
        self.assertTrue(source.released)
        self.assertIsNone(worker.error)
        with self.assertRaises(RuntimeError):
            worker.start()

    def test_open_failure_is_reported_and_released(self) -> None:
        source = FakeSource(opened=False)
        buffer = LatestFrameBuffer()
        worker = CameraWorker(buffer, source_factory=lambda _: source)
        worker.start()
        self.assertIsNone(buffer.wait_for_newer(timeout=1))
        self.assertTrue(worker.stop(timeout=1))
        self.assertIsInstance(worker.error, RuntimeError)
        self.assertTrue(source.released)

    def test_capture_failure_is_reported(self) -> None:
        source = FakeSource()
        source.read = lambda: (False, None)
        buffer = LatestFrameBuffer()
        worker = CameraWorker(buffer, source_factory=lambda _: source)
        worker.start()
        self.assertIsNone(buffer.wait_for_newer(timeout=1))
        self.assertTrue(worker.stop(timeout=1))
        self.assertIsInstance(worker.error, RuntimeError)
        self.assertTrue(source.released)
