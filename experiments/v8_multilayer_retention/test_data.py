import unittest

import torch

from data import Batch, FixedMemory, build_audit, class_partition, make_history


def synthetic_data(classes=10, per_class=12):
    labels = torch.arange(classes).repeat_interleave(per_class).long()
    pixels = torch.arange(len(labels) * 3 * 32 * 32, dtype=torch.int64).remainder(256).to(torch.uint8)
    return pixels.reshape(-1, 3, 32, 32), labels


class TestDataStreams(unittest.TestCase):
    def test_class_partition_is_reproducible_and_disjoint(self):
        first, second = class_partition(), class_partition()
        self.assertEqual(first, second)
        self.assertEqual(set(range(100)), set(first["train"] + first["dev"] + first["audit"]))
        self.assertFalse(set(first["train"]) & set(first["dev"]))
        self.assertFalse(set(first["train"]) & set(first["audit"]))
        self.assertFalse(set(first["dev"]) & set(first["audit"]))

    def test_history_is_deterministic_and_train_monitor_are_disjoint(self):
        images, labels = synthetic_data(classes=10, per_class=12)
        one = make_history(images, labels, list(range(10)), root=7, tasks=2, classes_per_task=5,
                           train_per_class=2, monitor_per_class=1, eval_per_class=2)
        two = make_history(images, labels, list(range(10)), root=7, tasks=2, classes_per_task=5,
                           train_per_class=2, monitor_per_class=1, eval_per_class=2)
        self.assertEqual(10, len(one[0]["train"]))
        self.assertEqual(5, len(one[0]["monitor"]))
        self.assertEqual(10, len(one[0]["eval"]))
        self.assertTrue(torch.equal(one[0]["train"].ids, two[0]["train"].ids))
        all_train = set(torch.cat([part["train"].ids for part in one]).tolist())
        all_monitor = set(torch.cat([part["monitor"].ids for part in one]).tolist())
        all_eval = set(torch.cat([part["eval"].ids for part in one]).tolist())
        self.assertFalse(all_train & all_monitor)
        self.assertFalse(all_train & all_eval)
        self.assertFalse(all_monitor & all_eval)
        self.assertTrue(torch.all((one[0]["train"].x >= -1) & (one[0]["train"].x <= 1)))

    def test_audit_ids_do_not_overlap_training_ids(self):
        images, labels = synthetic_data(classes=10, per_class=12)
        history = make_history(images, labels, list(range(10)), root=3, tasks=2, classes_per_task=5,
                               train_per_class=2, monitor_per_class=1, eval_per_class=2)
        audits = build_audit(images, labels, [x["classes"] for x in history], per_class=2)
        train_ids = torch.cat([x["train"].ids for x in history])
        audit_ids = torch.cat([x.ids for x in audits])
        self.assertFalse(set(train_ids.tolist()) & set(audit_ids.tolist()))

    def test_all_streams_use_task_local_five_way_labels(self):
        images, labels = synthetic_data(classes=10, per_class=12)
        history = make_history(images, labels, [7, 2, 9, 0, 5, 1, 3, 4, 6, 8], root=4,
                               tasks=2, classes_per_task=5, train_per_class=2,
                               monitor_per_class=1, eval_per_class=2)
        audits = build_audit(images, labels, [item["classes"] for item in history], per_class=2)
        for task, item in enumerate(history):
            for stream in (item["train"], item["monitor"], item["eval"], audits[task]):
                self.assertEqual(set(range(5)), set(stream.y.tolist()))
                self.assertTrue(torch.all(stream.task == task))

    def test_every_history_prefix_is_class_balanced(self):
        images, labels = synthetic_data(classes=5, per_class=12)
        item = make_history(images, labels, list(range(5)), root=8, tasks=1, classes_per_task=5,
                            train_per_class=4, monitor_per_class=2, eval_per_class=2)[0]
        for stream in (item["train"], item["monitor"], item["eval"]):
            for end in range(1, len(stream) + 1):
                counts = torch.bincount(stream.y[:end], minlength=5)
                self.assertLessEqual(int(counts.max() - counts.min()), 1)


def batch(task, ids):
    n = len(ids)
    return Batch(torch.zeros(n, 3, 32, 32), torch.full((n,), task, dtype=torch.long),
                 torch.full((n,), task, dtype=torch.long), torch.tensor(ids, dtype=torch.long))


class TestFixedMemory(unittest.TestCase):
    def test_task_balanced_capacity_and_ref_alignment(self):
        memory = FixedMemory(replay_capacity=5, monitor_capacity=4)
        for task in range(3):
            train = batch(task, [100 * task + i for i in range(4)])
            monitor = batch(task, [100 * task + 50 + i for i in range(3)])
            memory.observe_task(task, train, monitor, torch.tensor([10 * task + i for i in range(3)]))
        replay = memory.replay()
        monitor, refs = memory.monitor()
        self.assertEqual(5, len(replay))
        self.assertEqual(4, len(monitor))
        self.assertEqual(4, len(refs))
        self.assertEqual([0., 1., 10., 20.], refs.tolist())  # quotas 2, 1, 1; prefix trim
        self.assertLessEqual(len(memory.retained_replay_ids), 5)
        self.assertLessEqual(len(memory.retained_monitor_ids), 4)

    def test_discarded_old_data_never_returns(self):
        memory = FixedMemory(replay_capacity=1, monitor_capacity=1)
        memory.observe_task(0, batch(0, [1, 2]), batch(0, [11, 12]), torch.tensor([1., 2.]))
        memory.observe_task(1, batch(1, [3, 4]), batch(1, [13, 14]), torch.tensor([3., 4.]))
        self.assertEqual([1], memory.retained_replay_ids.tolist())
        self.assertEqual([11], memory.retained_monitor_ids.tolist())
        memory.observe_task(2, batch(2, [5, 6]), batch(2, [15, 16]), torch.tensor([5., 6.]))
        self.assertNotIn(2, memory.retained_replay_ids.tolist())
        self.assertNotIn(12, memory.retained_monitor_ids.tolist())

    def test_rejects_overlapping_streams(self):
        memory = FixedMemory(2, 2)
        with self.assertRaises(ValueError):
            memory.observe_task(0, batch(0, [1, 2]), batch(0, [2, 3]), torch.tensor([1., 2.]))


if __name__ == "__main__":
    unittest.main()
