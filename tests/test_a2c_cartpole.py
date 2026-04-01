import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs" / "chapter9" / "a2c_cartpole.py"


def load_module():
    spec = importlib.util.spec_from_file_location("a2c_cartpole", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class A2CCartPoleTest(unittest.TestCase):
    def test_run_experiment_smoke(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = module.run_experiment(
                output_dir=pathlib.Path(tmp_dir),
                train_episodes=3,
                eval_episodes=2,
                max_steps=50,
                hidden_dim=8,
                seed=7,
            )
            self.assertTrue(result["summary_path"].exists())
            self.assertTrue(result["train_reward_path"].exists())
            self.assertTrue(result["train_ma_reward_path"].exists())
            self.assertTrue(result["eval_reward_path"].exists())
            summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))

        self.assertEqual(len(summary["train_rewards"]), 3)
        self.assertEqual(len(summary["eval_rewards"]), 2)


if __name__ == "__main__":
    unittest.main()
