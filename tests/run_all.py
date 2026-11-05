"""Run every component test in dependency order. From the project root:  python -m tests.run_all"""

import importlib
import time

MODULES = ["test_embeddings", "test_layer_norm", "test_ffn", "test_rope",
           "test_attention", "test_decoder_block", "test_transformer_model"]

start = time.time()
for name in MODULES:
    module = importlib.import_module(f"tests.{name}")
    print(f"\n=== {name} ===")
    tests = [getattr(module, f) for f in dir(module) if f.startswith("test_")]
    tests.sort(key=lambda f: f.__code__.co_firstlineno)  # run in the order they are written
    for test in tests:
        if test.__code__.co_argcount:  # parametrized LayerNorm comparison
            test(random_params=False)
            test(random_params=True)
        else:
            test()
print(f"\nALL TESTS PASSED ({time.time() - start:.0f}s)")