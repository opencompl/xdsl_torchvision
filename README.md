# Torchvision XDSL

Installing torch-mlir directly from requirements.txt may not work since it is
a pre-release. For now, it can be installed using the command:

```
pip --pre torch-mlir torchvision -f https://github.com/llvm/torch-mlir/releases/expanded_assets/snapshot-20221026.638 --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```
