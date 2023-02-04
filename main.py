from torchvision.models import alexnet

import torch
import torch_mlir


def make_model(model: torch.nn.Module, example_inputs: torch.Tensor):
    with torch.no_grad():
        model.eval()
    module = torch_mlir.compile(
        model,
        example_inputs,
        use_tracing=True,
        output_type=torch_mlir.OutputType.TORCH,
    )
    return module


example_input = torch.randn(1, 3, 224, 224)
mlir_module = make_model(alexnet(), example_input)
mlir_str = str(
    mlir_module.operation.get_asm(
        print_generic_op_form=True, large_elements_limit=10
    )
)
print(mlir_str)
