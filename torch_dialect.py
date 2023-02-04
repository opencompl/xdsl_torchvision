from xdsl.ir import (Data, Dialect, MLIRType, ParametrizedAttribute, Operation, Region,
                     Attribute)
from xdsl.irdl import (irdl_attr_definition, attr_constr_coercion,
                       irdl_data_definition, irdl_to_attr_constraint,
                       irdl_op_definition, builder, ParameterDef, OpAttr, ResultDef,
                       Generic, GenericData, AttrConstraint,
                       AnyAttr, Annotated)
from xdsl.parser import BaseParser
from xdsl.printer import Printer
from xdsl.dialects import builtin
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, ArrayAttr, IntAttr, IntegerAttr, FloatAttr, AnyFloat, AnyIntegerAttr

@irdl_attr_definition
class BoolAttr(Data[bool]):
    name = "bool"

    @staticmethod
    def parse_parameter(parser: BaseParser) -> bool:
        data = parser.try_parse_boolean_literal()
        if data.text == "true":
            return True
        return False

    @staticmethod
    def print_parameter(data: bool, printer: Printer) -> None:
        printer.print_string(f'{data}')
    
    @staticmethod
    @builder
    def from_bool(data: bool):
        return BoolAttr(data)

@irdl_attr_definition
class IntegerType(ParametrizedAttribute):
    name = "torch.int"

@irdl_attr_definition
class FloatType(ParametrizedAttribute):
    name = "torch.float"

@irdl_attr_definition
class BoolType(ParametrizedAttribute):
    name = "torch.bool"

@irdl_attr_definition
class NoneType(ParametrizedAttribute):
    name = "torch.none"

@irdl_attr_definition
class VTensorType(ParametrizedAttribute):
    name = "torch.vtensor"
    dimensions: ParameterDef[ArrayAttr[AnyIntegerAttr]]
    type: ParameterDef[IntegerType | AnyFloat]

@irdl_op_definition
class ConstantIntOp(Operation):
    name = "torch.constant.int"
    value: OpAttr[IntAttr]
    res: Annotated[ResultDef, IntegerType]

@irdl_op_definition
class ConstantFloatOp(Operation):
    name = "torch.constant.float"
    value: OpAttr[FloatAttr]
    res: Annotated[ResultDef, AnyFloat]

@irdl_op_definition
class ConstantBoolOp(Operation):
    name = "torch.constant.bool"
    value: OpAttr[BoolAttr]
    res: Annotated[ResultDef, BoolType]

@irdl_op_definition
class ConstantNoneOp(Operation):
    name = "torch.constant.none"
    res: Annotated[ResultDef, NoneType]

@irdl_op_definition
class VTensorLitteralOp(Operation):
    name = "torch.vtensor.litteral"
    value: OpAttr[DenseIntOrFPElementsAttr]
    res: Annotated[ResultDef, VTensorType]

Torch = Dialect([ConstantIntOp,
                ConstantFloatOp,
                ConstantBoolOp,
                ConstantNoneOp,
                VTensorLitteralOp],
                [IntegerType,
                FloatType,
                BoolType,
                NoneType,
                VTensorType])

### PARSING

from xdsl.xdsl_opt_main import xDSLOptMain

class MyXDSLOptMain(xDSLOptMain):
    def register_all_dialects(self):
        self.ctx.register_dialect(Torch)

MyXDSLOptMain().run()
