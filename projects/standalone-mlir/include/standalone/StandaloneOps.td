// ==- StandaloneOps.td - Standalone dialect 完整定义 -==//

#ifndef STANDALONE_MLIR_TD
#define STANDALONE_MLIR_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// ═══════════════════════════════════════════════════════════════════════════
// 1. Dialect
// ═══════════════════════════════════════════════════════════════════════════
def Standalone_Dialect : Dialect {
  let name = "standalone";
  let summary = "Standalone MLIR dialect";
  let cppNamespace = "::mlir::standalone";
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Op 基类
// ═══════════════════════════════════════════════════════════════════════════
class Standalone_Op<string mnemonic, list<Trait> traits = []>
    : Op<Standalone_Dialect, mnemonic, traits>;

// ═══════════════════════════════════════════════════════════════════════════
// 3. Operations（使用 AnyType 简化类型系统，避免 TypeDef 复杂依赖）
// ═══════════════════════════════════════════════════════════════════════════

def ConstantOp : Standalone_Op<"constant"> {
  let summary = "constant tensor";
  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyType:$output);
  let assemblyFormat = "`{` $value `}` attr-dict `:` type($output)";
}

def AddOp : Standalone_Op<"add"> {
  let summary = "element-wise addition";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def MulOp : Standalone_Op<"mul"> {
  let summary = "element-wise multiplication";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def TransposeOp : Standalone_Op<"transpose"> {
  let summary = "matrix transpose";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = "`(` $input `)` attr-dict `:` type($input) `->` type($output)";
}

def PrintOp : Standalone_Op<"print"> {
  let summary = "print tensor";
  let arguments = (ins AnyType:$input);
  let assemblyFormat = "`(` $input `)` attr-dict `:` type($input)";
}

def ReturnOp : Standalone_Op<"return"> {
  let summary = "return from function";
  let arguments = (ins Optional<AnyType>:$input);
  let assemblyFormat = "attr-dict ($input^ `:` type($input))?";
}

#endif
