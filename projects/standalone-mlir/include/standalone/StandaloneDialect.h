// ==- StandaloneDialect.h - 最简头文件 -==//

#ifndef STANDALONE_MLIR_DIALECT_H
#define STANDALONE_MLIR_DIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// TableGen 生成的 dialect 和 op 声明
#include "standalone/StandaloneDialect.h.inc"
#include "standalone/StandaloneOps.h.inc"

namespace mlir::standalone {
void registerStandalonePasses();
} // namespace mlir::standalone

#endif
