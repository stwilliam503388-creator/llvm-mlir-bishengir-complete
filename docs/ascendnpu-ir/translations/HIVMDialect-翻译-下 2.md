### `hivm.hir.vgather` (hivm::VGatherOp)

_向量收集操作_

Syntax:

```mlir
operation ::= `hivm.hir.vgather` attr-dict `ins` `(` $src `:` type($src) `)`
              `indices` `(` $indices `:` type($indices) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
```

Retrieve elements from a tensor/memref according to given indices, 
and store these elements in another tensor/memref.
The gather axis is the last dimension.

参数：

  * `src`：从其收集元素的tensor/memref
  * `indices`：从src收集元素的索引
  * `dst`：存储元素的tensor/memref
  * `temp_buffer`：gather操作所需的额外内存

Traits: `AlwaysSpeculatableImplTrait`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | Tensor or Memref
| `indices` | Tensor or Memref
| `dst` | Tensor or Memref
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vinterleave` (hivm::VInterleaveOp)

_向量交织操作_

Syntax:

```mlir
operation ::= `hivm.hir.vinterleave` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `interleave_channel_nums` `=` $interleave_channel_nums
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
```

Interleaves the values of `N` tensors along their last dimension.
All tensors must have the same shape.

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>interleave_channel_nums</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of any type
| `dst` | Tensor or Memref
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vln` (hivm::VLnOp)

_逐元素向量自然对数操作_

Syntax:

```mlir
operation ::= `hivm.hir.vln` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入/初始化操作数与结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of shaped of any type values
| `dst` | variadic of shaped of any type values
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vmax` (hivm::VMaxOp)

_逐元素二元向量最大值操作_

Syntax:

```mlir
operation ::= `hivm.hir.vmax` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入/初始化操作数与结果具有相同的元素类型。
  2. 同时支持向量-向量和向量-标量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of any type
| `dst` | variadic of shaped of any type values
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vmin` (hivm::VMinOp)

_逐元素二元向量最小值操作_

Syntax:

```mlir
operation ::= `hivm.hir.vmin` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入/初始化操作数与结果具有相同的元素类型。
  2. 同时支持向量-向量和向量-标量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of any type
| `dst` | variadic of shaped of any type values
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vmod` (hivm::VModOp)

_逐元素向量取模操作_

Syntax:

```mlir
operation ::= `hivm.hir.vmod` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入/初始化操作数与结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of any type
| `dst` | variadic of shaped of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vmul` (hivm::VMulOp)

_逐元素二元向量乘法操作_

Syntax:

```mlir
operation ::= `hivm.hir.vmul` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入/初始化操作数与结果具有相同的元素类型。
  2. 同时支持向量-向量和向量-标量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of any type
| `dst` | variadic of shaped of any type values
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vmulext` (hivm::VMulExtOp)

_Elementwise Binary Vector Multiplication that Calculates
    the Most Significant 32-bits._

Syntax:

```mlir
operation ::= `hivm.hir.vmulext` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入/初始化操作数与结果具有相同的元素类型。
  2. 支持向量-向量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of any type
| `dst` | variadic of shaped of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vmulextended` (hivm::VMulextendedOp)

_向量扩展乘法操作_

Syntax:

```mlir
operation ::= `hivm.hir.vmulextended` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
```

对两个张量执行vmul操作。同时获取高16位和低16位。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of Tensor or Memref
| `dst` | variadic of Tensor or Memref
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vnot` (hivm::VNotOp)

_逐元素向量非操作_

Syntax:

```mlir
operation ::= `hivm.hir.vnot` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入/初始化操作数与结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of shaped of any type values
| `dst` | variadic of shaped of any type values
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vor` (hivm::VOrOp)

_逐元素二元向量或操作_

Syntax:

```mlir
operation ::= `hivm.hir.vor` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入/初始化操作数与结果具有相同的元素类型。
  2. 仅支持向量-向量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`, `VectorOnlyTrait<1>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of any type
| `dst` | variadic of shaped of any type values
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vpad` (hivm::VPadOp)

_向量填充操作_

Syntax:

```mlir
operation ::= `hivm.hir.vpad` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `low` `` custom<DynamicIndexList>($low, $static_low)
              `high` `` custom<DynamicIndexList>($high, $static_high)
              `pad_value` $pad_value `:` type($pad_value)
              (`->` type($result)^)?
```

Pads the input operand. Operation semantic is similar to
`tensor.pad`.

参数：

  * `src`：要填充值的tensor/memref
  * `dst`：为bufferization保留
  * `pad_value`：填充值
  * `low`：沿每个维度起点的填充长度
  * `high`：沿每个维度终点的填充长度

示例：

```mlir
hivm.hir.vpad ins(%src : tensor<2x16xf32>) outs(%dst: tensor<?x16xf32>)
              low[%first_dim_low, 0] high[%first_dim_high, 0]
              pad_value %pad_value : f32
                -> tensor<?x16xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `BiShengIRAggregatedOpInterface`, `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>static_low</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>static_high</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | Tensor or Memref
| `dst` | Tensor or Memref
| `pad_value` | any type
| `low` | variadic of index
| `high` | variadic of index

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vpow` (hivm::VPowOp)

_逐元素二元向量幂操作_

Syntax:

```mlir
operation ::= `hivm.hir.vpow` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入/初始化操作数与结果具有相同的元素类型。
  2. 同时支持向量-向量和向量-标量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`, `VectorOnlyTrait<1>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of any type
| `dst` | variadic of shaped of any type values
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vrec` (hivm::VRecOp)

_逐元素向量倒数操作_

Syntax:

```mlir
operation ::= `hivm.hir.vrec` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入/初始化操作数与结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of shaped of any type values
| `dst` | variadic of shaped of any type values
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vreduce` (hivm::VReduceOp)

_向量规约操作_

Syntax:

```mlir
operation ::= `hivm.hir.vreduce` attr-dict $arith `ins` `(` $src `:` type($src) `)`
              (`indices` `(` $indices^ `:` type($indices) `)`)?
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              `reduce_dims` `=` $reduce_dims
              (`->` type($result)^)?
```

Reduce one or more axes of the source vector according to
the reduction axes array, starting from an init value.

约束：

  1. The input vector and output vector must have the same rank
     and the same element type.
  2. 对于输出操作数，规约轴的大小必须为1。
  3. The reduction indices array can not be empty,
     nor can be larger than the ranks of the input vector.
  4. 规约索引必须在`[0, RankOfDstVec)`范围内。

示例：

```mlir
hivm.hir.vreduce <add> ins(%src : memref<?xf32>) outs(%dst : memref<1xf32>) reduce_dims : [1]
%result = hivm.hir.vreduce <max> ins(%src : tensor<?xf32>) outs(%dst : tensor<1xf32>) reduce_dims : [0] -> tensor<1xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `BiShengIRAggregatedOpInterface`, `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>arith</code></td><td>::mlir::hivm::ReduceOpAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM规约算术操作属性。
  {{% /markdown %}}</details></td></tr>
<tr><td><code>reduce_dims</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | Tensor or Memref
| `dst` | variadic of Tensor or Memref
| `temp_buffer` | memref of any type values
| `indices` | Tensor or Memref

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vrelu` (hivm::VReluOp)

_逐元素向量ReLU（线性整流单元）操作_

Syntax:

```mlir
operation ::= `hivm.hir.vrelu` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入/初始化操作数与结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of shaped of any type values
| `dst` | variadic of shaped of any type values
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vrsqrt` (hivm::VRsqrtOp)

_逐元素向量平方根倒数操作_

Syntax:

```mlir
operation ::= `hivm.hir.vrsqrt` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入/初始化操作数与结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of shaped of any type values
| `dst` | variadic of shaped of any type values
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vsel` (hivm::VSelOp)

_逐元素向量选择操作_

Syntax:

```mlir
operation ::= `hivm.hir.vsel` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

Select elements from two source vector according to the binary `condition` vector.
If the corresponding bit of the indicator is 1, select `src0`. Otherwise,
select `src1`.

额外约束：

  1. The input vectors and output vector must have the same ranks.
  2. 指示向量的元素类型必须为bool。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<3>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of any type
| `dst` | variadic of shaped of any type values
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vshl` (hivm::VShLOp)

_逐元素二元向量左移操作_

Syntax:

```mlir
operation ::= `hivm.hir.vshl` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入向量和结果具有相同的元素类型。
  2. 仅支持向量-标量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `ScalarOnlyHWTrait<1>`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of any type
| `dst` | variadic of shaped of any type values
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vshr` (hivm::VShROp)

_逐元素二元向量右移操作_

Syntax:

```mlir
operation ::= `hivm.hir.vshr` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`round` `:` $round^ )?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入向量和结果具有相同的元素类型。
  2. 仅支持向量-标量操作。
  3. If `round` is set to true, rounding is applied during arithmetic
     shift right.

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `ScalarOnlyHWTrait<1>`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>round</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of any type
| `dst` | variadic of shaped of any type values
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vsin` (hivm::VSinOp)

_逐元素向量正弦操作_

Syntax:

```mlir
operation ::= `hivm.hir.vsin` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入/初始化操作数与结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of any type
| `dst` | variadic of shaped of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vsort` (hivm::VSortOp)

_向量排序操作_

Syntax:

```mlir
operation ::= `hivm.hir.vsort` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `descending` `=` $descending
              `sort_axis` `=` $sort_axis
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
```

Sort the sorting axis of `src` in ascending or descending order, and output
the sorted value and the index corresponding to the value.

约束：

  1. 输入向量和输出向量必须具有相同的秩。
  2. 当前仅支持尾轴排序。

参数：

  * `src`：要排序的tensor/memref
  * `dst_value`：存储排序后值的tensor/memref
  * `dst_index`：存储与dst_value对应索引的tensor/memref
  * `descending`: determines whether to sort in ascending or descending
                  order. The default is false, which means ascending order
  * `sort_axis`：要排序的轴

示例：

```mlir
hivm.hir.vsort ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) descending = true sort_axis = 0
%result = hivm.hir.vsort ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) descending = true sort_axis = 0 -> tensor<?xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `VectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>descending</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>sort_axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | Tensor or Memref
| `dst` | variadic of Tensor or Memref
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vsqrt` (hivm::VSqrtOp)

_逐元素向量平方根操作_

Syntax:

```mlir
operation ::= `hivm.hir.vsqrt` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入/初始化操作数与结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of shaped of any type values
| `dst` | variadic of shaped of any type values
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vsub` (hivm::VSubOp)

_逐元素二元向量减法操作_

Syntax:

```mlir
operation ::= `hivm.hir.vsub` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入/初始化操作数与结果具有相同的元素类型。
  2. 同时支持向量-向量和向量-标量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of any type
| `dst` | variadic of shaped of any type values
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vtanh` (hivm::VTanhOp)

_逐元素向量双曲正切操作_

Syntax:

```mlir
operation ::= `hivm.hir.vtanh` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入/初始化操作数与结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of any type
| `dst` | variadic of shaped of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vtranspose` (hivm::VTransposeOp)

_向量转置操作_

Syntax:

```mlir
operation ::= `hivm.hir.vtranspose` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`permutation` `=` $permutation^)?
              (`disable_align` `=` $disable_align^)?
              (`->` type($result)^)?
```

Permutes the dimensions of 'src' according to the given `permutation`. In
other words:
  `dim(dst, i) = dim(src, permutation[i])`.

约束：

  1. 输入向量和输出向量必须具有相同的秩和相同的元素类型。

示例：

```mlir
 hivm.hir.vtranspose ins(%src : memref<32x8xf32>) outs(%dst : memref<8x32xf32>) permutation = [1, 0]
 %result = hivm.hir.vtranspose ins(%src : tensor<32x8xf32>) outs(%dst: tensor<8x32xf32>) permutation = [1, 0] -> tensor<8x32xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `BiShengIRAggregatedOpInterface`, `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>permutation</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>disable_align</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | Tensor or Memref
| `dst` | Tensor or Memref
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.vxor` (hivm::VXorOp)

_逐元素二元向量异或操作_

Syntax:

```mlir
operation ::= `hivm.hir.vxor` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*From the Elementwise Nary Vector Op template:*

This operation performs element-wise operation on N operands and produces a single result.
It may perform either transpose or broadcast along the way (but not both).

通用约束：

  1. 遵循DestinationStyleOpInterface。
  2. 输入操作数为N个；输出/结果数为一个。
  3. 输入/初始化操作数与结果具有相同的秩。
  4. 第一个输入仅为向量。

额外约束：

  1. 输入/初始化操作数与结果具有相同的元素类型。
  2. 仅支持向量-向量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`, `VectorOnlyTrait<1>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `src` | variadic of shaped of any type values
| `dst` | variadic of shaped of any type values
| `temp_buffer` | memref of any type values

#### Results

| Result | Description |
| :----: | ----------- |
| `result` | variadic of ranked tensor of any type values

### `hivm.hir.wait_flag` (hivm::WaitFlagOp)

_HIVM 等待标志。_

Syntax:

```mlir
operation ::= `hivm.hir.wait_flag` `[`
              $set_pipe
              `,` $wait_pipe
              `,` custom<EventID>($static_event_id, $dynamic_event_id)
              `]` attr-dict
```

Interfaces: `InferCoreTypeInterface`

#### Attributes

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>set_pipe</code></td><td>::mlir::hivm::PipeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM操作管道属性。
  {{% /markdown %}}</details></td></tr>
<tr><td><code>wait_pipe</code></td><td>::mlir::hivm::PipeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM操作管道属性。
  {{% /markdown %}}</details></td></tr>
<tr><td><code>static_event_id</code></td><td>::mlir::hivm::EventAttr</td><td><details><summary></summary>{{% markdown %}}
    用于同步的HIVM事件属性。
  {{% /markdown %}}</details></td></tr>
</table>

#### Operands

| Operand | Description |
| :-----: | ----------- |
| `dynamic_event_id` | 64-bit signless integer

## 属性

### AddressSpaceAttr

Syntax:

```mlir
#hivm.address_space<
  ::mlir::hivm::AddressSpace   # address_space
>
```

HIVM地址空间映射属性。映射到GM、L1、L0A、L0B、L0C和UB。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| address_space | `::mlir::hivm::AddressSpace` | an enum of type AddressSpace |

### AlignKindAttr

对齐类型信息

Syntax:

```mlir
#hivm.align_kind<
  ::mlir::hivm::AlignKind   # value
>
```

HIVM对齐类型属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::AlignKind` | an enum of type AlignKind |

### AllocAlignDimsAttr

Syntax: `#hivm.alloc_align_dims`

HIVM分配对齐维度。

### AllocAlignValueInByteAttr

Syntax: `#hivm.alloc_align_value_in_byte`

HIVM按字节的分配对齐值。

### AtomicKindAttr

StoreOp的原子操作类型

Syntax:

```mlir
#hivm.atomic_kind<
  ::mlir::hivm::AtomicKind   # value
>
```

HIVM原子存储类型属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::AtomicKind` | an enum of type AtomicKind |

### AxisKindAttr

HIVM操作轴类型信息

Syntax:

```mlir
#hivm.axis_kind<
  ::mlir::hivm::AxisKind   # value
>
```

HIVM操作轴类型属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::AxisKind` | an enum of type AxisKind |

### HIVMBlockMappingAttr

Syntax:

```mlir
#hivm.block<
  std::optional<int32_t>   # order
>
```

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| order | `std::optional<int32_t>` |  |

### CompareModeAttr

VCmpOp的比较模式

Syntax:

```mlir
#hivm.compare_mode<
  ::mlir::hivm::CompareMode   # value
>
```

HIVM比较模式属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::CompareMode` | an enum of type CompareMode |

### DCCIModeAttr

HIVM DCCI模式

Syntax:

```mlir
#hivm.DCCIMode<
  ::mlir::hivm::DCCIMode   # value
>
```

HIVM DCCI模式属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::DCCIMode` | an enum of type DCCIMode |

### DataCacheKindAttr

HIVM数据缓存类型

Syntax:

```mlir
#hivm.DataCacheKind<
  ::mlir::hivm::DataCacheKind   # value
>
```

HIVM数据缓存类型属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::DataCacheKind` | an enum of type DataCacheKind |

### DataLayoutAttr

Syntax:

```mlir
#hivm.data_layout<
  ::mlir::hivm::DataLayout,   # data_layout
  std::optional<bool>,   # transpose
  std::optional<DenseI64ArrayAttr>   # fractalSizes
>
```

HIVM数据布局 mapping attribute. Maps to DOTA_ND, DOTB_ND, DOTC_ND, zN, nZ and ND.

  - `transpose`：指示布局已转置。
                 仅对DOTA_ND和DOTB_ND布局有效且必须存在。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| data_layout | `::mlir::hivm::DataLayout` | an enum of type DataLayout |
| transpose | `std::optional<bool>` |  |
| fractalSizes | `std::optional<DenseI64ArrayAttr>` |  |

### DeinterleaveModeAttr

HIVM解交织模式

Syntax:

```mlir
#hivm.deinterleave_mode<
  ::mlir::hivm::DeinterleaveMode   # value
>
```

HIVM解交织索引模式

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::DeinterleaveMode` | an enum of type DeinterleaveMode |

### DescaleModeAttr

matmul的反缩放模式

Syntax:

```mlir
#hivm.descale_mode<
  ::mlir::hivm::DescaleMode   # value
>
```

matmul操作的HIVM反缩放模式属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::DescaleMode` | an enum of type DescaleMode |

### DisableAutoInjectBlockSyncAttr

Syntax: `#hivm.disable_auto_inject_block_sync`

禁用自动注入块同步，跳过块同步注入。

### EventAttr

Syntax:

```mlir
#hivm.event<
  ::mlir::hivm::EVENT   # event
>
```

用于同步的HIVM事件属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| event | `::mlir::hivm::EVENT` | an enum of type EVENT |

### FixpipePreQuantModeAttr

HIVM fixpipe预量化模式

Syntax:

```mlir
#hivm.fixpipe_pre_quant_mode<
  ::mlir::hivm::FixpipePreQuantMode   # value
>
```

HIVM fixpipe预量化模式

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::FixpipePreQuantMode` | an enum of type FixpipePreQuantMode |

### FixpipePreReluModeAttr

HIVM fixpipe预ReLU模式

Syntax:

```mlir
#hivm.fixpipe_pre_relu_mode<
  ::mlir::hivm::FixpipePreReluMode   # value
>
```

HIVM fixpipe预ReLU模式

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::FixpipePreReluMode` | an enum of type FixpipePreReluMode |

### HIVMFuncDynMemrefArgsAttr

Syntax: `#hivm.func_dyn_memref_args`

HIVM FuncDynMemrefArgs用于标记函数动态memref参数的索引数组


### InsertSliceSourceIndexAttr

Syntax: `#hivm.insert_slice_source_index`

指定vconcat操作中哪个操作数是insert_slice源

### MultiBufferAttr

Syntax: `#hivm.multi_buffer`

HIVM多缓冲属性。

### PadModeAttr

Syntax:

```mlir
#hivm.padmode<
  ::mlir::hivm::PadMode   # padmode
>
```

HIVM填充模式属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| padmode | `::mlir::hivm::PadMode` | an enum of type PadMode |

### ParallelLoopAttr

Syntax: `#hivm.parallel_loop`

标记的循环可以并行执行。

### PipeAttr

Syntax:

```mlir
#hivm.pipe<
  ::mlir::hivm::PIPE   # pipe
>
```

HIVM操作管道属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| pipe | `::mlir::hivm::PIPE` | an enum of type PIPE |

### ReduceOpAttr

Syntax:

```mlir
#hivm.reduce_op<
  ::mlir::hivm::ReduceOperation   # reduce_op
>
```

HIVM规约算术操作属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| reduce_op | `::mlir::hivm::ReduceOperation` | an enum of type ReduceOperation |

### RoundModeAttr

VCastOp的舍入模式

Syntax:

```mlir
#hivm.round_mode<
  ::mlir::hivm::RoundMode   # value
>
```

- RINT：舍入到最近偶数（C语言rint）
- ROUND：舍入到最近，远离零（C语言round）
- FLOOR：向负无穷舍入（C语言floor）
- CEIL：向正无穷舍入（C语言ceil）
- TRUNC：向零舍入（C语言trunc）
- ODD：舍入到奇数（冯·诺依曼舍入）

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::RoundMode` | an enum of type RoundMode |

### StorageAlignedAttr

Syntax: `#hivm.storage_aligned`

如果模块标记了此属性，则表示该模块内所有设备函数中的所有操作都已对齐。

如果函数标记了此属性，则表示该函数中的所有操作都已对齐。


### StrideAlignDimsAttr

Syntax: `#hivm.stride_align_dims`

HIVM步幅对齐维度。

### StrideAlignValueInByteAttr

Syntax: `#hivm.stride_align_value_in_byte`

HIVM按字节的步幅对齐值。

### HIVMSubBlockMappingAttr

Syntax:

```mlir
#hivm.sub_block<
  ::mlir::hivm::MappingId   # sub_block
>
```

混合函数cv块维度比例的HIVM子块映射属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| sub_block | `::mlir::hivm::MappingId` | an enum of type MappingId |

### SyncBlockInstrModeAttr

Syntax:

```mlir
#hivm.sync_block_instr_mode<
  ::mlir::hivm::SyncBlockInstrMode   # sync_instr_mode
>
```

HIVM同步块指令模式属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| sync_instr_mode | `::mlir::hivm::SyncBlockInstrMode` | an enum of type SyncBlockInstrMode |

### SyncBlockModeAttr

Syntax:

```mlir
#hivm.sync_block_mode<
  ::mlir::hivm::SyncBlockMode   # sync_mode
>
```

HIVM同步块模式属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| sync_mode | `::mlir::hivm::SyncBlockMode` | an enum of type SyncBlockMode |

### TCoreTypeAttr

Syntax:

```mlir
#hivm.tcore_type<
  ::mlir::hivm::TCoreType   # tcoretype
>
```

HIVM操作核心类型属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| tcoretype | `::mlir::hivm::TCoreType` | an enum of type TCoreType |

### TCoreTypeMarkerAttr

Syntax:

```mlir
#hivm.tcore_type_marker<
  ::mlir::hivm::TCoreType   # tcoretype
>
```

HIVM操作核心类型标记属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| tcoretype | `::mlir::hivm::TCoreType` | an enum of type TCoreType |

### TFuncCoreTypeAttr

Syntax:

```mlir
#hivm.func_core_type<
  ::mlir::hivm::TFuncCoreType   # funcCoreType
>
```

HIVM函数核心类型属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| funcCoreType | `::mlir::hivm::TFuncCoreType` | an enum of type TFuncCoreType |

### TModuleCoreTypeAttr

Syntax:

```mlir
#hivm.module_core_type<
  ::mlir::hivm::TModuleCoreType   # moduleCoreType
>
```

HIVM模块核心类型属性。

如果模块内所有函数都具有`AIV`函数核心类型，则模块核心类型为`AIV`。
module core type is `AIV`.

如果模块内所有函数都具有`AIC`函数核心类型，则模块核心类型为`AIC`。
module core type is `AIC`.

否则，模块核心类型为`MIX`。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| moduleCoreType | `::mlir::hivm::TModuleCoreType` | an enum of type TModuleCoreType |

### TPartOfMixAttr

Syntax: `#hivm.part_of_mix`

HIVM函数是混合内核的一部分。

### TypeFnAttr

VCastOp的转换

Syntax:

```mlir
#hivm.cast<
  ::mlir::hivm::TypeFn   # value
>
```

HIVM转换属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::TypeFn` | an enum of type TypeFn |

### UnitFlagAttr

Syntax:

```mlir
#hivm.unit_flag<
  ::mlir::hivm::UNIT_FLAG   # unit_flag
>
```

用于同步的HIVM单元标志属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| unit_flag | `::mlir::hivm::UNIT_FLAG` | an enum of type UNIT_FLAG |

### UnlikelyConditionAttr

Syntax: `#hivm.unlikely_condition`

标记的条件不太可能求值为true。

### VFModeAttr

HIVM VF模式

Syntax:

```mlir
#hivm.vf_mode<
  ::mlir::hivm::VFMode   # value
>
```

HIVM VF模式属性。

#### Parameters

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::VFMode` | an enum of type VFMode |

## 枚举

### AddressSpace

HIVM地址空间

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| Zero | `0` | zero |
| GM | `1` | gm |
| L1 | `2` | cbuf |
| L0A | `3` | ca |
| L0B | `4` | cb |
| L0C | `5` | cc |
| UB | `6` | ub |

### AlignKind

对齐类型信息

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| ALIGN | `0` | align |
| UNALIGNED | `1` | unaligned |
| UNKNOWN | `2` | unknown |

### AtomicKind

StoreOp的原子操作类型

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| NONE | `0` | none |
| ADD | `1` | add |
| MAX | `2` | max |
| MIN | `3` | min |
| AND | `4` | and |
| OR | `5` | or |
| XOR | `6` | xor |
| CAS | `7` | or |
| XCHG | `8` | xor |
| UMAX | `9` | umax |
| UMIN | `10` | umin |

### AxisKind

HIVM操作轴类型信息

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| FIRST | `0` | first |
| MIDDLE | `1` | middle |
| LAST | `2` | last |

### CompareMode

VCmpOp的比较模式

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| EQ | `0` | eq |
| NE | `1` | ne |
| LT | `2` | lt |
| GT | `3` | gt |
| GE | `4` | ge |
| LE | `5` | le |

### DCCIMode

HIVM DCCI模式

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| SINGLE_CACHE_LINE | `0` | single_cache_line |
| ALL_CACHE_LINES | `1` | all_cache_lines |

### DataCacheKind

HIVM数据缓存类型

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| ALL | `0` | all |
| UB | `1` | ub |
| OUT | `2` | out |
| ATOMIC | `3` | atomic |

### DataLayout

HIVM数据布局

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| DOTA_ND | `1` | dotA_ND |
| DOTB_ND | `2` | dotB_ND |
| DOTC_ND | `3` | dotC_ND |
| nZ | `4` | nZ |
| zN | `5` | zN |
| ND | `6` | ND |

### DeinterleaveMode

HIVM解交织模式

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| CHANNEL_0 | `0` | CHANNEL_0 |
| CHANNEL_1 | `1` | CHANNEL_1 |
| ALL_CHANNELS | `999` | ALL_CHANNELS |

### DescaleMode

matmul的反缩放模式

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| DescaleNull | `0` | DescaleNull |
| DescalePerChannel | `1` | DescalePerChannel |
| DescalePerTensor | `2` | DescalePerTensor |

### EVENT

同步事件ID

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| EVENT_ID0 | `0` | EVENT_ID0 |
| EVENT_ID1 | `1` | EVENT_ID1 |
| EVENT_ID2 | `2` | EVENT_ID2 |
| EVENT_ID3 | `3` | EVENT_ID3 |
| EVENT_ID4 | `4` | EVENT_ID4 |
| EVENT_ID5 | `5` | EVENT_ID5 |
| EVENT_ID6 | `6` | EVENT_ID6 |
| EVENT_ID7 | `7` | EVENT_ID7 |

### FixpipePreQuantMode

HIVM fixpipe预量化模式

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| NO_QUANT | `0` | NO_QUANT |
| S322I8 | `9` | S322I8 |
| F322F16 | `1` | F322F16 |
| F322BF16 | `16` | F322BF16 |

### FixpipePreReluMode

HIVM fixpipe预ReLU模式

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| NO_RELU | `0` | NO_RELU |
| NORMAL_RELU | `1` | NORMAL_RELU |
| LEAKY_RELU | `2` | LEAKY_RELU |
| P_RELU | `3` | P_RELU |

### IteratorType

HIVM结构化操作迭代器类型

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| kParallel | `0` | parallel |
| kBroadcast | `1` | broadcast |
| kTranspose | `2` | transpose |
| kReduction | `3` | reduction |
| kInterleave | `4` | interleave |
| kDeinterleave | `5` | deinterleave |
| kInverse | `6` | inverse |
| kPad | `7` | pad |
| kConcat | `8` | concat |
| kGather | `9` | gather |
| kCumulative | `10` | cumulative |
| kOpaque | `99` | opaque |

### MatmulBiasMode

本地matmul操作的偏置模式

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| NoBias | `0` | NoBias |
| PerChannelAdd | `1` | PerChannelAdd |
| PerChannelAddWithSplitK | `2` | PerChannelAddWithSplitK |
| ElementwiseCrossLoopAdd | `4` | ElementwiseCrossLoopAdd |
| ElementwiseAdd | `3` | ElementwiseAdd |

### MemPlanMode

内存计划模式

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| LOCAL_MEM_PLAN | `0` | LOCAL_MEM_PLAN |
| GLOBAL_WORKSPACE_PLAN | `1` | GLOBAL_WORKSPACE_PLAN |

### PadMode

LoadOp的填充模式

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| PadNull | `0` | PadNull |
| PadFirstElem | `1` | PadFirstElem |
| PadValue | `2` | PadValue |

### PIPE

HIVM操作管道

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| PIPE_S | `0` | PIPE_S |
| PIPE_V | `1` | PIPE_V |
| PIPE_M | `2` | PIPE_M |
| PIPE_MTE1 | `3` | PIPE_MTE1 |
| PIPE_MTE2 | `4` | PIPE_MTE2 |
| PIPE_MTE3 | `5` | PIPE_MTE3 |
| PIPE_ALL | `6` | PIPE_ALL |
| PIPE_MTE4 | `7` | PIPE_MTE4 |
| PIPE_MTE5 | `8` | PIPE_MTE5 |
| PIPE_V2 | `9` | PIPE_V2 |
| PIPE_FIX | `10` | PIPE_FIX |
| VIRTUAL_PIPE_MTE2_L1A | `11` | VIRTUAL_PIPE_MTE2_L1A |
| VIRTUAL_PIPE_MTE2_L1B | `12` | VIRTUAL_PIPE_MTE2_L1B |
| PIPE_NUM | `13` | PIPE_NUM |
| PIPE_UNASSIGNED | `99` | PIPE_UNASSIGNED |

### ReduceOperation

VReduceOp的规约类型

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| sum | `1` | sum |
| prod | `2` | prod |
| max | `3` | max |
| min | `4` | min |
| max_with_index_left | `5` | max_with_index_left |
| max_with_index_right | `6` | max_with_index_right |
| min_with_index_left | `7` | min_with_index_left |
| min_with_index_right | `8` | min_with_index_right |
| any | `9` | any |
| all | `10` | all |
| xori | `11` | xori |
| ori | `12` | ori |
| andi | `13` | andi |
| none | `0` | none |

### RoundMode

VCastOp的舍入模式

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| RINT | `0` | rint |
| ROUND | `1` | round |
| FLOOR | `2` | floor |
| CEIL | `3` | ceil |
| TRUNC | `4` | trunc |
| ODD | `5` | odd |
| TRUNCWITHOVERFLOW | `6` | truncwithoverflow |

### SyncBlockInstrMode

HIVM同步块指令模式

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| INTER_BLOCK_SYNCHRONIZATION | `0` | INTER_BLOCK_SYNCHRONIZATION |
| INTER_SUBBLOCK_SYNCHRONIZATION | `1` | INTER_SUBBLOCK_SYNCHRONIZATION |
| INTRA_BLOCK_SYNCHRONIZATION | `2` | INTRA_BLOCK_SYNCHRONIZATION |

### SyncBlockMode

HIVM同步块模式

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| ALL_CUBE | `0` | ALL_CUBE |
| ALL_VECTOR | `1` | ALL_VECTOR |
| ALL_SUB_VECTOR | `2` | ALL_SUB_VECTOR |
| BARRIER_CUBE | `3` | BARRIER_CUBE |
| BARRIER_VECTOR | `4` | BARRIER_VECTOR |
| ALL | `5` | ALL |

### TCoreType

HIVM操作核心类型

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| CUBE | `1` | CUBE |
| VECTOR | `2` | VECTOR |
| CUBE_OR_VECTOR | `3` | CUBE_OR_VECTOR |
| CUBE_AND_VECTOR | `4` | CUBE_AND_VECTOR |

### TFuncCoreType

HIVM函数核心类型

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| AIC | `1` | AIC |
| AIV | `2` | AIV |
| MIX | `3` | MIX |
| AIC_OR_AIV | `4` | AIC_OR_AIV |

### TModuleCoreType

HIVM模块核心类型

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| AIC | `1` | AIC |
| AIV | `2` | AIV |
| MIX | `3` | MIX |

### TypeFn

VCastOp的转换

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| cast_signed | `0` | cast_signed |
| cast_unsigned | `1` | cast_unsigned |
| bitcast | `2` | bitcast |

### UNIT_FLAG

同步单元标志模式

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| DISABLED | `0` | DISABLED |
| RESERVED | `1` | RESERVED |
| ENABLED_WITHOUT_UPDATE | `2` | ENABLED_WITHOUT_UPDATE |
| ENABLED_WITH_UPDATE | `3` | ENABLED_WITH_UPDATE |
| ENABLED_ONLY_LAST_ITER | `4` | ENABLED_ONLY_LAST_ITER |
| ENABLED_ONLY_FIRST_ITER | `5` | ENABLED_ONLY_FIRST_ITER |
| ENABLED_ONLY_FIRST_AND_LAST_ITERS | `6` | ENABLED_ONLY_FIRST_AND_LAST_ITERS |

### VFMode

HIVM VF模式

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| SIMD | `0` | SIMD |
| SIMT | `1` | SIMT |
| MIX | `2` | MIX |

### MappingId

循环映射的映射ID

#### 枚举值

| Symbol | Value | String |
| :----: | :---: | ------ |
| DimX | `0` | x |
