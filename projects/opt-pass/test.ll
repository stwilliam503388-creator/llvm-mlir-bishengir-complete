define i32 @has_dead_code() {
entry:
  %x = add i32 1, 2        ; 死代码：定义了但从未使用
  %y = mul i32 %x, 3       ; 死代码：x 已死，y 也用不到
  ret i32 42
}

define i32 @all_used(i32 %a) {
entry:
  %x = add i32 %a, 1        ; 活代码：被 ret 使用
  ret i32 %x
}
