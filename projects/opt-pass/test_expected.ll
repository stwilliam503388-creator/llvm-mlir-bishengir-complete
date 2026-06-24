define i32 @has_dead_code() {
entry:
  ret i32 42
}

define i32 @all_used(i32 %a) {
entry:
  %x = add i32 %a, 1
  ret i32 %x
}
